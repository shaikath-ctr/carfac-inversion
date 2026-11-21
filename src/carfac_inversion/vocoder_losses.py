"""Losses, causal Butterworth filter, and GAN discriminators for Phase 4.

Combines AFHI ``arcfac`` stabilization primitives (causal 4th-order 20 Hz output
high-pass filtering, masked warm-up energy penalties, and healthy-only
waveform/STFT regularization masking) with multi-resolution STFT and HiFi-GAN /
UnivNet discriminators for direct NAP-to-audio and perceptual auralization
training.
"""

import functools
from collections.abc import Callable
from collections.abc import Sequence
from typing import cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
from hp_acoustic.models import sai as sai_lib

# 32 kHz vocoder STFT specs matching AFHI/arcfac: 16, 32, and 64 ms windows.
STFT_RESOLUTIONS_32K: tuple[tuple[int, int, int], ...] = (
  (512, 128, 512),
  (1024, 256, 1024),
  (2048, 512, 2048),
)
DISCRIMINATOR_PERIODS: tuple[int, ...] = (2, 3, 5, 7, 11)
_DISCRIMINATOR_LEAK = 0.1
_SAI_HYPERS = sai_lib.SaiHypers()


def stft_resolutions_for_rate(
  sample_rate_hz: float,
) -> tuple[tuple[int, int, int], ...]:
  """Returns (n_fft, hop, win_length) specs scaled cleanly for sample rate."""
  if int(round(sample_rate_hz)) == 32000:
    return STFT_RESOLUTIONS_32K
  scale = max(0.25, float(sample_rate_hz) / 32000.0)
  out: list[tuple[int, int, int]] = []
  for n_fft, hop, win_len in STFT_RESOLUTIONS_32K:
    n = max(128, 1 << int(round(np.log2(n_fft * scale))))
    h = max(32, n // max(1, n_fft // hop))
    w = min(n, max(h, int(round(win_len * (n / n_fft)))))
    out.append((n, h, w))
  return tuple(out)


def delay_samples(
  array: jnp.ndarray,
  samples: int,
  *,
  axis: int = 1,
) -> jnp.ndarray:
  """Delays ``array`` by ``samples`` along ``axis``, zero-padding the start."""
  if samples < 0:
    raise ValueError(f"samples must be non-negative, got {samples}.")
  if samples == 0:
    return array
  norm_axis = axis if axis >= 0 else array.ndim + axis
  pads = [(0, 0)] * array.ndim
  pads[norm_axis] = (samples, 0)
  padded = jnp.pad(array, pads)
  return jax.lax.slice_in_dim(padded, 0, array.shape[norm_axis], axis=norm_axis)


def _biquad_associative(
  x: jnp.ndarray,
  b0: float,
  b1: float,
  b2: float,
  a1: float,
  a2: float,
) -> jnp.ndarray:
  """Evaluates an underdamped biquad over ``(T, ...)`` in O(log T) stages.

  Uses modal (complex conjugate 1st-order AR) coordinates so the parallel
  prefix scan has condition number 1 instead of companion-matrix ~400 near
  20 Hz @ 32 kHz, matching float64 ``scipy.signal.sosfilt`` to ~1.4e-6.
  """
  t_len = x.shape[0]
  if t_len == 0:
    return x
  orig_shape = x.shape
  x_2d = x.reshape(t_len, -1).astype(jnp.float32)

  disc = 4.0 * a2 - a1 * a1
  if disc <= 1e-24:
    raise ValueError(
      f"Expected complex conjugate Butterworth poles, got discriminant={disc}."
    )
  pr = -0.5 * a1
  pi = 0.5 * float(np.sqrt(disc))
  pole = np.complex64(pr + 1j * pi)
  d1 = b1 - a1 * b0
  d2 = b2 - a2 * b0
  ar = 0.5 * d1
  ai = (-0.5 * d2 - ar * pr) / pi
  alpha = np.complex64(ar + 1j * ai)

  p_all = jnp.full((t_len,), pole, dtype=jnp.complex64)
  u_all = x_2d.astype(jnp.complex64)

  def _combine(
    left: tuple[jnp.ndarray, jnp.ndarray],
    right: tuple[jnp.ndarray, jnp.ndarray],
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    pl, ul = left
    pr_t, ur = right
    return pr_t * pl, pr_t[:, None] * ul + ur

  _, s_next = jax.lax.associative_scan(_combine, (p_all, u_all), axis=0)
  s_in = jnp.concatenate(
    [jnp.zeros((1, x_2d.shape[1]), dtype=jnp.complex64), s_next[:-1]], axis=0
  )
  v_2d = jnp.float32(b0) * x_2d + 2.0 * jnp.real(jnp.asarray(alpha) * s_in)
  return v_2d.astype(x.dtype).reshape(orig_shape)


def causal_highpass(
  y: jnp.ndarray,
  cutoff_hz: float = 20.0,
  fs: float = 32000.0,
) -> jnp.ndarray:
  """Applies a causal 4th-order Butterworth high-pass along the last axis.

  Uses an exact O(log T) ``jax.lax.associative_scan`` over modal biquad
  sections so CARFAC's blindness below 20 Hz cannot induce drift or slow
  sequential TPU scans.
  """
  if cutoff_hz <= 0.0:
    return y
  sos = scipy.signal.butter(4, cutoff_hz, "highpass", fs=fs, output="sos")
  out = jnp.moveaxis(y, -1, 0)
  for row in np.asarray(sos, dtype=np.float64):
    b0, b1, b2, _, a1, a2 = (float(v) for v in row)
    out = _biquad_associative(out, b0, b1, b2, a1, a2)
  return jnp.moveaxis(out, 0, -1)


def _framed_rfft(
  x: jnp.ndarray,
  n_fft: int,
  hop: int,
  win_length: int,
) -> jnp.ndarray:
  """Computes reflection-padded complex Hann rFFT frames ``(batch, frames, bins)``."""
  if x.ndim != 2:
    raise ValueError(f"Expected (batch, samples), got shape {x.shape}.")
  if win_length > n_fft:
    raise ValueError(f"win_length {win_length} exceeds n_fft {n_fft}.")
  pad = n_fft // 2
  if x.shape[1] <= pad:
    raise ValueError(f"{x.shape[1]} samples is too short for n_fft {n_fft}.")
  x_pad = jnp.pad(x, ((0, 0), (pad, pad)), mode="reflect")
  frames = 1 + (x_pad.shape[1] - n_fft) // hop
  idx = jnp.arange(frames)[:, None] * hop + jnp.arange(n_fft)[None, :]
  n = jnp.arange(win_length, dtype=jnp.float32)
  window = 0.5 - 0.5 * jnp.cos(2.0 * jnp.pi * n / float(win_length))
  left = (n_fft - win_length) // 2
  window = jnp.pad(window, (left, n_fft - win_length - left))
  return jnp.fft.rfft(x_pad[:, idx] * window[None, None, :], axis=-1)


def stft_magnitude(
  x: jnp.ndarray,
  n_fft: int,
  hop: int,
  win_length: int,
  eps: float = 1e-7,
) -> jnp.ndarray:
  """Computes STFT magnitude ``(batch, frames, n_fft // 2 + 1)``."""
  spec = _framed_rfft(x, n_fft, hop, win_length)
  return jnp.sqrt(jnp.real(spec) ** 2 + jnp.imag(spec) ** 2 + eps)


@functools.lru_cache(maxsize=16)
def _mel_filterbank_np(
  sample_rate_hz: float,
  n_fft: int,
  n_mels: int,
) -> np.ndarray:
  """Builds an area-normalized triangular mel filterbank ``(bins, n_mels)``."""
  hi_hz = 0.5 * float(sample_rate_hz)
  hi_mel = 2595.0 * np.log10(1.0 + hi_hz / 700.0)
  mels = np.linspace(0.0, hi_mel, n_mels + 2)
  hz = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
  freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate_hz))
  fb = np.zeros((freqs.shape[0], n_mels), dtype=np.float32)
  for m in range(n_mels):
    f0, f1, f2 = hz[m], hz[m + 1], hz[m + 2]
    tri = np.maximum(
      0.0,
      np.minimum(
        (freqs - f0) / max(f1 - f0, 1e-8),
        (f2 - freqs) / max(f2 - f1, 1e-8),
      ),
    )
    denom = float(np.sum(tri))
    fb[:, m] = (tri / denom if denom > 1e-8 else tri).astype(np.float32)
  return fb


def multi_scale_mel_loss(
  y: jnp.ndarray,
  x: jnp.ndarray,
  *,
  sample_rate_hz: float = 32000.0,
  specs: Sequence[tuple[int, int, int]] = (
    (512, 128, 64),
    (1024, 256, 80),
    (2048, 512, 128),
  ),
  eps: float = 1e-5,
) -> jnp.ndarray:
  """Computes per-clip multi-resolution log-mel L1 distance ``(batch,)``."""
  if y.shape != x.shape:
    raise ValueError(f"y {y.shape} and x {x.shape} differ in shape.")
  valid = [s for s in specs if x.shape[1] > s[0] // 2]
  if not valid:
    valid = [(256, 64, 40)]
  terms: list[jnp.ndarray] = []
  for n_fft, hop, n_mels in valid:
    mag_y = stft_magnitude(y, n_fft, hop, n_fft, eps=eps * eps)
    mag_x = stft_magnitude(x, n_fft, hop, n_fft, eps=eps * eps)
    fb = jnp.asarray(_mel_filterbank_np(float(sample_rate_hz), n_fft, n_mels))
    mel_y = jnp.matmul(mag_y, fb)
    mel_x = jnp.matmul(mag_x, fb)
    terms.append(
      jnp.mean(
        jnp.abs(jnp.log(mel_y + eps) - jnp.log(mel_x + eps)), axis=(1, 2)
      )
    )
  return jnp.mean(jnp.stack(terms, axis=0), axis=0)


def low_freq_complex_stft_loss(
  y: jnp.ndarray,
  x: jnp.ndarray,
  *,
  sample_rate_hz: float = 32000.0,
  cutoff_hz: float = 900.0,
  specs: Sequence[tuple[int, int]] = ((1024, 256), (2048, 512)),
  eps: float = 1e-5,
) -> jnp.ndarray:
  """Computes per-clip relative complex STFT L2 error below ``cutoff_hz``."""
  if y.shape != x.shape:
    raise ValueError(f"y {y.shape} and x {x.shape} differ in shape.")
  valid = [s for s in specs if x.shape[1] > s[0] // 2]
  if not valid:
    valid = [(256, 64)]
  terms: list[jnp.ndarray] = []
  for n_fft, hop in valid:
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate_hz))
    k_max = max(2, int(np.searchsorted(freqs, cutoff_hz, side="right")))
    zy = _framed_rfft(y, n_fft, hop, n_fft)[:, :, :k_max]
    zx = _framed_rfft(x, n_fft, hop, n_fft)[:, :, :k_max]
    diff_sq = jnp.square(zy.real - zx.real) + jnp.square(zy.imag - zx.imag)
    ref_sq = jnp.square(zx.real) + jnp.square(zx.imag)
    err = jnp.sqrt(jnp.mean(diff_sq, axis=(1, 2)) + 1e-12)
    ref = jnp.maximum(jnp.sqrt(jnp.mean(ref_sq, axis=(1, 2)) + 1e-12), eps)
    terms.append(err / ref)
  return jnp.mean(jnp.stack(terms, axis=0), axis=0)


def multiscale_log_rms_loss(
  y: jnp.ndarray,
  x: jnp.ndarray,
  *,
  sample_rate_hz: float = 32000.0,
  windows_ms: Sequence[float] = (12.5, 50.0),
  eps: float = 1e-4,
) -> jnp.ndarray:
  """Computes per-clip global + local frame log-RMS envelope L1 distance."""
  if y.shape != x.shape:
    raise ValueError(f"y {y.shape} and x {x.shape} differ in shape.")
  ry = jnp.sqrt(jnp.mean(jnp.square(y), axis=1) + 1e-12)
  rx = jnp.sqrt(jnp.mean(jnp.square(x), axis=1) + 1e-12)
  terms: list[jnp.ndarray] = [jnp.abs(jnp.log(ry + eps) - jnp.log(rx + eps))]
  for w_ms in windows_ms:
    win = max(1, int(round(float(w_ms) * 1e-3 * float(sample_rate_hz))))
    n_f = x.shape[1] // win
    if n_f > 0:
      yf = y[:, : n_f * win].reshape(y.shape[0], n_f, win)
      xf = x[:, : n_f * win].reshape(x.shape[0], n_f, win)
      r_yf = jnp.sqrt(jnp.mean(jnp.square(yf), axis=2) + 1e-12)
      r_xf = jnp.sqrt(jnp.mean(jnp.square(xf), axis=2) + 1e-12)
      terms.append(
        jnp.mean(jnp.abs(jnp.log(r_yf + eps) - jnp.log(r_xf + eps)), axis=1)
      )
  return jnp.mean(jnp.stack(terms, axis=0), axis=0)


def stft_loss(
  y: jnp.ndarray,
  x: jnp.ndarray,
  n_fft: int,
  hop: int,
  win_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes spectral convergence and log-magnitude L1 at one resolution."""
  mag_y = stft_magnitude(y, n_fft, hop, win_length)
  mag_x = stft_magnitude(x, n_fft, hop, win_length)
  num = jnp.sqrt(jnp.sum((mag_y - mag_x) ** 2, axis=(1, 2)) + 1e-12)
  den = jnp.maximum(jnp.sqrt(jnp.sum(mag_x**2, axis=(1, 2))), 1e-6)
  sc = jnp.mean(num / den)
  log_mag = jnp.mean(jnp.abs(jnp.log(mag_y) - jnp.log(mag_x)))
  return sc, log_mag


def multi_resolution_stft_loss(
  y: jnp.ndarray,
  x: jnp.ndarray,
  resolutions: Sequence[tuple[int, int, int]] = STFT_RESOLUTIONS_32K,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  """Computes multi-resolution spectral convergence + log-magnitude L1."""
  if y.shape != x.shape:
    raise ValueError(f"y {y.shape} and x {x.shape} differ in shape.")
  valid = [r for r in resolutions if x.shape[1] > r[0] // 2]
  if not valid:
    raise ValueError(
      f"Sequence length {x.shape[1]} too short for resolutions {resolutions}."
    )
  parts = [stft_loss(y, x, *spec) for spec in valid]
  sc = jnp.mean(jnp.stack([p[0] for p in parts]))
  log_mag = jnp.mean(jnp.stack([p[1] for p in parts]))
  return sc + log_mag, {
    "spectral_convergence": sc,
    "log_magnitude": log_mag,
  }


def spectral_excess_loss(
  y: jnp.ndarray,
  x: jnp.ndarray,
  resolutions: Sequence[tuple[int, int, int]] = STFT_RESOLUTIONS_32K,
  eps: float = 1e-5,
) -> jnp.ndarray:
  """Asymmetric squared penalty on positive log-STFT magnitude excess."""
  if y.shape != x.shape:
    raise ValueError(f"y {y.shape} and x {x.shape} differ in shape.")
  valid = [r for r in resolutions if x.shape[1] > r[0] // 2]
  if not valid:
    raise ValueError(
      f"Sequence length {x.shape[1]} too short for resolutions {resolutions}."
    )
  terms: list[jnp.ndarray] = []
  for n_fft, hop, win_len in valid:
    mag_y = stft_magnitude(y, n_fft, hop, win_len, eps=eps**2)
    mag_x = stft_magnitude(x, n_fft, hop, win_len, eps=eps**2)
    excess = jax.nn.relu(jnp.log(mag_y + eps) - jnp.log(mag_x + eps))
    terms.append(jnp.mean(excess * excess))
  return jnp.mean(jnp.stack(terms))


class PeriodDiscriminator(nn.Module):
  """HiFi-GAN period sub-discriminator lowered to 1D NWC convolutions."""

  period: int
  channels: Sequence[int] = (32, 64, 128, 256)
  kernel: int = 5
  stride: int = 3

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """Returns flattened logits and intermediate feature maps for ``x``."""
    batch, samples = x.shape
    extra = (-samples) % self.period
    if extra:
      x = jnp.pad(x, ((0, 0), (0, extra)), mode="reflect")
    t_folded = x.shape[1] // self.period
    # Keep batch outermost so folded periods can run via TPU-friendly 1D convs.
    h = jnp.transpose(x.reshape(batch, t_folded, self.period), (0, 2, 1))
    h = h.reshape(batch * self.period, t_folded, 1)

    def _to_period_map(act: jnp.ndarray, ch: int) -> jnp.ndarray:
      return jnp.transpose(
        act.reshape(batch, self.period, -1, ch), (0, 2, 1, 3)
      )

    features: list[jnp.ndarray] = []
    for ch in self.channels:
      h = nn.Conv(ch, (self.kernel,), strides=(self.stride,))(h)
      h = nn.leaky_relu(h, _DISCRIMINATOR_LEAK)
      features.append(_to_period_map(h, ch))
    last_ch = self.channels[-1]
    h = nn.Conv(last_ch, (self.kernel,))(h)
    h = nn.leaky_relu(h, _DISCRIMINATOR_LEAK)
    features.append(_to_period_map(h, last_ch))
    logits = _to_period_map(nn.Conv(1, (3,))(h), 1)
    return logits.reshape(batch, -1), features


class MultiPeriodDiscriminator(nn.Module):
  """HiFi-GAN Multi-Period Discriminator across prime folding periods."""

  periods: Sequence[int] = DISCRIMINATOR_PERIODS
  channels: Sequence[int] = (32, 64, 128, 256)

  @nn.compact
  def __call__(
    self, x: jnp.ndarray
  ) -> list[tuple[jnp.ndarray, list[jnp.ndarray]]]:
    """Evaluates all period sub-discriminators on ``(batch, samples)``."""
    return [
      PeriodDiscriminator(p, self.channels, name=f"period_{p}")(x)
      for p in self.periods
    ]


class ResolutionDiscriminator(nn.Module):
  """Linear-STFT magnitude sub-discriminator using 1D temporal NWC convs."""

  n_fft: int
  hop: int
  win_length: int
  channels: int = 32
  num_strided: int = 3

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """Returns flattened logits and feature maps over STFT spectral frames."""
    batch = x.shape[0]
    h = stft_magnitude(x, self.n_fft, self.hop, self.win_length)
    hidden = max(1, self.channels * 4)
    mid = max(1, self.channels * 2)
    features: list[jnp.ndarray] = []
    h = nn.leaky_relu(nn.Conv(hidden, (3,))(h), _DISCRIMINATOR_LEAK)
    features.append(h)
    for _ in range(self.num_strided):
      h = nn.Conv(hidden, (5,), strides=(2,))(h)
      h = nn.leaky_relu(h, _DISCRIMINATOR_LEAK)
      features.append(h)
    h = nn.leaky_relu(nn.Conv(mid, (3,))(h), _DISCRIMINATOR_LEAK)
    features.append(h)
    logits = nn.Conv(1, (3,))(h)
    return logits.reshape(batch, -1), features


class MultiResolutionDiscriminator(nn.Module):
  """UnivNet/Vocos Multi-Resolution Discriminator over STFT magnitudes."""

  resolutions: Sequence[tuple[int, int, int]] = STFT_RESOLUTIONS_32K
  channels: int = 32

  @nn.compact
  def __call__(
    self, x: jnp.ndarray
  ) -> list[tuple[jnp.ndarray, list[jnp.ndarray]]]:
    """Evaluates all resolution sub-discriminators on ``(batch, samples)``."""
    return [
      ResolutionDiscriminator(
        *spec, channels=self.channels, name=f"res_{spec[0]}"
      )(x)
      for spec in self.resolutions
    ]


def _masked_batch_mean(
  per_example: jnp.ndarray, mask: jnp.ndarray | None
) -> jnp.ndarray:
  if mask is None:
    return jnp.mean(per_example)
  w = jnp.asarray(mask, dtype=jnp.float32)
  return jnp.sum(per_example * w) / jnp.maximum(jnp.sum(w), 1.0)


def discriminator_loss(
  real_outputs: Sequence[tuple[jnp.ndarray, Sequence[jnp.ndarray]]],
  fake_outputs: Sequence[tuple[jnp.ndarray, Sequence[jnp.ndarray]]],
  *,
  mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Computes least-squares GAN discriminator loss summed across heads."""
  total = jnp.float32(0.0)
  for (r_log, _), (f_log, _) in zip(real_outputs, fake_outputs, strict=True):
    b = r_log.shape[0]
    r_per = jnp.mean(((r_log - 1.0) ** 2).reshape(b, -1), axis=1)
    f_per = jnp.mean((f_log**2).reshape(b, -1), axis=1)
    total = total + _masked_batch_mean(r_per + f_per, mask)
  return total


def generator_loss(
  fake_outputs: Sequence[tuple[jnp.ndarray, Sequence[jnp.ndarray]]],
  *,
  mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Computes least-squares GAN generator adversarial loss."""
  total = jnp.float32(0.0)
  for f_log, _ in fake_outputs:
    b = f_log.shape[0]
    f_per = jnp.mean(((f_log - 1.0) ** 2).reshape(b, -1), axis=1)
    total = total + _masked_batch_mean(f_per, mask)
  return total


def feature_matching_loss(
  real_outputs: Sequence[tuple[jnp.ndarray, Sequence[jnp.ndarray]]],
  fake_outputs: Sequence[tuple[jnp.ndarray, Sequence[jnp.ndarray]]],
  *,
  mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
  """Computes HiFi-GAN L1 hidden-feature matching loss with fixed targets."""
  total = jnp.float32(0.0)
  for (_, r_feats), (_, f_feats) in zip(
    real_outputs, fake_outputs, strict=True
  ):
    for real_f, fake_f in zip(r_feats, f_feats, strict=True):
      b = real_f.shape[0]
      fm_per = jnp.mean(
        jnp.abs(jax.lax.stop_gradient(real_f) - fake_f).reshape(b, -1), axis=1
      )
      total = total + _masked_batch_mean(fm_per, mask)
  return total


def _masked_mean_value(
  per_clip: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
  active_mask = weights > 0.0
  return jnp.sum(jnp.where(active_mask, per_clip, 0.0)) / jnp.maximum(
    jnp.sum(active_mask), 1
  )


def compute_arcfac_inverter_loss(
  y: jnp.ndarray,
  x: jnp.ndarray,
  nap_target: jnp.ndarray,
  healthy_ear_fn: Callable[[jnp.ndarray], jnp.ndarray],
  *,
  lookahead: int = 256,
  mask_samples: int = 8000,
  output_scale: float = 1.0,
  nap_weight: float = 1.0,
  relative_nap: bool = False,
  lam: jnp.ndarray | float = 0.01,
  warmup_weight: float = 0.01,
  stft_weight: jnp.ndarray | float = 0.0,
  apply_stft: bool = False,
  excess_stft_weight: jnp.ndarray | float = 0.0,
  apply_excess_stft: bool = False,
  mel_weight: jnp.ndarray | float = 0.0,
  apply_mel: bool = False,
  lf_complex_weight: jnp.ndarray | float = 0.0,
  apply_lf_complex: bool = False,
  rms_env_weight: jnp.ndarray | float = 0.0,
  apply_rms_env: bool = False,
  sample_rate_hz: float = 32000.0,
  sai_weight: float = 0.0,
  stft_resolutions: Sequence[tuple[int, int, int]] = STFT_RESOLUTIONS_32K,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  """Computes the masked NAP/SAI + perceptual/phase-free vocoder losses.

  Args:
    y: Predicted waveforms of shape ``(batch, samples)``.
    x: Source clean waveforms (before impairment) of shape ``(batch, samples)``.
    nap_target: Target cochlear NAPs of shape ``(batch, samples, channels)``.
    healthy_ear_fn: Differentiable healthy cochlea callable mapping
      ``(batch, samples) -> (batch, samples, channels)``.
    lookahead: Causal lookahead delay ``L`` applied to ``nap_target`` and ``x``.
    mask_samples: Prefix samples ignored by CARFAC loss (start-from-rest mask).
    output_scale: Corpus RMS floor preventing quiet-clip gradient spikes.
    nap_weight: Scalar multiplier on cochlear NAP L1 consistency (setting to
      ``0.0`` alongside ``sai_weight=0.0`` skips running the CARFAC forward
      and backward pass completely).
    relative_nap: When ``True``, normalizes each stream's NAP L1 loss by its
      target steady-state NAP magnitude so quiet impaired ears keep equal weight.
    lam: Waveform L1 tie-breaker weight (scalar or ``(batch,)`` with ``0`` for
      impaired targets so outputs do not regress toward un-impaired ``x``).
    warmup_weight: Weight on normalized prefix energy before ``start``.
    stft_weight: Multi-resolution STFT weight (scalar or ``(batch,)`` with
      ``0`` on impaired targets).
    apply_stft: Static switch enabling MR-STFT computation when ``stft_weight > 0``.
    excess_stft_weight: Weight on ``spectral_excess_loss`` penalizing predicted
      spectral peaks above target (can target all ears safely).
    apply_excess_stft: Static switch enabling ``spectral_excess_loss``.
    mel_weight: Multi-scale log-mel L1 loss weight (phase-invariant).
    apply_mel: Static switch enabling multi-scale log-mel loss.
    lf_complex_weight: Low-frequency complex STFT L2 loss weight.
    apply_lf_complex: Static switch enabling low-frequency complex STFT loss.
    rms_env_weight: Multi-scale log-RMS envelope L1 loss weight.
    apply_rms_env: Static switch enabling multi-scale log-RMS envelope loss.
    sample_rate_hz: Audio sample rate in Hz for mel/LF/RMS frequency grids.
    sai_weight: Optional weight on unmasked healthy CARFAC SAI L1 consistency.
    stft_resolutions: STFT window specs for MR-STFT loss.

  Returns:
    ``(total_loss, scalar_metrics)``.
  """
  start = max(int(mask_samples), int(lookahead))
  if y.shape[1] <= start:
    raise ValueError(
      f"Sequence length {y.shape[1]} leaves no samples past start={start}."
    )

  use_cochlea = nap_weight > 0.0 or sai_weight > 0.0
  tgt_delayed = delay_samples(nap_target, lookahead, axis=1)
  if use_cochlea:
    nap_y = healthy_ear_fn(y)
  else:
    nap_y = jnp.zeros_like(tgt_delayed)

  if nap_weight > 0.0:
    nap_steady = jnp.mean(
      jnp.abs(nap_y[:, start:] - tgt_delayed[:, start:]), axis=(1, 2)
    )
    if start > lookahead:
      nap_onset = jnp.mean(
        jnp.abs(nap_y[:, lookahead:start] - tgt_delayed[:, lookahead:start]),
        axis=(1, 2),
      )
      nap_total = nap_steady + 0.25 * nap_onset
    else:
      nap_total = nap_steady
    if relative_nap:
      nap_denom = jnp.maximum(
        jnp.mean(jnp.abs(tgt_delayed[:, start:]), axis=(1, 2)), 1e-4
      )
      nap_total = nap_total / nap_denom
    nap_term = jnp.float32(nap_weight) * nap_total
  else:
    nap_steady = jnp.zeros((y.shape[0],), dtype=jnp.float32)
    nap_term = nap_steady

  x_delayed = delay_samples(x, lookahead, axis=1)
  abs_floor = max(0.1 * float(output_scale), 1e-7)
  pow_floor = abs_floor**2
  wave_per_clip = jnp.mean(
    jnp.abs(y[:, lookahead:] - x_delayed[:, lookahead:]), axis=1
  ) / jnp.maximum(
    jnp.mean(jnp.abs(x_delayed[:, lookahead:]), axis=1), abs_floor
  )
  lookahead_energy = (
    jnp.mean(y[:, :lookahead] ** 2, axis=1)
    if lookahead > 0
    else jnp.zeros((y.shape[0],), dtype=jnp.float32)
  )
  excess_warm = jnp.mean(
    jnp.maximum(y[:, :start] ** 2 - x_delayed[:, :start] ** 2, 0.0), axis=1
  )
  warm_per_clip = (lookahead_energy + excess_warm) / jnp.maximum(
    jnp.mean(x_delayed[:, start:] ** 2, axis=1), pow_floor
  )

  lam_arr = jnp.asarray(lam, dtype=jnp.float32)
  total_per_clip = (
    nap_term + lam_arr * wave_per_clip + warmup_weight * warm_per_clip
  )

  if sai_weight > 0.0:
    hop = max(1, _SAI_HYPERS.win_len // 2)
    first_frame = -(-start // hop)
    sai_y = jnp.moveaxis(sai_lib.from_naps(nap_y, _SAI_HYPERS), 0, 1)
    sai_tgt = jnp.moveaxis(
      sai_lib.from_naps(jax.lax.stop_gradient(tgt_delayed), _SAI_HYPERS),
      0,
      1,
    )
    if sai_y.shape[1] > first_frame:
      sai_per_clip = jnp.mean(
        jnp.abs(sai_y[:, first_frame:] - sai_tgt[:, first_frame:]),
        axis=(1, 2, 3),
      )
    else:
      sai_per_clip = jnp.zeros((y.shape[0],), dtype=jnp.float32)
    total_per_clip = total_per_clip + float(sai_weight) * sai_per_clip
    sai_mean = jnp.mean(sai_per_clip)
  else:
    sai_mean = jnp.float32(0.0)

  y_seg = y[:, lookahead:]
  x_seg = x_delayed[:, lookahead:]

  if apply_stft:
    stft_w = jnp.asarray(stft_weight, dtype=jnp.float32)
    stft_per_clip = jax.vmap(
      lambda a, b: multi_resolution_stft_loss(
        a[None, :], b[None, :], resolutions=stft_resolutions
      )[0]
    )(y_seg, x_seg)
    total_per_clip = total_per_clip + stft_w * stft_per_clip
    stft_mean = _masked_mean_value(stft_per_clip, stft_w)
  else:
    stft_mean = jnp.float32(0.0)

  if apply_excess_stft:
    ex_w = jnp.asarray(excess_stft_weight, dtype=jnp.float32)
    ex_per_clip = jax.vmap(
      lambda a, b: spectral_excess_loss(
        a[None, :], b[None, :], resolutions=stft_resolutions
      )
    )(y_seg, x_seg)
    total_per_clip = total_per_clip + ex_w * ex_per_clip
    excess_mean = _masked_mean_value(ex_per_clip, ex_w)
  else:
    excess_mean = jnp.float32(0.0)

  if apply_mel:
    mel_w = jnp.asarray(mel_weight, dtype=jnp.float32)
    mel_per_clip = multi_scale_mel_loss(
      y_seg, x_seg, sample_rate_hz=sample_rate_hz
    )
    total_per_clip = total_per_clip + mel_w * mel_per_clip
    mel_mean = _masked_mean_value(mel_per_clip, mel_w)
  else:
    mel_mean = jnp.float32(0.0)

  if apply_lf_complex:
    lf_w = jnp.asarray(lf_complex_weight, dtype=jnp.float32)
    lf_per_clip = low_freq_complex_stft_loss(
      y_seg, x_seg, sample_rate_hz=sample_rate_hz
    )
    total_per_clip = total_per_clip + lf_w * lf_per_clip
    lf_mean = _masked_mean_value(lf_per_clip, lf_w)
  else:
    lf_mean = jnp.float32(0.0)

  if apply_rms_env:
    rms_w = jnp.asarray(rms_env_weight, dtype=jnp.float32)
    rms_per_clip = multiscale_log_rms_loss(
      y_seg, x_seg, sample_rate_hz=sample_rate_hz
    )
    total_per_clip = total_per_clip + rms_w * rms_per_clip
    rms_mean = _masked_mean_value(rms_per_clip, rms_w)
  else:
    rms_mean = jnp.float32(0.0)

  loss = jnp.mean(total_per_clip)
  return loss, {
    "loss": cast(jnp.ndarray, loss),
    "nap_l1": cast(jnp.ndarray, jnp.mean(nap_steady)),
    "sai_l1": cast(jnp.ndarray, sai_mean),
    "l1_waveform": cast(jnp.ndarray, jnp.mean(wave_per_clip)),
    "warmup_energy": cast(jnp.ndarray, jnp.mean(warm_per_clip)),
    "stft": cast(jnp.ndarray, stft_mean),
    "excess_stft": cast(jnp.ndarray, excess_mean),
    "mel": cast(jnp.ndarray, mel_mean),
    "lf_complex": cast(jnp.ndarray, lf_mean),
    "rms_env": cast(jnp.ndarray, rms_mean),
  }
