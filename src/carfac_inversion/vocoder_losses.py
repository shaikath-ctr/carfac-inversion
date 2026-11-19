"""Losses, causal Butterworth filter, and GAN discriminators for Phase 4.

Combines AFHI ``arcfac`` stabilization primitives (causal 4th-order 20 Hz output
high-pass filtering, masked warm-up energy penalties, and healthy-only
waveform/STFT regularization masking) with multi-resolution STFT and HiFi-GAN /
UnivNet discriminators for direct NAP-to-audio and perceptual auralization
training.
"""

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


def stft_magnitude(
  x: jnp.ndarray,
  n_fft: int,
  hop: int,
  win_length: int,
  eps: float = 1e-7,
) -> jnp.ndarray:
  """Computes STFT magnitude ``(batch, frames, n_fft // 2 + 1)``."""
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
  spec = jnp.fft.rfft(x_pad[:, idx] * window[None, None, :], axis=-1)
  return jnp.sqrt(jnp.real(spec) ** 2 + jnp.imag(spec) ** 2 + eps)


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


class PeriodDiscriminator(nn.Module):
  """HiFi-GAN period sub-discriminator operating on folded waveform slices."""

  period: int
  channels: Sequence[int] = (32, 128, 512, 1024)
  kernel: int = 5
  stride: int = 3

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """Returns flattened logits and intermediate feature maps for ``x``."""
    batch, samples = x.shape
    extra = (-samples) % self.period
    if extra:
      x = jnp.pad(x, ((0, 0), (0, extra)), mode="reflect")
    h = x.reshape(batch, -1, self.period, 1)
    features: list[jnp.ndarray] = []
    for ch in self.channels:
      h = nn.Conv(ch, (self.kernel, 1), strides=(self.stride, 1))(h)
      h = nn.leaky_relu(h, _DISCRIMINATOR_LEAK)
      features.append(h)
    h = nn.Conv(self.channels[-1], (self.kernel, 1))(h)
    h = nn.leaky_relu(h, _DISCRIMINATOR_LEAK)
    features.append(h)
    logits = nn.Conv(1, (3, 1))(h)
    return logits.reshape(batch, -1), features


class MultiPeriodDiscriminator(nn.Module):
  """HiFi-GAN Multi-Period Discriminator across prime folding periods."""

  periods: Sequence[int] = DISCRIMINATOR_PERIODS
  channels: Sequence[int] = (32, 128, 512, 1024)

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
  """UnivNet/Vocos 2D linear-STFT magnitude sub-discriminator."""

  n_fft: int
  hop: int
  win_length: int
  channels: int = 32
  num_strided: int = 3

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """Returns flattened logits and feature maps over linear STFT bins."""
    batch = x.shape[0]
    h = stft_magnitude(x, self.n_fft, self.hop, self.win_length)[..., None]
    features: list[jnp.ndarray] = []
    h = nn.leaky_relu(nn.Conv(self.channels, (3, 9))(h), _DISCRIMINATOR_LEAK)
    features.append(h)
    for _ in range(self.num_strided):
      h = nn.Conv(self.channels, (3, 9), strides=(1, 2))(h)
      h = nn.leaky_relu(h, _DISCRIMINATOR_LEAK)
      features.append(h)
    h = nn.leaky_relu(nn.Conv(self.channels, (3, 3))(h), _DISCRIMINATOR_LEAK)
    features.append(h)
    logits = nn.Conv(1, (3, 3))(h)
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


def discriminator_loss(
  real_outputs: Sequence[tuple[jnp.ndarray, Sequence[jnp.ndarray]]],
  fake_outputs: Sequence[tuple[jnp.ndarray, Sequence[jnp.ndarray]]],
) -> jnp.ndarray:
  """Computes least-squares GAN discriminator loss summed across heads."""
  total = jnp.float32(0.0)
  for (r_log, _), (f_log, _) in zip(real_outputs, fake_outputs, strict=True):
    total = total + jnp.mean((r_log - 1.0) ** 2) + jnp.mean(f_log**2)
  return total


def generator_loss(
  fake_outputs: Sequence[tuple[jnp.ndarray, Sequence[jnp.ndarray]]],
) -> jnp.ndarray:
  """Computes least-squares GAN generator adversarial loss."""
  total = jnp.float32(0.0)
  for f_log, _ in fake_outputs:
    total = total + jnp.mean((f_log - 1.0) ** 2)
  return total


def feature_matching_loss(
  real_outputs: Sequence[tuple[jnp.ndarray, Sequence[jnp.ndarray]]],
  fake_outputs: Sequence[tuple[jnp.ndarray, Sequence[jnp.ndarray]]],
) -> jnp.ndarray:
  """Computes HiFi-GAN L1 hidden-feature matching loss with fixed targets."""
  total = jnp.float32(0.0)
  for (_, r_feats), (_, f_feats) in zip(
    real_outputs, fake_outputs, strict=True
  ):
    for real_f, fake_f in zip(r_feats, f_feats, strict=True):
      total = total + jnp.mean(jnp.abs(jax.lax.stop_gradient(real_f) - fake_f))
  return total


def compute_arcfac_inverter_loss(
  y: jnp.ndarray,
  x: jnp.ndarray,
  nap_target: jnp.ndarray,
  healthy_ear_fn: Callable[[jnp.ndarray], jnp.ndarray],
  *,
  lookahead: int = 256,
  mask_samples: int = 8000,
  output_scale: float = 1.0,
  lam: jnp.ndarray | float = 0.01,
  warmup_weight: float = 0.01,
  stft_weight: jnp.ndarray | float = 0.0,
  apply_stft: bool = False,
  sai_weight: float = 0.0,
  stft_resolutions: Sequence[tuple[int, int, int]] = STFT_RESOLUTIONS_32K,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  """Computes the masked NAP/SAI + healthy-only tie-breaker loss from ``arcfac``.

  Args:
    y: Predicted waveforms of shape ``(batch, samples)``.
    x: Source clean waveforms (before impairment) of shape ``(batch, samples)``.
    nap_target: Target cochlear NAPs of shape ``(batch, samples, channels)``.
    healthy_ear_fn: Differentiable healthy cochlea callable mapping
      ``(batch, samples) -> (batch, samples, channels)``.
    lookahead: Causal lookahead delay ``L`` applied to ``nap_target`` and ``x``.
    mask_samples: Prefix samples ignored by CARFAC loss (start-from-rest mask).
    output_scale: Corpus RMS floor preventing quiet-clip gradient spikes.
    lam: Waveform L1 tie-breaker weight (scalar or ``(batch,)`` with ``0`` for
      impaired targets so outputs do not regress toward un-impaired ``x``).
    warmup_weight: Weight on normalized prefix energy before ``start``.
    stft_weight: Multi-resolution STFT weight (scalar or ``(batch,)`` with
      ``0`` on impaired targets).
    apply_stft: Static switch enabling MR-STFT computation when ``stft_weight > 0``.
    sai_weight: Optional weight on unmasked healthy CARFAC SAI L1 consistency
      (applies cleanly to both healthy and impaired ears).
    stft_resolutions: STFT window specs for MR-STFT loss.

  Returns:
    ``(total_loss, scalar_metrics)``.
  """
  start = max(int(mask_samples), int(lookahead))
  if y.shape[1] <= start:
    raise ValueError(
      f"Sequence length {y.shape[1]} leaves no samples past start={start}."
    )

  nap_y = healthy_ear_fn(y)
  tgt_delayed = delay_samples(nap_target, lookahead, axis=1)
  nap_per_clip = jnp.mean(
    jnp.abs(nap_y[:, start:] - tgt_delayed[:, start:]), axis=(1, 2)
  )

  x_delayed = delay_samples(x, lookahead, axis=1)
  abs_floor = max(0.1 * float(output_scale), 1e-7)
  pow_floor = abs_floor**2
  wave_per_clip = jnp.mean(
    jnp.abs(y[:, start:] - x_delayed[:, start:]), axis=1
  ) / jnp.maximum(jnp.mean(jnp.abs(x_delayed[:, start:]), axis=1), abs_floor)
  warm_per_clip = jnp.mean(y[:, :start] ** 2, axis=1) / jnp.maximum(
    jnp.mean(x_delayed[:, start:] ** 2, axis=1), pow_floor
  )

  lam_arr = jnp.asarray(lam, dtype=jnp.float32)
  total_per_clip = (
    nap_per_clip + lam_arr * wave_per_clip + warmup_weight * warm_per_clip
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

  if apply_stft:
    stft_w = jnp.asarray(stft_weight, dtype=jnp.float32)
    stft_per_clip = jax.vmap(
      lambda a, b: multi_resolution_stft_loss(
        a[None, :], b[None, :], resolutions=stft_resolutions
      )[0]
    )(y[:, start:], x_delayed[:, start:])
    total_per_clip = total_per_clip + stft_w * stft_per_clip
    active_mask = stft_w > 0.0
    stft_mean = jnp.sum(
      jnp.where(active_mask, stft_per_clip, 0.0)
    ) / jnp.maximum(jnp.sum(active_mask), 1)
  else:
    stft_mean = jnp.float32(0.0)

  loss = jnp.mean(total_per_clip)
  return loss, {
    "loss": cast(jnp.ndarray, loss),
    "nap_l1": cast(jnp.ndarray, jnp.mean(nap_per_clip)),
    "sai_l1": cast(jnp.ndarray, sai_mean),
    "l1_waveform": cast(jnp.ndarray, jnp.mean(wave_per_clip)),
    "warmup_energy": cast(jnp.ndarray, jnp.mean(warm_per_clip)),
    "stft": cast(jnp.ndarray, stft_mean),
  }
