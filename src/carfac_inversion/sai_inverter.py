"""Phase 5b generative direct SAI-to-audio neural inverter in JAX.

Unlike NAPs, CARFAC Stabilized Auditory Images (SAIs) compute windowed
autocorrelations per ~21 ms hop (`hop = 682` samples at 32 kHz), discarding both
within-frame carrier phase and cross-channel timing. :class:`SaiDirectInverter`
extracts per-channel zero-lag energy, multi-scale lag profiles, and high-CF
unresolved harmonic-summed summary autocorrelation pitch/voicing (`SACF`), and
synthesizes audio via a DDSP sample-rate harmonic additive bank plus a COLA
STFT-filtered white-noise branch.
"""

import dataclasses
import functools
import importlib
import json
import math
import pathlib
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any
from typing import cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from hp_acoustic.models import sai as sai_lib

from carfac_inversion import nap_inverter
from carfac_inversion import tpu
from carfac_inversion import vocoder_losses

DEFAULT_SAI_HYPERS = sai_lib.SaiHypers()


def _sai_sine_window_f32(win_len: int, fft_len: int) -> jnp.ndarray:
  """Builds an RMS-normalized sine window strictly in ``float32``."""
  idx = jnp.arange(win_len, dtype=jnp.float32)
  window = jnp.sin(jnp.float32(math.pi) * (idx + 1.0) / float(win_len + 1))
  fft_win = jnp.pad(window, (0, fft_len - win_len))
  return fft_win / jnp.sqrt(jnp.mean(fft_win**2))


def sai_from_naps_f32(
  naps: jnp.ndarray,
  hypers: sai_lib.SaiHypers = DEFAULT_SAI_HYPERS,
) -> jnp.ndarray:
  """Computes differentiable ``float32`` SAIs ``(B, F, lags, C)`` from NAPs.

  Matches :func:`hp_acoustic.models.sai.from_naps` with ``(batch, frames, lags,
  channels)`` ordering while staying strictly ``float32``/``complex64`` with
  static lag slicing for Pathways TPU execution.
  """
  naps_f32 = jnp.asarray(naps, dtype=jnp.float32)
  if naps_f32.ndim != 3:
    raise ValueError(
      f"Expected naps of shape (batch, samples, channels), got {naps_f32.shape}."
    )
  batch, signal_len, channels = naps_f32.shape
  fft_len = int(hypers.fft_len)
  win_len = int(hypers.win_len)
  hop = win_len // 2
  lag_limit = fft_len // int(hypers.laglim)
  num_lags = 1 + 2 * lag_limit

  num_frames = max(0, math.ceil((signal_len - win_len) / hop) + 1)
  if num_frames == 0:
    return jnp.zeros((batch, 0, num_lags, channels), dtype=jnp.float32)

  whole, rest = divmod(fft_len, hop)
  num_blocks = num_frames + whole
  padded_naps = jnp.pad(
    naps_f32, ((0, 0), (0, num_blocks * hop - signal_len), (0, 0))
  )
  blocks = padded_naps.reshape((batch, num_blocks, hop, channels))
  parts = [blocks[:, k : k + num_frames] for k in range(whole)]
  if rest > 0:
    parts.append(blocks[:, whole : whole + num_frames, :rest])
  stacked = jnp.concatenate(parts, axis=2)  # (batch, num_frames, fft_len, C)

  fft_win = _sai_sine_window_f32(win_len, fft_len)[None, None, :, None]
  spec = jnp.fft.rfft(stacked * fft_win, axis=2)
  power = jnp.real(spec) ** 2 + jnp.imag(spec) ** 2
  autocorr = jnp.real(jnp.fft.irfft(power, n=fft_len, axis=2)) / float(fft_len)
  autocorr = jnp.fft.fftshift(autocorr, axes=2)
  center = fft_len // 2
  sliced = autocorr[:, :, center - lag_limit : center + lag_limit + 1, :]
  return jnp.maximum(sliced, 0.0)


def _scale_grad(x: jnp.ndarray, scale: float) -> jnp.ndarray:
  """Passes ``x`` forward unchanged while scaling backward gradients."""
  sg = jax.lax.stop_gradient(x)
  return sg + jnp.float32(scale) * (x - sg)


def _phase_from_freq_f32(
  f0_samp: jnp.ndarray, fs: float, block_size: int = 64
) -> jnp.ndarray:
  """Integrates ``f0_samp / fs`` into wrapped phase ``[0, 1)`` without drift."""
  b, t_len = f0_samp.shape
  dphi = f0_samp / jnp.float32(fs)
  pad_t = (-t_len) % block_size
  dphi_pad = jnp.pad(dphi, ((0, 0), (0, pad_t))) if pad_t > 0 else dphi
  n_blocks = dphi_pad.shape[1] // block_size
  blocks = jnp.swapaxes(dphi_pad.reshape((b, n_blocks, block_size)), 0, 1)

  def _step(
    carry: jnp.ndarray, blk: jnp.ndarray
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    raw = carry[:, None] + jnp.cumsum(blk, axis=-1)
    phi_blk = raw - jax.lax.stop_gradient(jnp.floor(raw))
    next_carry = raw[:, -1] - jax.lax.stop_gradient(jnp.floor(raw[:, -1]))
    return next_carry, phi_blk

  _, phi_blocks = jax.lax.scan(
    _step, jnp.zeros((b,), dtype=jnp.float32), blocks
  )
  return jnp.swapaxes(phi_blocks, 0, 1).reshape((b, n_blocks * block_size))[
    :, :t_len
  ]


def _pad_or_slice_lag(x: jnp.ndarray, target_len: int) -> jnp.ndarray:
  """Pads with trailing zeros or slices axis 2 of ``(B, F, L)`` to ``target_len``."""
  cur = x.shape[2]
  if cur < target_len:
    return jnp.pad(x, ((0, 0), (0, 0), (0, target_len - cur)))
  if cur > target_len:
    return x[:, :, :target_len]
  return x


def _hann_f32(n_fft: int) -> jnp.ndarray:
  n = jnp.arange(n_fft, dtype=jnp.float32)
  return 0.5 - 0.5 * jnp.cos(jnp.float32(2.0 * math.pi) * n / float(n_fft))


def _hann_512_f32() -> jnp.ndarray:
  return _hann_f32(512)


def _hann_1024_f32() -> jnp.ndarray:
  return _hann_f32(1024)


def _cola_istft_hop128(
  spec_complex: jnp.ndarray,
  *,
  n_fft: int,
  target_samples: int,
) -> jnp.ndarray:
  """Exact scatter-free COLA ISTFT for ``hop=128`` Hann frames."""
  n_bins = n_fft // 2 + 1
  n_overlaps = n_fft // 128
  half_win = n_fft // 2
  if spec_complex.ndim != 3 or spec_complex.shape[2] != n_bins:
    raise ValueError(
      f"Expected spec_complex of shape (batch, frames, {n_bins}), got {spec_complex.shape}."
    )
  batch, f_sub, _ = spec_complex.shape
  hann = _hann_f32(n_fft)
  frames_win = (
    jnp.fft.irfft(spec_complex, n=n_fft, axis=-1) * hann[None, None, :]
  ).reshape((batch, f_sub, n_overlaps, 128))
  out_blocks = f_sub + n_overlaps - 1
  y_ola = sum(
    (
      jnp.pad(frames_win[:, :, k, :], ((0, 0), (k, n_overlaps - 1 - k), (0, 0)))
      for k in range(n_overlaps)
    ),
    jnp.zeros((batch, out_blocks, 128), dtype=jnp.float32),
  ).reshape((batch, out_blocks * 128))

  win_sq_frames = jnp.broadcast_to(
    (hann**2).reshape((1, 1, n_overlaps, 128)), (1, f_sub, n_overlaps, 128)
  )
  win_sq_env = sum(
    (
      jnp.pad(
        win_sq_frames[:, :, k, :], ((0, 0), (k, n_overlaps - 1 - k), (0, 0))
      )
      for k in range(n_overlaps)
    ),
    jnp.zeros((1, out_blocks, 128), dtype=jnp.float32),
  ).reshape((1, out_blocks * 128))

  y_norm = y_ola / jnp.maximum(win_sq_env, 1e-6)
  return y_norm[:, half_win : half_win + target_samples]


def cola_istft_512_128(
  spec_complex: jnp.ndarray,
  target_samples: int,
) -> jnp.ndarray:
  """Exact scatter-free COLA ISTFT for ``n_fft=512, hop=128`` Hann frames."""
  return _cola_istft_hop128(
    spec_complex, n_fft=512, target_samples=target_samples
  )


def cola_istft_1024_128(
  spec_complex: jnp.ndarray,
  target_samples: int,
) -> jnp.ndarray:
  """Exact scatter-free COLA ISTFT for ``n_fft=1024, hop=128`` Hann frames."""
  return _cola_istft_hop128(
    spec_complex, n_fft=1024, target_samples=target_samples
  )


def _stft_hop128_complex(x_pad: jnp.ndarray, *, n_fft: int) -> jnp.ndarray:
  """Computes center-padded complex STFT ``(B, 1 + T_pad // 128, n_fft // 2 + 1)``."""
  batch, t_pad = x_pad.shape
  if t_pad % 128 != 0:
    raise ValueError(f"Expected t_pad divisible by 128, got {t_pad}.")
  half_win = n_fft // 2
  f_sub = 1 + t_pad // 128
  x_ext = jnp.pad(x_pad, ((0, 0), (half_win, half_win)))
  idx = jnp.arange(f_sub)[:, None] * 128 + jnp.arange(n_fft)[None, :]
  frames = x_ext[:, idx] * _hann_f32(n_fft)[None, None, :]
  return jnp.fft.rfft(frames, n=n_fft, axis=-1).reshape(
    (batch, f_sub, n_fft // 2 + 1)
  )


def _stft_512_128_complex(x_pad: jnp.ndarray) -> jnp.ndarray:
  """Computes center-padded ``n_fft=512, hop=128`` complex STFT."""
  return _stft_hop128_complex(x_pad, n_fft=512)


def _stft_1024_128_complex(x_pad: jnp.ndarray) -> jnp.ndarray:
  """Computes center-padded ``n_fft=1024, hop=128`` complex STFT."""
  return _stft_hop128_complex(x_pad, n_fft=1024)


def _linear_interp_axis1(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
  """Linearly samples ``x`` `(B, F, D)` at continuous positions ``u`` `(M,)`."""
  f_len = x.shape[1]
  u_c = jnp.clip(u, 0.0, float(max(f_len - 1, 0)))
  idx0 = jnp.floor(u_c).astype(jnp.int32)
  idx1 = jnp.minimum(idx0 + 1, max(f_len - 1, 0))
  alpha = (u_c - idx0.astype(jnp.float32))[None, :, None]
  return (1.0 - alpha) * x[:, idx0, :] + alpha * x[:, idx1, :]


class _SymmetricConvNeXt1D(nn.Module):
  """Symmetric SAME-padded 1D depthwise ConvNeXt residual block."""

  dim: int
  mlp_dim: int
  kernel_size: int = 5

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    h = nn.LayerNorm(epsilon=1e-5)(x)
    h = nn.Conv(
      features=self.dim,
      kernel_size=(self.kernel_size,),
      feature_group_count=self.dim,
      padding="SAME",
    )(h)
    h = nn.gelu(nn.Dense(self.mlp_dim)(h))
    h = nn.Dense(self.dim, kernel_init=nn.initializers.zeros_init())(h)
    return x + h


@dataclasses.dataclass(frozen=True)
class SaiInverterConfig:
  """Configuration for :class:`SaiDirectInverter`."""

  sample_rate_hz: float = 32000.0
  model_dim: int = 256
  stem_blocks: int = 4
  decoder_blocks: int = 8
  stem_mlp_dim: int = 512
  decoder_mlp_dim: int = 768
  n_fft: int = 1024
  hop_length: int = 128
  sai_hop: int = 682
  sai_win_len: int = 1365
  amp_scale: float = 1.0
  log_scale: float = 0.02
  output_scale: float = 1.0
  output_highpass_hz: float = 20.0
  n_harmonics: int = 48
  noise_std: float = 0.35

  def __post_init__(self) -> None:
    """Validates architectural hyperparameters."""
    if self.n_fft not in (512, 1024) or self.hop_length != 128:
      raise ValueError(
        "SaiDirectInverter requires n_fft in (512, 1024) and hop_length=128."
      )
    if self.model_dim < 1 or self.stem_blocks < 1 or self.decoder_blocks < 1:
      raise ValueError(
        "model_dim, stem_blocks, and decoder_blocks must be >= 1."
      )
    if self.n_harmonics < 1:
      raise ValueError("n_harmonics must be >= 1.")

  def to_json(self) -> str:
    """Serializes the configuration to a sorted JSON string."""
    return json.dumps(dataclasses.asdict(self), sort_keys=True)

  @classmethod
  def from_json(cls, payload: str) -> "SaiInverterConfig":
    """Deserializes a configuration saved via :meth:`to_json`."""
    return cls(**json.loads(payload))


class SaiDirectInverter(nn.Module):
  """Causal Vocos + Harmonic-Source generative SAI-to-audio inverter."""

  config: SaiInverterConfig = dataclasses.field(
    default_factory=SaiInverterConfig
  )

  @nn.compact
  def __call__(
    self,
    sai: jnp.ndarray,
    *,
    target_samples: int | None = None,
    noise_key: jax.Array | None = None,
  ) -> jnp.ndarray:
    """Synthesizes audio ``(B, target_samples)`` from SAI ``(B, F, L, C)``."""
    cfg = self.config
    if sai.ndim != 4:
      raise ValueError(
        f"Expected sai of shape (batch, frames, lags, channels), got {sai.shape}."
      )
    b, f_sai, n_lags, c = sai.shape
    if f_sai < 1:
      raise ValueError("SAI must contain at least 1 frame.")
    t_target = (
      int(target_samples)
      if target_samples is not None
      else cfg.sai_hop * max(f_sai - 1, 0) + cfg.sai_win_len
    )

    lag_center = n_lags // 2
    s_pos = jnp.maximum(sai[:, :, lag_center:, :], 0.0)  # (B, F, L_pos, C)
    l_pos = s_pos.shape[2]
    sqrt_s = jnp.sqrt(s_pos + 1e-12)
    amp = sqrt_s / max(float(cfg.amp_scale), 1e-8)
    log_amp = jnp.log1p(sqrt_s / max(float(cfg.log_scale), 1e-8))
    frame_env = jnp.sqrt(
      jnp.mean(s_pos[:, :, 0, :], axis=-1, keepdims=True) + 1e-12
    ) / max(float(cfg.amp_scale), 1e-8)

    e0 = amp[:, :, 0, :]
    log_e0 = log_amp[:, :, 0, :]

    # Multi-scale normalized lag profile across 16 fine + 16 mid/pitch bins.
    max_lag_feat = min(513, l_pos)
    norm_lag_raw = amp[:, :, 1:max_lag_feat, :] / (amp[:, :, :1, :] + 1e-4)
    if norm_lag_raw.shape[2] < 512:
      norm_lag = jnp.pad(
        norm_lag_raw, ((0, 0), (0, 0), (0, 512 - norm_lag_raw.shape[2]), (0, 0))
      )
    else:
      norm_lag = norm_lag_raw[:, :, :512, :]
    fine_bins = (
      norm_lag[:, :, :64, :].reshape((b, f_sai, 16, 4, c)).mean(axis=3)
    )
    mid_bins = (
      norm_lag[:, :, 64:384, :].reshape((b, f_sai, 16, 20, c)).mean(axis=3)
    )
    lag_summary = jnp.concatenate([fine_bins, mid_bins], axis=2).reshape(
      (b, f_sai, 32 * c)
    )
    lag_proj = nn.Dense(cfg.model_dim, name="lag_proj")(lag_summary)

    # High-CF unresolved SACF pitch + voicing extraction (harmonic sum, stop-grad).
    c_hi = max(1, (50 * c) // 77)
    s_hi = s_pos[:, :, :, :c_hi]
    e0_hi = s_hi[:, :, :1, :]
    w_hi = e0_hi / (
      jnp.mean(e0_hi, axis=-1, keepdims=True) * 0.25 + e0_hi + 1e-8
    )
    r_hi = jnp.sum(w_hi * (s_hi / (e0_hi + 1e-8)), axis=-1) / (
      jnp.sum(w_hi, axis=-1) + 1e-8
    )
    e0_all = s_pos[:, :, :1, :]
    r_all = jnp.mean(s_pos / (e0_all + 1e-8), axis=-1)

    fs = float(cfg.sample_rate_hz)
    min_lag = max(16, min(l_pos // 2 - 2, int(round(64.0 * fs / 32000.0))))
    max_lag = min(
      l_pos // 2 - 1, max(min_lag + 1, int(round(335.0 * fs / 32000.0)))
    )
    lags_idx = jnp.arange(min_lag, max_lag + 1, dtype=jnp.int32)
    lags_f = lags_idx.astype(jnp.float32)
    hsr = (
      0.6 * r_hi[:, :, lags_idx]
      + 0.4 * r_hi[:, :, 2 * lags_idx]
      - 0.08 * (lags_f[None, None, :] / float(max_lag))
    )
    tau_weights = jax.nn.softmax(150.0 * hsr, axis=-1)
    tau_0 = jnp.sum(tau_weights * lags_f[None, None, :], axis=-1, keepdims=True)
    f0_sacf_hz = jax.lax.stop_gradient(fs / jnp.maximum(tau_0, 1.0))
    v_sacf = jax.lax.stop_gradient(
      jnp.clip(jnp.max(r_hi[:, :, lags_idx], axis=-1, keepdims=True), 0.0, 1.5)
    )

    r_hi_400 = _pad_or_slice_lag(r_hi, 400)
    r_all_256 = _pad_or_slice_lag(r_all, 256)
    r_hi_proj = nn.gelu(nn.Dense(128, name="r_hi_proj")(r_hi_400))
    r_all_proj = nn.gelu(nn.Dense(96, name="r_all_proj")(r_all_256))

    h_frame = nn.Dense(cfg.model_dim, name="stem_in")(
      jnp.concatenate(
        [
          lag_proj,
          e0,
          log_e0,
          r_hi_proj,
          r_all_proj,
          v_sacf,
          f0_sacf_hz / 300.0,
        ],
        axis=-1,
      )
    )
    for idx in range(cfg.stem_blocks):
      h_frame = _SymmetricConvNeXt1D(
        dim=cfg.model_dim,
        mlp_dim=cfg.stem_mlp_dim,
        kernel_size=5,
        name=f"stem_block_{idx}",
      )(h_frame)
    h_frame = nn.LayerNorm(epsilon=1e-5, name="stem_out_ln")(h_frame)

    # Frame-center aligned interpolation onto sub-frames (hop=128).
    center_offset = 0.5 * float(cfg.sai_win_len)
    t_pad = int(math.ceil(t_target / cfg.hop_length) * cfg.hop_length)
    f_sub = 1 + t_pad // cfg.hop_length
    u_m = (
      jnp.arange(f_sub, dtype=jnp.float32) * float(cfg.hop_length)
      - center_offset
    ) / float(cfg.sai_hop)
    h_sub = _linear_interp_axis1(h_frame, u_m)
    f0_sacf_sub = _linear_interp_axis1(f0_sacf_hz, u_m)

    for idx in range(cfg.decoder_blocks):
      h_sub = _SymmetricConvNeXt1D(
        dim=cfg.model_dim,
        mlp_dim=cfg.decoder_mlp_dim,
        kernel_size=7,
        name=f"dec_block_{idx}",
      )(h_sub)
    h_dec = nn.LayerNorm(epsilon=1e-5, name="dec_out_ln")(h_sub)

    # DDSP harmonic bank + COLA STFT-filtered continuous white noise heads.
    n_harm = int(cfg.n_harmonics)
    harm_amps_sub = nn.softplus(nn.Dense(n_harm, name="harm_amp_head")(h_dec))
    harm_phase_sub = jnp.float32(math.pi) * jnp.tanh(
      nn.Dense(
        n_harm,
        kernel_init=nn.initializers.zeros_init(),
        name="harm_phase_head",
      )(h_dec)
    )
    delta_f0_sub = nn.Dense(
      1, kernel_init=nn.initializers.zeros_init(), name="f0_nudge_head"
    )(h_dec)
    f0_sub = jnp.clip(
      f0_sacf_sub * jnp.exp(0.08 * jnp.tanh(delta_f0_sub)),
      50.0,
      min(650.0, 0.45 * fs),
    )

    n_bins = cfg.n_fft // 2 + 1
    noise_mag_sub = nn.softplus(
      nn.Dense(n_bins, name="noise_filter_head")(h_dec) - 1.0
    )
    log_gain_sub = jnp.clip(
      nn.Dense(1, kernel_init=nn.initializers.zeros_init(), name="gain_head")(
        h_dec
      ),
      -5.5,
      4.5,
    )

    u_sub_s = jnp.arange(t_target, dtype=jnp.float32) / float(cfg.hop_length)
    f0_samp = _linear_interp_axis1(f0_sub, u_sub_s)[..., 0]
    harm_amps_samp = _linear_interp_axis1(harm_amps_sub, u_sub_s)
    harm_phase_samp = _linear_interp_axis1(harm_phase_sub, u_sub_s)
    log_gain_samp = _linear_interp_axis1(log_gain_sub, u_sub_s)[..., 0]

    # Scale backward grad through phase integration so decoder grads stay balanced.
    f0_samp_sg = _scale_grad(f0_samp, 0.01)
    phi = _phase_from_freq_f32(f0_samp_sg, fs)
    k_vec = jnp.arange(1, n_harm + 1, dtype=jnp.float32)[None, None, :]
    nyq = (k_vec * f0_samp_sg[:, :, None] < 0.48 * fs).astype(jnp.float32)
    psi_k = -0.25 * (k_vec**1.3)
    y_harm = jnp.sum(
      harm_amps_samp
      * nyq
      * jnp.sin(
        jnp.float32(2.0 * math.pi) * k_vec * phi[:, :, None]
        + psi_k
        + harm_phase_samp
      ),
      axis=-1,
    ) / math.sqrt(float(n_harm))

    if noise_key is not None:
      noise_wav = jax.random.normal(noise_key, (b, t_pad), dtype=jnp.float32)
    else:
      fixed_noise = jax.random.normal(
        jax.random.PRNGKey(0), (1, t_pad), dtype=jnp.float32
      )
      noise_wav = jnp.broadcast_to(fixed_noise, (b, t_pad))

    spec_noise = (
      _stft_hop128_complex(noise_wav, n_fft=cfg.n_fft) * noise_mag_sub
    )
    y_noise = _cola_istft_hop128(
      spec_noise, n_fft=cfg.n_fft, target_samples=t_target
    )

    u_env_s = (jnp.arange(t_target, dtype=jnp.float32) - center_offset) / float(
      cfg.sai_hop
    )
    env_samp = _linear_interp_axis1(frame_env, u_env_s)[..., 0]
    y = (
      (y_harm + float(cfg.noise_std) * y_noise)
      * float(cfg.output_scale)
      * env_samp
      * jnp.exp(log_gain_samp)
    )
    if cfg.output_highpass_hz > 0.0:
      y = vocoder_losses.causal_highpass(
        y, cutoff_hz=cfg.output_highpass_hz, fs=cfg.sample_rate_hz
      )
    return y


def save_sai_checkpoint(
  path: str | pathlib.Path,
  params: Mapping[str, Any],
  config: SaiInverterConfig,
) -> None:
  """Saves Phase 5b SAI inverter ``params`` and ``config`` to ``.npz``."""
  out_path = pathlib.Path(path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  flat = traverse_util.flatten_dict(params, sep="/")
  payload: dict[str, np.ndarray] = {
    k: np.asarray(v, dtype=np.float32) for k, v in flat.items()
  }
  payload["__config_json__"] = np.asarray(config.to_json())
  np.savez_compressed(out_path, **cast(dict[str, Any], payload))


@functools.lru_cache(maxsize=8)
def _load_sai_cached(
  resolved_path: str, mtime_ns: int
) -> tuple[SaiDirectInverter, dict[str, Any], SaiInverterConfig]:
  del mtime_ns
  with np.load(resolved_path, allow_pickle=False) as data:
    config = SaiInverterConfig.from_json(str(data["__config_json__"].item()))
    flat = {
      k: jnp.asarray(data[k], dtype=jnp.float32)
      for k in data.files
      if k != "__config_json__"
    }
  params = traverse_util.unflatten_dict(flat, sep="/")
  return SaiDirectInverter(config=config), params, config


def load_sai_checkpoint(
  path: str | pathlib.Path,
) -> tuple[SaiDirectInverter, dict[str, Any], SaiInverterConfig]:
  """Loads a saved Phase 5b SAI checkpoint ``.npz`` file."""
  p = pathlib.Path(path).resolve()
  if not p.is_file():
    raise FileNotFoundError(f"Phase 5b SAI checkpoint not found: {path}")
  return _load_sai_cached(str(p), p.stat().st_mtime_ns)


@functools.partial(jax.jit, static_argnames=("model", "target_samples"))
def _apply_sai_jit(
  model: SaiDirectInverter,
  params: Mapping[str, Any],
  sai: jnp.ndarray,
  target_samples: int | None,
) -> jnp.ndarray:
  return cast(
    jnp.ndarray,
    model.apply(params, sai, target_samples=target_samples),
  )


def invert_sai_direct(
  target_sai: np.ndarray | jnp.ndarray,
  checkpoint_path: str | pathlib.Path,
  *,
  target_samples: int | None = None,
) -> np.ndarray:
  """Inverts an SAI `(F, L, C)` or `(B, F, L, C)` directly to audio."""
  model, params, _ = load_sai_checkpoint(checkpoint_path)
  sai_np = np.asarray(target_sai, dtype=np.float32)
  unbatched = sai_np.ndim == 3
  sai_4d = sai_np[None, :, :, :] if unbatched else sai_np
  if sai_4d.ndim != 4:
    raise ValueError(f"target_sai must be 3D or 4D, got shape {sai_np.shape}.")
  y = np.asarray(
    _apply_sai_jit(
      model,
      params,
      jnp.asarray(sai_4d, dtype=jnp.float32),
      target_samples,
    ),
    dtype=np.float32,
  )
  return y[0] if unbatched else y


def default_strobe_sai_params(
  sample_rate_hz: float = 32000.0,
  num_channels: int = 77,
) -> dict[str, int]:
  """Returns canonical CARFAC strobe-SAI parameters for ``sample_rate_hz``."""
  hop = max(1, int(round(0.01 * float(sample_rate_hz))))
  sai_width = max(hop + 2, int(round(0.025 * float(sample_rate_hz))))
  return {
    "num_channels": int(num_channels),
    "sai_width": sai_width,
    "future_lags": sai_width // 2 - 1,
    "num_triggers_per_frame": 2,
    "trigger_window_width": max(sai_width + 1, 2 * hop),
    "input_segment_width": hop,
    "num_ears": 1,
  }


def _strobe_padding(
  n_samples: int,
  p_dict: Mapping[str, int],
) -> tuple[int, int, int, int]:
  """Returns ``(n_frames, lead_frames, pre_pad, post_pad)`` for strobe SAI."""
  hop = int(p_dict["input_segment_width"])
  n_frames = max(1, math.ceil(n_samples / hop))
  delay = (
    int(p_dict["future_lags"])
    + max(0, int(p_dict["trigger_window_width"]) - hop) // 2
  )
  lead = math.ceil(delay / hop)
  pre = lead * hop - delay
  post = (n_frames + lead) * hop - pre - n_samples
  return n_frames, lead, pre, post


def _strobe_trigger_window_f32(trigger_window_width: int) -> np.ndarray:
  """Builds CARFAC sine trigger window matching ``carfac.jax.sai.design_sai``."""
  n = int(trigger_window_width)
  start = math.pi / n
  step = (math.pi - start) / max(n - 1, 1)
  win = [math.sin(start + i * step) for i in range(n - 1)]
  win.append(math.sin(math.pi))
  return np.asarray(win, dtype=np.float32)


def _run_strobe_sai_pure_jax(
  padded_naps: jnp.ndarray,
  p_dict: Mapping[str, int],
  trigger_window: jnp.ndarray,
) -> jnp.ndarray:
  """Pure-JAX ``jax.lax.scan`` implementation of monaural CARFAC strobe SAI."""
  batch, total_samples, n_ch = padded_naps.shape
  hop = int(p_dict["input_segment_width"])
  sai_w = int(p_dict["sai_width"])
  future_lags = int(p_dict["future_lags"])
  n_trig = int(p_dict["num_triggers_per_frame"])
  tw = int(p_dict["trigger_window_width"])
  buf_w = sai_w + int((1.0 + float(n_trig - 1) / 2.0) * tw)
  win_hop = tw // 2
  win_start = (buf_w - tw) - (n_trig - 1) * win_hop
  win_range_start = win_start - future_lags
  offset_range_start = 1 + win_start - sai_w

  fallback_idx = jnp.argmax(trigger_window)
  fallback_val = trigger_window[fallback_idx]
  n_tot = total_samples // hop
  segments = jnp.transpose(
    padded_naps.reshape((batch, n_tot, hop, n_ch)), (1, 0, 3, 2)
  )

  def _stabilize_ch(ch_in: jax.Array, ch_out: jax.Array) -> jax.Array:
    out = ch_out
    for w_idx in range(n_trig):
      off = w_idx * win_hop
      start = win_range_start + off
      windowed = ch_in[start : start + tw] * trigger_window
      p_idx = jnp.argmax(windowed)
      p_val = windowed[p_idx]
      use_fb = p_val <= 0.0
      trig_t = jnp.where(use_fb, fallback_idx, p_idx) + off
      p_eff = jnp.where(use_fb, fallback_val, p_val)
      alpha = (0.025 + p_eff) / (0.5 + p_eff)
      aligned = jax.lax.dynamic_slice_in_dim(
        ch_in, trig_t + offset_range_start, sai_w
      )
      out = (1.0 - alpha) * out + alpha * aligned
    return out

  def _scan_step(
    carry: tuple[jax.Array, jax.Array], seg: jax.Array
  ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
    prev_in, prev_out = carry
    in_buf = jnp.concatenate([prev_in[..., hop:], seg], axis=-1)
    out_buf = jax.vmap(jax.vmap(_stabilize_ch))(in_buf, prev_out)
    return (in_buf, out_buf), out_buf

  init_state = (
    jnp.zeros((batch, n_ch, buf_w), dtype=jnp.float32),
    jnp.zeros((batch, n_ch, sai_w), dtype=jnp.float32),
  )
  _, frames = jax.lax.scan(_scan_step, init_state, segments)
  return jnp.transpose(frames, (1, 0, 3, 2))


def build_strobe_sai_extractor(
  sample_rate_hz: float = 32000.0,
  num_channels: int = 77,
  mesh: jax.sharding.Mesh | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Builds a JIT-safe strobe-SAI extractor ``naps -> sai (B, F, W, C)``.

  Args:
    sample_rate_hz: Sample rate in Hz matching the NAP sequence.
    num_channels: Cochlear channel count ``C``.
    mesh: Optional TPU/device mesh onto which static window weights are replicated.

  Returns:
    Pure JAX callable safe inside ``@jax.jit`` and ``jax.lax.scan``.
  """
  p_dict = default_strobe_sai_params(sample_rate_hz, num_channels)
  try:
    carfac_jax_sai = importlib.import_module("carfac.jax.sai")
  except ImportError:
    carfac_jax_sai = None

  if carfac_jax_sai is not None and hasattr(carfac_jax_sai, "run_sai"):
    sai_mod: Any = carfac_jax_sai
    sai_p = sai_mod.SaiDesignParameters(**p_dict)
    with tpu.host_cpu_context():
      w_host = tpu.to_host_tree(sai_mod.design_sai(sai_p))
    w_dev = (
      tpu.replicate_tree(w_host, mesh)
      if mesh is not None
      else jax.tree.map(jnp.asarray, w_host)
    )

    def _extract_carfac_jax(naps: jnp.ndarray) -> jnp.ndarray:
      naps_j = jnp.asarray(naps, dtype=jnp.float32)
      n_samples = int(naps_j.shape[1])
      n_frames, lead, pre, post = _strobe_padding(n_samples, p_dict)
      padded = jnp.pad(naps_j, ((0, 0), (pre, post), (0, 0)))
      raw = jax.vmap(
        lambda n: sai_mod.run_sai(n[..., None], sai_p, w_dev)[0][:, 0]
      )(padded)
      return jnp.transpose(raw, (0, 1, 3, 2))[:, lead : lead + n_frames]

    return _extract_carfac_jax

  win_host = _strobe_trigger_window_f32(p_dict["trigger_window_width"])
  win_dev = cast(
    jnp.ndarray,
    (
      tpu.replicate_tree(win_host, mesh)
      if mesh is not None
      else jnp.asarray(win_host, dtype=jnp.float32)
    ),
  )

  def _extract_pure_jax(naps: jnp.ndarray) -> jnp.ndarray:
    naps_j = jnp.asarray(naps, dtype=jnp.float32)
    n_samples = int(naps_j.shape[1])
    n_frames, lead, pre, post = _strobe_padding(n_samples, p_dict)
    padded = jnp.pad(naps_j, ((0, 0), (pre, post), (0, 0)))
    return _run_strobe_sai_pure_jax(padded, p_dict, win_dev)[
      :, lead : lead + n_frames
    ]

  return _extract_pure_jax


def build_causal_strobe_sai_nap_fn(
  sample_rate_hz: float = 32000.0,
  num_channels: int = 77,
  mesh: jax.sharding.Mesh | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Returns a JIT-safe transform mapping ``(B, T, C)`` NAPs through causal strobe SAI -> NAP.

  Args:
    sample_rate_hz: Sample rate in Hz matching the NAP sequence.
    num_channels: Cochlear channel count ``C``.
    mesh: Optional TPU/device mesh for replicating static SAI window weights.

  Returns:
    Pure JAX callable mapping target NAPs to causal strobe-SAI reconstructed NAPs
    of identical shape ``(B, T, C)``.
  """
  extractor = build_strobe_sai_extractor(
    sample_rate_hz=sample_rate_hz,
    num_channels=num_channels,
    mesh=mesh,
  )

  def _strobe_nap_fn(naps: jnp.ndarray) -> jnp.ndarray:
    sais = extractor(naps)
    return causal_strobe_sai_to_nap(
      sais,
      sample_rate_hz=sample_rate_hz,
      target_samples=int(naps.shape[1]),
    )

  return _strobe_nap_fn


def strobe_sai_from_naps_f32(
  naps: jnp.ndarray | np.ndarray,
  sample_rate_hz: float = 32000.0,
) -> jnp.ndarray:
  """Computes canonical strobe-triggered CARFAC SAIs ``(B, F, W, C)``.

  Uses ``carfac.jax.sai`` on JAX/TPU backends when available, falling back to
  ``carfac.sai.SAI`` when an older ``carfac`` wheel is installed.
  """
  naps_j = jnp.asarray(naps, dtype=jnp.float32)
  if naps_j.ndim != 3:
    raise ValueError(
      f"Expected naps of shape (batch, samples, channels), got {naps_j.shape}."
    )
  batch, n_samples, n_ch = naps_j.shape
  p_dict = default_strobe_sai_params(sample_rate_hz, n_ch)
  hop = p_dict["input_segment_width"]
  n_frames, lead, pre, post = _strobe_padding(n_samples, p_dict)
  padded = jnp.pad(naps_j, ((0, 0), (pre, post), (0, 0)))

  try:
    carfac_jax_sai = importlib.import_module("carfac.jax.sai")
  except ImportError:
    carfac_jax_sai = None

  if carfac_jax_sai is not None and hasattr(carfac_jax_sai, "run_sai"):
    with tpu.host_cpu_context():
      sai_p = carfac_jax_sai.SaiDesignParameters(**p_dict)
      weights = carfac_jax_sai.design_sai(sai_p)
    weights_dev = jax.device_put(weights, naps_j.device)
    raw = jax.vmap(
      lambda n: carfac_jax_sai.run_sai(n[..., None], sai_p, weights_dev)[0][
        :, 0
      ]
    )(padded)
    return jnp.transpose(raw, (0, 1, 3, 2))[:, lead : lead + n_frames]

  carfac_np_sai = importlib.import_module("carfac.sai")
  sai_p_np = carfac_np_sai.SAIParams(**p_dict)
  padded_np = np.asarray(padded, dtype=np.float32)
  total_frames = padded_np.shape[1] // hop
  out = np.zeros((batch, n_frames, p_dict["sai_width"], n_ch), dtype=np.float32)
  for b_idx in range(batch):
    sai_obj = carfac_np_sai.SAI(sai_p_np)
    frames_b = np.zeros(
      (total_frames, p_dict["sai_width"], n_ch), dtype=np.float32
    )
    for f_idx in range(total_frames):
      seg = padded_np[b_idx, f_idx * hop : (f_idx + 1) * hop, :].T
      frames_b[f_idx] = sai_obj.RunSegment(seg).T
    out[b_idx] = frames_b[lead : lead + n_frames]
  return jnp.asarray(out, dtype=jnp.float32)


def causal_strobe_sai_to_nap(
  sai: jnp.ndarray,
  *,
  sample_rate_hz: float = 32000.0,
  target_samples: int | None = None,
  persistence_decay: float | None = None,
) -> jnp.ndarray:
  """Strictly causal single-frame OLA estimator from strobe SAI to NAP.

  Recovers sample-level ``(B, T, C)`` NAPs in a forward-only ``jax.lax.scan``
  over frames ``0..F-1`` using CARFAC's 2-trigger exponential persistence law
  and scale-invariant tonotopic cross-correlation without accessing future
  frames (algorithmic latency of one 10 ms SAI frame).
  """
  sai_j = jnp.asarray(sai, dtype=jnp.float32)
  if sai_j.ndim != 4:
    raise ValueError(
      f"Expected strobe sai of shape (batch, frames, width, ch), got {sai_j.shape}."
    )
  batch, n_frames, sai_width, n_ch = sai_j.shape
  hop = max(1, int(round(0.01 * float(sample_rate_hz))))
  max_shift = max(8, hop // 2)
  l_buf = sai_width + 2 * max_shift
  n_fft = 1 << int(math.ceil(math.log2(l_buf + sai_width)))
  win = _hann_f32(sai_width)[None, :, None]
  win_sq = win * win

  # Future lag window around trigger zero-lag (middle index = sai_width // 2).
  w0 = sai_width // 2
  half_w = max(2, sai_width // 200)
  z_lo = max(0, w0 - half_w)
  z_hi = min(sai_width, w0 + half_w + 1)

  shifts = jnp.arange(-max_shift, max_shift + 1, dtype=jnp.float32)
  lag_pen = jnp.float32(0.02) * ((shifts / float(max(1, max_shift))) ** 2)
  lag_pen = lag_pen[None, :, None]

  frames_t = jnp.swapaxes(sai_j, 0, 1)  # (F, B, W, C)
  b_ax = jnp.arange(batch, dtype=jnp.int32)[:, None, None]
  c_ax = jnp.arange(n_ch, dtype=jnp.int32)[None, None, :]
  pos_ax = jnp.arange(l_buf, dtype=jnp.int32)[None, :, None]

  def _scan_step(
    carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    sai_f: jax.Array,
  ) -> tuple[
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
  ]:
    step_idx, prev_sai, prev_alpha, buf_val, buf_w = carry
    pos_f = jax.nn.relu(sai_f)
    peak_f = jnp.max(pos_f[:, z_lo:z_hi, :], axis=1, keepdims=True)
    alpha_f = jnp.clip(
      (jnp.float32(0.025) + peak_f) / (0.5 + peak_f), 0.05, 1.0
    )
    if persistence_decay is None:
      rem = (1.0 - prev_alpha) * (1.0 - alpha_f)
      dep_f = jnp.where(
        step_idx == 0,
        pos_f,
        jnp.clip(
          (sai_f - rem * prev_sai) / jnp.maximum(1.0 - rem, 0.12),
          0.0,
          jnp.float32(1.5) * pos_f,
        ),
      )
    else:
      dep_f = jnp.where(
        step_idx == 0,
        pos_f,
        jax.nn.relu(sai_f - jnp.float32(persistence_decay) * prev_sai),
      )
    cur_val = dep_f * win
    cur_spec = jnp.conj(jnp.fft.rfft(cur_val, n=n_fft, axis=1))
    buf_spec = jnp.fft.rfft(buf_val, n=n_fft, axis=1)
    cc = jnp.fft.irfft(buf_spec * cur_spec, n=n_fft, axis=1)[
      :, : 2 * max_shift + 1, :
    ]
    buf_norm = jnp.sqrt(
      jnp.mean(buf_val * buf_val, axis=1, keepdims=True) + 1e-12
    )
    cur_norm = jnp.sqrt(
      jnp.mean(cur_val * cur_val, axis=1, keepdims=True) + 1e-12
    )
    cc_n = cc / (jnp.float32(sai_width) * buf_norm * cur_norm + 1e-8)
    c_left = jnp.concatenate([cc_n[:, :, :1], cc_n[:, :, :-1]], axis=-1)
    c_right = jnp.concatenate([cc_n[:, :, 1:], cc_n[:, :, -1:]], axis=-1)
    cc_local = (
      jnp.float32(0.5) * cc_n
      + jnp.float32(0.25) * c_left
      + jnp.float32(0.25) * c_right
    )
    score = (
      cc_local
      + jnp.float32(0.20) * jnp.mean(cc_n, axis=-1, keepdims=True)
      - lag_pen
    )
    best_k = jnp.argmax(score, axis=1)
    shift_k = jnp.where(step_idx == 0, max_shift, best_k)[:, None, :]

    raw_idx = pos_ax - shift_k
    valid = (raw_idx >= 0) & (raw_idx < sai_width)
    safe_idx = jnp.clip(raw_idx, 0, sai_width - 1)
    add_val = jnp.where(valid, cur_val[b_ax, safe_idx, c_ax], 0.0)
    add_w = jnp.where(valid, win_sq[0, safe_idx, 0], 0.0)

    comb_val = buf_val + add_val
    comb_w = buf_w + add_w
    emit = comb_val[:, :hop, :] / jnp.maximum(comb_w[:, :hop, :], 1e-6)
    next_val = jnp.pad(comb_val[:, hop:, :], ((0, 0), (0, hop), (0, 0)))
    next_w = jnp.pad(comb_w[:, hop:, :], ((0, 0), (0, hop), (0, 0)))
    return (step_idx + 1, sai_f, alpha_f, next_val, next_w), emit

  init_carry = (
    jnp.asarray(0, dtype=jnp.int32),
    jnp.zeros((batch, sai_width, n_ch), dtype=jnp.float32),
    jnp.ones((batch, 1, n_ch), dtype=jnp.float32),
    jnp.zeros((batch, l_buf, n_ch), dtype=jnp.float32),
    jnp.zeros((batch, l_buf, n_ch), dtype=jnp.float32),
  )
  (_, _, _, tail_val, tail_w), emitted = jax.lax.scan(
    _scan_step, init_carry, frames_t
  )
  main_nap = jnp.swapaxes(emitted, 0, 1).reshape((batch, n_frames * hop, n_ch))
  tail_nap = tail_val / jnp.maximum(tail_w, 1e-6)
  full_nap = jnp.concatenate([main_nap, tail_nap], axis=1)[:, max_shift:, :]
  t_out = int(target_samples) if target_samples is not None else n_frames * hop
  if full_nap.shape[1] < t_out:
    full_nap = jnp.pad(
      full_nap, ((0, 0), (0, t_out - full_nap.shape[1]), (0, 0))
    )
  return full_nap[:, :t_out, :]


def invert_strobe_sai_causal(
  target_sai: np.ndarray | jnp.ndarray,
  nap_checkpoint_path: str | pathlib.Path,
  *,
  sample_rate_hz: float = 32000.0,
  target_samples: int | None = None,
) -> np.ndarray:
  """Causally reconstructs audio from strobe-triggered CARFAC SAI via NAP."""
  sai_np = np.asarray(target_sai, dtype=np.float32)
  unbatched = sai_np.ndim == 3
  sai_4d = sai_np[None, :, :, :] if unbatched else sai_np
  if sai_4d.ndim != 4:
    raise ValueError(f"target_sai must be 3D or 4D, got shape {sai_np.shape}.")
  est_nap = np.asarray(
    jax.jit(
      lambda s: causal_strobe_sai_to_nap(
        s, sample_rate_hz=sample_rate_hz, target_samples=target_samples
      )
    )(jnp.asarray(sai_4d, dtype=jnp.float32)),
    dtype=np.float32,
  )
  model, params, cfg = nap_inverter.load_direct_checkpoint(nap_checkpoint_path)
  if abs(float(sample_rate_hz) - float(cfg.fs)) > 1.0:
    raise ValueError(
      f"Strobe SAI rate {sample_rate_hz} Hz != NAP checkpoint rate {cfg.fs} Hz."
    )
  out = np.stack(
    [
      nap_inverter.invert_nap_direct(
        est_nap[i], nap_checkpoint_path, align_lookahead=True
      )
      for i in range(est_nap.shape[0])
    ],
    axis=0,
  )
  del model, params
  return out[0] if unbatched else out
