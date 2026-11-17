"""Phase 3 neural residual refiner for CARFAC NAP-to-audio inversion.

Refines the physics-based DSP warm start (``dsp_warm_start``) in a single
feedforward pass conditioned on centered cochleotopic NAP targets and residuals.
A dual-domain architecture combines a frame-rate complex STFT ConvNeXt branch
(synthesizing ``>4 kHz`` harmonic/fricative fine structure and recovering global
high-SPL AGC gain compression) with a lightweight sample-rate 1D U-Net residual
tail.
"""

import dataclasses
import functools
import json
import math
import pathlib
from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from hp_acoustic.models import carfac
from hp_acoustic.models import cochlea

# Default canonical cochleotopic projection bins for 16 kHz speech.
DEFAULT_COCH_BINS = 64
DEFAULT_CF_MIN_HZ = 40.0
DEFAULT_CF_MAX_HZ = 7200.0
_NAP_LOG_SCALE = 0.02
_GLOBAL_CLIP_SILENCE_RMS = 1e-6


@dataclasses.dataclass(frozen=True)
class NeuralRefinerConfig:
  """Architecture and signal configuration for :class:`NapResidualRefiner`.

  Attributes:
    dim: Channel width of the frame-rate STFT + NAP ConvNeXt backbone.
    dilations: Temporal dilation factors for the frame ConvNeXt blocks.
    n_coch_bins: Number of canonical log-spaced cochleotopic frequency bins.
    cf_min_hz: Lowest canonical cochleotopic CF in Hz.
    cf_max_hz: Highest canonical cochleotopic CF in Hz.
    n_fft: Analysis/synthesis STFT window length at ``sample_rate_hz``.
    hop_length: STFT hop size in samples (also aligns UNet stride divisibility).
    unet_base_dim: Base width for the lightweight sample-rate 1D U-Net tail.
    sample_rate_hz: Nominal model sample rate in Hz.
    max_log_gain_down: Lower clamp for the predicted global log-RMS multiplier.
    max_log_gain_up: Upper clamp for the predicted global log-RMS multiplier.
  """

  dim: int = 160
  dilations: tuple[int, ...] = (1, 2, 4, 8, 2, 1)
  n_coch_bins: int = DEFAULT_COCH_BINS
  cf_min_hz: float = DEFAULT_CF_MIN_HZ
  cf_max_hz: float = DEFAULT_CF_MAX_HZ
  n_fft: int = 256
  hop_length: int = 64
  unet_base_dim: int = 24
  sample_rate_hz: float = 16000.0
  max_log_gain_down: float = -2.0
  max_log_gain_up: float = 2.5

  def to_json(self) -> str:
    """Serializes the configuration to a JSON string."""
    data = dataclasses.asdict(self)
    data["dilations"] = list(self.dilations)
    return json.dumps(data, sort_keys=True)

  @classmethod
  def from_json(cls, payload: str) -> "NeuralRefinerConfig":
    """Deserializes configuration fields from JSON."""
    data = json.loads(payload)
    if "dilations" in data and isinstance(data["dilations"], Sequence):
      data["dilations"] = tuple(int(d) for d in data["dilations"])
    return cls(**data)


def channel_projection_matrix(
  channel_cfs_hz: np.ndarray,
  *,
  n_bins: int = DEFAULT_COCH_BINS,
  cf_min_hz: float = DEFAULT_CF_MIN_HZ,
  cf_max_hz: float = DEFAULT_CF_MAX_HZ,
) -> np.ndarray:
  """Builds a convex piecewise-linear log-CF resampling matrix ``(n_ch, n_bins)``.

  Args:
    channel_cfs_hz: Cochlea center frequencies in Hz, shape ``(n_ch,)``.
    n_bins: Target canonical cochleotopic bin count.
    cf_min_hz: Minimum canonical CF (Hz).
    cf_max_hz: Maximum canonical CF (Hz).

  Returns:
    Row-stochastic projection matrix of shape ``(n_ch, n_bins)`` in ``float32``
    such that ``nap @ W`` orders channels ascending from ``cf_min_hz`` to
    ``cf_max_hz``.
  """
  cfs = np.asarray(channel_cfs_hz, dtype=np.float64)
  n_ch = cfs.shape[0]
  if n_ch == 0:
    raise ValueError("channel_cfs_hz must be non-empty")
  if n_ch == 1:
    return np.ones((1, n_bins), dtype=np.float32)

  order = np.argsort(cfs)
  log_src = np.log(np.maximum(cfs[order], 1.0))
  log_tgt = np.log(np.geomspace(cf_min_hz, cf_max_hz, n_bins))

  weights = np.zeros((n_ch, n_bins), dtype=np.float32)
  for k, x in enumerate(log_tgt):
    if x <= log_src[0]:
      weights[order[0], k] = 1.0
    elif x >= log_src[-1]:
      weights[order[-1], k] = 1.0
    else:
      hi = int(np.searchsorted(log_src, x, side="right"))
      lo = max(0, hi - 1)
      hi = min(n_ch - 1, lo + 1)
      span = max(log_src[hi] - log_src[lo], 1e-12)
      alpha = float((x - log_src[lo]) / span)
      weights[order[lo], k] = 1.0 - alpha
      weights[order[hi], k] = alpha
  return weights


def cochlea_cf_projection(
  cochlea_model: cochlea.CochleaModel,
  config: NeuralRefinerConfig,
) -> np.ndarray:
  """Returns the canonical CF projection matrix for ``cochlea_model``."""
  if isinstance(cochlea_model, carfac.CarfacCochleaModel):
    cfs = np.asarray(cochlea_model.channel_frequencies, dtype=np.float64)
  else:
    n_ch = cochlea_model.num_channels
    cfs = np.geomspace(config.cf_min_hz, config.cf_max_hz, n_ch)
  return channel_projection_matrix(
    cfs,
    n_bins=config.n_coch_bins,
    cf_min_hz=config.cf_min_hz,
    cf_max_hz=config.cf_max_hz,
  )


def _compress_nap(nap: jnp.ndarray) -> jnp.ndarray:
  """Compresses signed centered NAP amplitudes logarithmically."""
  return jnp.sign(nap) * jnp.log1p(jnp.abs(nap) / _NAP_LOG_SCALE)


class _ConvNeXt1D(nn.Module):
  """FiLM-conditioned 1D ConvNeXt block with zero-initialized residual tail."""

  dim: int
  dilation: int = 1
  kernel_size: int = 7

  @nn.compact
  def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
    res = x
    h = nn.LayerNorm(epsilon=1e-5)(x)
    pad = ((self.kernel_size - 1) * self.dilation) // 2
    h_pad = jnp.pad(h, ((0, 0), (pad, pad), (0, 0)))
    h = nn.Conv(
      features=self.dim,
      kernel_size=(self.kernel_size,),
      kernel_dilation=(self.dilation,),
      feature_group_count=self.dim,
      padding="VALID",
    )(h_pad)
    film = nn.Dense(2 * self.dim)(nn.gelu(cond))
    scale, shift = jnp.split(film[:, None, :], 2, axis=-1)
    h = h * (1.0 + 0.5 * jnp.tanh(scale)) + shift
    h = nn.Dense(2 * self.dim)(h)
    h = nn.gelu(h)
    h = nn.Dense(self.dim, kernel_init=nn.initializers.zeros_init())(h)
    return res + h


class NapResidualRefiner(nn.Module):
  """Dual-domain (STFT + Time U-Net) neural residual refiner for NAP inversion.

  At initialization (zero-initialized output projections), outputs match
  ``ws_wave`` closely so gradient training starts near the Phase 2 DSP warm
  start rather than noise.
  """

  config: NeuralRefinerConfig = dataclasses.field(
    default_factory=NeuralRefinerConfig
  )

  @nn.compact
  def __call__(
    self,
    ws_wave: jnp.ndarray,
    target_nap: jnp.ndarray,
    ws_nap: jnp.ndarray,
  ) -> jnp.ndarray:
    """Refines ``ws_wave`` given centered cochleotopic NAP tensors.

    Args:
      ws_wave: Warm-start waveforms, shape ``(batch, samples)``.
      target_nap: Centered, canonically projected target NAPs, shape
        ``(batch, samples, n_coch_bins)``.
      ws_nap: Centered, canonically projected warm-start NAPs, shape
        ``(batch, samples, n_coch_bins)``.

    Returns:
      Refined waveforms of shape ``(batch, samples)``.
    """
    cfg = self.config
    batch_size, t_len = ws_wave.shape
    hop = cfg.hop_length
    n_fft = cfg.n_fft
    n_freqs = n_fft // 2 + 1
    pad_tail = (hop - (t_len % hop)) % hop
    t_pad = t_len + pad_tail

    ws_pad = jnp.pad(ws_wave, ((0, 0), (0, pad_tail)))
    tgt_pad = jnp.pad(target_nap, ((0, 0), (0, pad_tail), (0, 0)))
    wsn_pad = jnp.pad(ws_nap, ((0, 0), (0, pad_tail), (0, 0)))

    ws_rms = jnp.sqrt(jnp.mean(ws_wave**2, axis=1, keepdims=True) + 1e-16)
    ws_rms_safe = jnp.maximum(ws_rms, _GLOBAL_CLIP_SILENCE_RMS)
    silence_gate = (ws_rms**2) / (ws_rms**2 + _GLOBAL_CLIP_SILENCE_RMS**2)
    ws_norm = ws_pad / ws_rms_safe

    res_pad = tgt_pad - wsn_pad
    tgt_f = _compress_nap(tgt_pad)
    wsn_f = _compress_nap(wsn_pad)
    res_f = _compress_nap(res_pad)

    # Physical SPL + multi-band cochlear residual summary conditioning.
    n_macro = 8
    band_sz = cfg.n_coch_bins // n_macro
    t_bands = tgt_f[:, :, : n_macro * band_sz].reshape(
      batch_size, t_pad, n_macro, band_sz
    )
    r_bands = res_f[:, :, : n_macro * band_sz].reshape(
      batch_size, t_pad, n_macro, band_sz
    )
    t_raw_b = jnp.maximum(
      tgt_pad[:, :, : n_macro * band_sz].reshape(
        batch_size, t_pad, n_macro, band_sz
      ),
      0.0,
    )
    w_raw_b = jnp.maximum(
      wsn_pad[:, :, : n_macro * band_sz].reshape(
        batch_size, t_pad, n_macro, band_sz
      ),
      0.0,
    )
    t_b = jnp.mean(t_bands, axis=-1)
    r_b = jnp.mean(r_bands, axis=-1)
    t_rms = jnp.sqrt(jnp.mean(t_raw_b**2, axis=(1, 3)) + 1e-8)
    w_rms = jnp.sqrt(jnp.mean(w_raw_b**2, axis=(1, 3)) + 1e-8)

    cond_in = jnp.concatenate(
      [
        (jnp.log10(ws_rms_safe) + 3.0) / 1.5,
        jnp.mean(t_b, axis=1),
        jnp.std(t_b, axis=1),
        jnp.max(t_b, axis=1),
        jnp.mean(r_b, axis=1),
        jnp.std(r_b, axis=1),
        jnp.log((t_rms + 1e-4) / (w_rms + 1e-4)),
      ],
      axis=-1,
    )
    cond = nn.Dense(cfg.dim)(nn.gelu(nn.Dense(cfg.dim)(cond_in)))

    # Analysis STFT on normalized warm start.
    _, _, z_ws = jax.scipy.signal.stft(
      ws_norm,
      fs=cfg.sample_rate_hz,
      nperseg=n_fft,
      noverlap=n_fft - hop,
      nfft=n_fft,
      boundary="zeros",
      padded=True,
      axis=1,
    )
    z_t = jnp.transpose(z_ws, (0, 2, 1))
    spec_feats = jnp.concatenate(
      [z_t.real, z_t.imag, jnp.log1p(jnp.abs(z_t))], axis=-1
    )

    # Factored sample-to-frame NAP encoder matching STFT frame centers.
    nap_stack = jnp.concatenate([tgt_f, wsn_f, res_f], axis=-1)
    nap_low_rank = nn.gelu(nn.Dense(48)(nap_stack))
    half_win = n_fft // 2
    nap_win = jnp.pad(nap_low_rank, ((0, 0), (half_win, half_win), (0, 0)))
    nap_conv = nn.Conv(
      features=48,
      kernel_size=(n_fft,),
      strides=(hop,),
      feature_group_count=48,
      padding="VALID",
    )(nap_win)
    nap_conv = nn.gelu(nn.Dense(cfg.dim // 2)(nap_conv))

    pool_pad = hop
    nap_env_in = jnp.pad(
      jnp.concatenate([tgt_f, res_f], axis=-1),
      ((0, 0), (pool_pad, pool_pad), (0, 0)),
    )
    nap_avg = nn.avg_pool(
      nap_env_in, window_shape=(2 * hop,), strides=(hop,), padding="VALID"
    )
    nap_max = nn.max_pool(
      nap_env_in[:, :, : cfg.n_coch_bins],
      window_shape=(2 * hop,),
      strides=(hop,),
      padding="VALID",
    )

    h = nn.Dense(cfg.dim)(
      jnp.concatenate([spec_feats, nap_conv, nap_avg, nap_max], axis=-1)
    )
    for dil in cfg.dilations:
      h = _ConvNeXt1D(dim=cfg.dim, dilation=dil, kernel_size=7)(h, cond)
    h_norm = nn.LayerNorm(epsilon=1e-5)(h)

    # Global SPL log-gain multiplier (zero-init => delta_g = 0 at init).
    pool = jnp.concatenate(
      [jnp.mean(h_norm, axis=1), jnp.std(h_norm, axis=1), cond], axis=-1
    )
    delta_g = nn.Dense(
      1,
      kernel_init=nn.initializers.zeros_init(),
      bias_init=nn.initializers.zeros_init(),
    )(nn.gelu(nn.Dense(64)(pool)))
    delta_g = jnp.clip(delta_g, cfg.max_log_gain_down, cfg.max_log_gain_up)

    # Complex spectral ratio mask & additive complex STFT residual.
    spec_out = nn.Dense(
      4 * n_freqs,
      kernel_init=nn.initializers.zeros_init(),
      bias_init=nn.initializers.zeros_init(),
    )(h_norm)
    dm_r, dm_i, dr_r, dr_i = jnp.split(spec_out, 4, axis=-1)
    mask = (1.0 + 0.65 * jnp.tanh(dm_r)) + 1j * (0.65 * jnp.tanh(dm_i))
    residual_spec = (dr_r + 1j * dr_i) * silence_gate[:, :, None]
    z_out = jnp.transpose(mask * z_t + residual_spec, (0, 2, 1))
    _, x_spec = jax.scipy.signal.istft(
      z_out,
      fs=cfg.sample_rate_hz,
      nperseg=n_fft,
      noverlap=n_fft - hop,
      nfft=n_fft,
      boundary=True,
      time_axis=-1,
      freq_axis=-2,
    )
    x_spec = x_spec[:, :t_pad]

    # Sample-aligned linear interpolation of frame conditioning to sample rate.
    idx = jnp.arange(t_pad) // hop
    frac = ((jnp.arange(t_pad) % hop).astype(jnp.float32) / float(hop))[
      None, :, None
    ]
    h_small = nn.Dense(16)(h_norm)
    h_up = (1.0 - frac) * h_small[:, idx, :] + frac * h_small[:, idx + 1, :]

    # Lightweight sample-rate 1D U-Net for phase & transient sharpening.
    u_dim = cfg.unet_base_dim
    t_in = jnp.concatenate(
      [
        x_spec[:, :, None],
        ws_norm[:, :, None],
        h_up,
        tgt_f[:, :, ::4],
        res_f[:, :, ::4],
      ],
      axis=-1,
    )
    s0 = nn.gelu(nn.Conv(u_dim, (9,), padding="SAME")(t_in))
    d1 = nn.gelu(nn.Conv(2 * u_dim, (9,), strides=(4,), padding="SAME")(s0))
    b = _ConvNeXt1D(dim=2 * u_dim, dilation=1, kernel_size=7)(d1, cond)
    b = _ConvNeXt1D(dim=2 * u_dim, dilation=2, kernel_size=7)(b, cond)
    u1 = nn.Conv(u_dim, (7,), padding="SAME")(jnp.repeat(b, 4, axis=1)) + s0
    u1 = nn.gelu(u1)
    r_time = nn.Conv(
      1,
      (7,),
      padding="SAME",
      kernel_init=nn.initializers.zeros_init(),
      bias_init=nn.initializers.zeros_init(),
    )(u1)[:, :, 0]

    x_out = (ws_rms_safe * jnp.exp(delta_g)) * (x_spec + silence_gate * r_time)
    return x_out[:, :t_len]


def save_refiner_checkpoint(
  path: str | pathlib.Path,
  params: dict,
  config: NeuralRefinerConfig,
) -> None:
  """Saves ``params`` and ``config`` into a self-contained ``.npz`` checkpoint."""
  out_path = pathlib.Path(path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  flat = traverse_util.flatten_dict(params, sep="/")
  payload: dict[str, np.ndarray] = {
    k: np.asarray(v, dtype=np.float32) for k, v in flat.items()
  }
  payload["__config_json__"] = np.asarray(config.to_json())
  np.savez_compressed(out_path, **payload)


@functools.lru_cache(maxsize=8)
def _load_refiner_cached(
  resolved_path: str, mtime_ns: int
) -> tuple[NapResidualRefiner, dict, NeuralRefinerConfig]:
  del mtime_ns
  with np.load(resolved_path, allow_pickle=False) as data:
    config = NeuralRefinerConfig.from_json(str(data["__config_json__"].item()))
    flat = {
      k: jnp.asarray(data[k], dtype=jnp.float32)
      for k in data.files
      if k != "__config_json__"
    }
  params = traverse_util.unflatten_dict(flat, sep="/")
  model = NapResidualRefiner(config=config)
  return model, params, config


def load_refiner_checkpoint(
  path: str | pathlib.Path,
) -> tuple[NapResidualRefiner, dict, NeuralRefinerConfig]:
  """Loads a saved ``.npz`` checkpoint, caching by resolved path and mtime."""
  p = pathlib.Path(path).resolve()
  if not p.is_file():
    raise FileNotFoundError(f"Neural refiner checkpoint not found: {path}")
  return _load_refiner_cached(str(p), p.stat().st_mtime_ns)


@functools.partial(jax.jit, static_argnames=("model",))
def _apply_refiner_jit(
  model: NapResidualRefiner,
  params: dict,
  ws_wave: jnp.ndarray,
  target_nap_c: jnp.ndarray,
  ws_nap_c: jnp.ndarray,
) -> jnp.ndarray:
  return model.apply(params, ws_wave, target_nap_c, ws_nap_c)


def _refine_single_window(
  model: NapResidualRefiner,
  params: dict,
  ws_win: np.ndarray,
  tgt_win: np.ndarray,
  rest_nap_j: jnp.ndarray,
  proj_w_j: jnp.ndarray,
  cochlea_model: cochlea.CochleaModel,
) -> np.ndarray:
  """Runs one window ``(win_samples, ears)`` through the neural refiner."""
  n_ears = ws_win.shape[1]
  ws_j = jnp.asarray(ws_win, dtype=jnp.float32)
  ws_nap_raw = cochlea_model(ws_j)
  tgt_j = jnp.asarray(tgt_win, dtype=jnp.float32)

  # Project centered NAPs to canonical CF bins and stack ears along batch.
  tgt_canon = jnp.einsum("tce,ck->etk", tgt_j - rest_nap_j, proj_w_j)
  ws_canon = jnp.einsum("tce,ck->etk", ws_nap_raw - rest_nap_j, proj_w_j)
  ws_batch = jnp.transpose(ws_j, (1, 0))
  out_batch = _apply_refiner_jit(model, params, ws_batch, tgt_canon, ws_canon)
  out = np.asarray(jnp.transpose(out_batch, (1, 0)), dtype=np.float32)
  if out.shape != (ws_win.shape[0], n_ears):
    raise AssertionError(f"Unexpected output shape {out.shape}")
  return out


def refine_waveform(
  target_nap: np.ndarray | jnp.ndarray,
  warm_start: np.ndarray | jnp.ndarray,
  cochlea_model: cochlea.CochleaModel,
  checkpoint_path: str | pathlib.Path,
  *,
  chunk_seconds: float | None = 1.5,
  chunk_overlap_s: float = 0.128,
) -> np.ndarray:
  """Applies a trained Phase 3 neural residual refiner to ``warm_start``.

  Args:
    target_nap: Target NAP of shape ``(samples, channels)`` or
      ``(samples, channels, ears)``.
    warm_start: DSP warm-start waveform of shape ``(samples,)`` or
      ``(samples, ears)``.
    cochlea_model: Forward cochlea used to compute ``ws_nap`` and resting floor.
    checkpoint_path: Path to the ``.npz`` checkpoint saved by
      :func:`save_refiner_checkpoint`.
    chunk_seconds: Optional fixed OLA window length in seconds so XLA compiles
      once for arbitrary-duration clips.
    chunk_overlap_s: Cosine-squared overlap blend duration in seconds.

  Returns:
    Refined waveform matching ``warm_start`` shape in ``float32``.
  """
  model, params, config = load_refiner_checkpoint(checkpoint_path)

  ws_np = np.asarray(warm_start, dtype=np.float32)
  is_1d = ws_np.ndim == 1
  ws_2d = ws_np[:, np.newaxis] if is_1d else ws_np

  nap_np = np.asarray(target_nap, dtype=np.float32)
  nap_3d = nap_np[:, :, np.newaxis] if nap_np.ndim == 2 else nap_np
  t_len, _, n_ears = nap_3d.shape

  rest_len = min(512, t_len)
  rest_nap_j = jnp.asarray(
    cochlea_model(jnp.zeros((rest_len, n_ears), dtype=jnp.float32))[-1:],
    dtype=jnp.float32,
  )
  proj_w_j = jnp.asarray(
    cochlea_cf_projection(cochlea_model, config), dtype=jnp.float32
  )

  fs = float(cochlea_model.sample_rate)
  chunk_samples = (
    int(round(chunk_seconds * fs)) if chunk_seconds is not None else t_len
  )
  if chunk_seconds is None or t_len <= chunk_samples:
    out_2d = _refine_single_window(
      model, params, ws_2d, nap_3d, rest_nap_j, proj_w_j, cochlea_model
    )
    return out_2d[:, 0] if is_1d else out_2d

  win_len = max(256, chunk_samples)
  fade_len = min(int(round(chunk_overlap_s * fs)), win_len // 2)
  hop = max(1, win_len - fade_len)
  num_chunks = max(1, math.ceil(max(0, t_len - win_len) / hop) + 1)
  padded_len = (num_chunks - 1) * hop + win_len

  ws_pad = np.zeros((padded_len, n_ears), dtype=np.float32)
  ws_pad[:t_len] = ws_2d
  rest_np = np.asarray(rest_nap_j[0], dtype=np.float32)
  nap_pad = np.broadcast_to(
    rest_np[None, :, :], (padded_len, nap_3d.shape[1], n_ears)
  ).copy()
  nap_pad[:t_len] = nap_3d

  if fade_len > 0:
    ramp = (
      np.sin(0.5 * math.pi * np.arange(fade_len, dtype=np.float32) / fade_len)
      ** 2
    )[:, None]
  else:
    ramp = np.zeros((0, 1), dtype=np.float32)

  out_pad = np.zeros((padded_len, n_ears), dtype=np.float32)
  for chunk_idx in range(num_chunks):
    start = chunk_idx * hop
    end = start + win_len
    win_out = _refine_single_window(
      model,
      params,
      ws_pad[start:end],
      nap_pad[start:end],
      rest_nap_j,
      proj_w_j,
      cochlea_model,
    )
    if fade_len > 0 and chunk_idx > 0:
      out_pad[start : start + fade_len] = (
        out_pad[start : start + fade_len] * (1.0 - ramp)
        + win_out[:fade_len] * ramp
      )
      out_pad[start + fade_len : end] = win_out[fade_len:]
    else:
      out_pad[start:end] = win_out

  out_2d = out_pad[:t_len]
  return out_2d[:, 0] if is_1d else out_2d
