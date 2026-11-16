"""Inverts CARFAC neural activity patterns (NAPs) back to audio waveforms.

Combines a physics-based basilar-membrane matched-filter warm start with
scale-invariant JAX gradient optimization (Adam + L-BFGS) through a
differentiable ``CochleaModel``. Supports both same-cochlea round-trip
reconstruction (``mode="exact"``) and cross-cochlea hearing-loss auralization
(``mode="perceptual"``).
"""

import dataclasses
import logging
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from carfac.jax import carfac as carfac_jax
from hp_acoustic.filters import conductive
from hp_acoustic.filters import middle_ear
from hp_acoustic.losses import waveform_losses
from hp_acoustic.models import carfac
from hp_acoustic.models import cochlea
from hp_acoustic.models import sai

logger = logging.getLogger(__name__)

InversionMode = Literal["exact", "perceptual"]

# Small-signal impulse amplitude for measuring the linear CARFAC cascade IR.
_IMPULSE_AMPLITUDE = 1e-4

# Power threshold below which a centered NAP counts as resting silence.
_SILENCE_NAP_PEAK = 1e-7

# Default envelope smoothing cutoffs and bandpass regularization bounds.
_ENVELOPE_LPF_HZ = 250.0
_TAIL_LEAKAGE_HPF_HZ = 1200.0
_ENVELOPE_FAST_S = 0.002
_ENVELOPE_SLOW_S = 0.008


@dataclasses.dataclass(frozen=True)
class InversionConfig:
  """Configuration for inverting a neural activity pattern to an audio waveform.

  Attributes:
    mode: Optimization objective. ``"exact"`` minimizes scale-normalized NAP MSE
      (plus an optional high-frequency envelope anchor) for same-cochlea
      round-trip reconstruction. ``"perceptual"`` matches multi-scale NAP
      envelopes and stabilized auditory images (SAIs) for cross-cochlea
      hearing-loss auralization where traveling-wave group delays differ.
    adam_steps: Number of initial Adam iterations to align phase and amplitude.
    lbfgs_steps: Number of RMS-normalized L-BFGS refinement iterations.
    use_warm_start: Whether to initialize with the physics-based DSP warm
      start before gradient optimization.
    min_warm_start_cf_hz: Lowest CARFAC channel CF (Hz) included in the linear
      DSP warm start, excluding apical channels dominated by quadratic
      ``ohc_nlf`` combination-tone distortion.
    hf_crossover_hz: Channel CF (Hz) above which warm-start synthesis switches
      from phase-locked bandpass inversion to envelope-modulated noise.
    cutoff_low_hz: High-pass taper edge (Hz) below which CARFAC's 20 Hz AC
      coupler and elevated low-frequency thresholds permit sub-audible drift.
    cutoff_high_hz: Optional low-pass taper edge (Hz). Defaults to
      ``min(0.48 * sample_rate_hz, 1.15 * max_cf_hz)`` for CARFAC models.
    hf_env_weight: Weight of the >3.5 kHz smoothed-envelope term in ``"exact"``
      mode.
    adam_lr_scale: Initial Adam step size as a fraction of the initial waveform
      RMS.
    lbfgs_memory_size: History size for L-BFGS inverse Hessian approximation.
    seed: Random seed for high-frequency warm-start noise carriers.
  """

  mode: InversionMode = "exact"
  adam_steps: int = 80
  lbfgs_steps: int = 100
  use_warm_start: bool = True
  min_warm_start_cf_hz: float = 250.0
  hf_crossover_hz: float = 4200.0
  cutoff_low_hz: float = 60.0
  cutoff_high_hz: float | None = None
  hf_env_weight: float = 0.005
  adam_lr_scale: float = 0.08
  lbfgs_memory_size: int = 15
  seed: int = 0

  def __post_init__(self) -> None:
    """Validates configuration fields at construction."""
    if self.mode not in ("exact", "perceptual"):
      raise ValueError(
        f"mode must be 'exact' or 'perceptual', got {self.mode!r}"
      )
    if self.adam_steps < 0:
      raise ValueError(
        f"adam_steps must be non-negative, got {self.adam_steps}"
      )
    if self.lbfgs_steps < 0:
      raise ValueError(
        f"lbfgs_steps must be non-negative, got {self.lbfgs_steps}"
      )
    if self.min_warm_start_cf_hz <= 0.0:
      raise ValueError(
        f"min_warm_start_cf_hz must be positive, got {self.min_warm_start_cf_hz}"
      )
    if self.hf_crossover_hz <= self.min_warm_start_cf_hz:
      raise ValueError(
        "hf_crossover_hz must exceed min_warm_start_cf_hz, got "
        f"{self.hf_crossover_hz} <= {self.min_warm_start_cf_hz}"
      )
    if self.cutoff_low_hz < 0.0:
      raise ValueError(
        f"cutoff_low_hz must be non-negative, got {self.cutoff_low_hz}"
      )
    if self.cutoff_high_hz is not None and self.cutoff_high_hz <= 0.0:
      raise ValueError(
        f"cutoff_high_hz must be positive when set, got {self.cutoff_high_hz}"
      )
    if self.hf_env_weight < 0.0:
      raise ValueError(
        f"hf_env_weight must be non-negative, got {self.hf_env_weight}"
      )
    if self.adam_lr_scale <= 0.0:
      raise ValueError(
        f"adam_lr_scale must be positive, got {self.adam_lr_scale}"
      )
    if self.lbfgs_memory_size < 1:
      raise ValueError(
        f"lbfgs_memory_size must be at least 1, got {self.lbfgs_memory_size}"
      )


@dataclasses.dataclass(frozen=True)
class NapInversionResult:
  """Result of inverting a neural activity pattern back to audio.

  Attributes:
    waveform: Final reconstructed audio, shape ``(num_samples,)`` for 2D NAP
      inputs or ``(num_samples, num_ears)`` for 3D NAP inputs.
    warm_start: Initial DSP warm-start audio before gradient optimization, same
      shape as ``waveform``.
    reconstructed_nap: Forward cochlea output evaluated on ``waveform``, same
      shape as the input ``target_nap``.
    loss_history: Objective value at each optimization step (Adam followed by
      L-BFGS).
    final_loss: Minimum objective value achieved across optimization (or initial
      objective when optimization steps are zero).
    sample_rate_hz: Cochlea sample rate in Hz.
  """

  waveform: np.ndarray
  warm_start: np.ndarray
  reconstructed_nap: np.ndarray
  loss_history: tuple[float, ...]
  final_loss: float
  sample_rate_hz: float


def _normalize_target_nap(
  target_nap: np.ndarray | jnp.ndarray,
  cochlea_model: cochlea.CochleaModel,
) -> tuple[np.ndarray, bool]:
  """Validates ``target_nap`` and normalizes it to ``(samples, channels, ears)``."""
  nap_np = np.asarray(target_nap, dtype=np.float32)
  if nap_np.ndim == 2:
    is_2d = True
    nap_3d = nap_np[:, :, np.newaxis]
  elif nap_np.ndim == 3:
    is_2d = False
    nap_3d = nap_np
  else:
    raise ValueError(
      "target_nap must have 2 dimensions (samples, channels) or 3 dimensions "
      f"(samples, channels, ears), got shape {nap_np.shape}"
    )

  num_samples, num_channels, num_ears = nap_3d.shape
  if num_samples < 16:
    raise ValueError(
      f"target_nap must have at least 16 samples, got {num_samples}"
    )
  if num_channels != cochlea_model.num_channels:
    raise ValueError(
      f"target_nap channel count ({num_channels}) must match cochlea_model "
      f"num_channels ({cochlea_model.num_channels})"
    )
  if num_ears < 1:
    raise ValueError(f"target_nap ear count must be at least 1, got {num_ears}")
  if isinstance(cochlea_model, carfac.CarfacCochleaModel):
    model_ears = len(cochlea_model.hypers.ears)
    if num_ears != model_ears:
      raise ValueError(
        f"target_nap ear count ({num_ears}) must match CarfacCochleaModel "
        f"ear count ({model_ears})"
      )
  if not np.all(np.isfinite(nap_3d)):
    raise ValueError("target_nap must contain only finite values")
  return nap_3d, is_2d


def _measure_carfac_bm_irs(
  cochlea_model: carfac.CarfacCochleaModel,
  ir_len: int,
  ohc_scale: float = 1.0,
) -> np.ndarray:
  """Measures open-loop CARFAC basilar-membrane impulse responses.

  Args:
    cochlea_model: Target CARFAC cochlea model.
    ir_len: Impulse response length in samples.
    ohc_scale: Multiplier in ``(0, 1]`` applied to OHC undamping so loud
      signals use an AGC-damped operating Q for the linear inverse filter.

  Returns:
    Array of shape ``(ir_len, num_channels, num_ears)``.
  """
  num_ears = len(cochlea_model.hypers.ears)
  imp = jnp.zeros((ir_len, num_ears), dtype=jnp.float32)
  imp = imp.at[0, :].set(_IMPULSE_AMPLITUDE)
  w = conductive.apply(imp, cochlea_model.conductive_filter)
  w = middle_ear.apply(w, cochlea_model.middle_ear_filter)

  state_ears = []
  for ear_idx in range(num_ears):
    ear_hypers = cochlea_model.hypers.ears[ear_idx]
    ear_weights = cochlea_model.weights.ears[ear_idx]
    ear_state = cochlea_model.initial_state.ears[ear_idx]
    undamping = ear_weights.car.ohc_health * ohc_scale
    zb = ear_hypers.car.zr_coeffs * undamping
    g = carfac_jax.stage_g(ear_weights.car, undamping)
    car_state = dataclasses.replace(ear_state.car, zb_memory=zb, g_memory=g)
    state_ears.append(dataclasses.replace(ear_state, car=car_state))

  state_proxy = dataclasses.replace(
    cochlea_model.initial_state, ears=state_ears
  )
  out = carfac_jax.run_segment(
    w,
    cochlea_model.hypers,
    cochlea_model.weights,
    state_proxy,
    open_loop=True,
  )
  return np.asarray(out.bm, dtype=np.float64) / _IMPULSE_AMPLITUDE


def _ihc_lpf_inverse(
  cochlea_model: carfac.CarfacCochleaModel,
  freqs_hz: np.ndarray,
  ear_idx: int,
  sample_rate_hz: float,
) -> np.ndarray:
  """Returns the bounded Wiener inverse of CARFAC's IHC + synapse LPF."""
  ear_hypers = cochlea_model.hypers.ears[ear_idx]
  ear_weights = cochlea_model.weights.ears[ear_idx]
  if ear_hypers.ihc.ihc_style == 0:
    return np.ones_like(freqs_hz, dtype=np.complex128)

  z_inv = np.exp(-1j * 2.0 * math.pi * freqs_hz / sample_rate_hz)
  alpha_ihc = float(ear_weights.ihc.lpf_coeff)
  h_ihc = alpha_ihc / (1.0 - (1.0 - alpha_ihc) * z_inv)

  if ear_hypers.ihc.ihc_style == 1:
    h_lpf = h_ihc * h_ihc
  else:
    h_lpf = h_ihc
    if ear_hypers.syn.do_syn and ear_weights.syn is not None:
      alpha_syn = float(ear_weights.syn.lpf_coeff)
      h_syn = alpha_syn / (1.0 - (1.0 - alpha_syn) * z_inv)
      h_lpf = h_lpf * h_syn

  return np.conj(h_lpf) / (np.abs(h_lpf) ** 2 + 0.05)


def _calibrate_scalar_gain(
  unit_waveform: jnp.ndarray,
  target_centered_j: jnp.ndarray,
  rest_nap_j: jnp.ndarray,
  cochlea_model: cochlea.CochleaModel,
  center_rms: float | None = None,
) -> float:
  """Finds the scalar gain minimizing per-channel positive NAP RMS mismatch."""
  target_ch_rms = jnp.sqrt(
    jnp.mean(jnp.maximum(target_centered_j, 0.0) ** 2, axis=(0, 2)) + 1e-8
  )

  @jax.jit
  def eval_rms_err(gain: jnp.ndarray) -> jnp.ndarray:
    pred_centered = cochlea_model(gain * unit_waveform) - rest_nap_j
    pred_ch_rms = jnp.sqrt(
      jnp.mean(jnp.maximum(pred_centered, 0.0) ** 2, axis=(0, 2)) + 1e-8
    )
    return jnp.mean((pred_ch_rms - target_ch_rms) ** 2)

  if center_rms is None:
    coarse = np.geomspace(5e-5, 0.4, 14)
    best_gain = float(
      coarse[
        int(np.argmin([float(eval_rms_err(jnp.float32(g))) for g in coarse]))
      ]
    )
    fine = np.geomspace(best_gain * 0.65, best_gain * 1.55, 7)
  else:
    fine = np.geomspace(max(center_rms * 0.5, 1e-6), center_rms * 2.0, 7)
  return float(
    fine[int(np.argmin([float(eval_rms_err(jnp.float32(g))) for g in fine]))]
  )


def _carfac_dsp_warm_start(
  nap_3d: np.ndarray,
  rest_nap: np.ndarray,
  cochlea_model: carfac.CarfacCochleaModel,
  config: InversionConfig,
) -> np.ndarray:
  """Constructs a delay- and gain-calibrated waveform from CARFAC NAPs."""
  t_len, n_ch, n_ears = nap_3d.shape
  fs = float(cochlea_model.sample_rate)
  cfs = np.asarray(cochlea_model.channel_frequencies, dtype=np.float64)
  nap_c = nap_3d.astype(np.float64) - rest_nap[None, :, :]

  # Estimate AGC-damped proxy OHC scale from overall NAP drive.
  nap_rms = float(np.sqrt(np.mean(nap_c**2)))
  ohc_scale = min(1.0, 1.0 / (1.0 + 1.2 * max(0.0, nap_rms - 0.3)))

  ir_len = min(
    4096, max(1024, 1 << int(math.ceil(math.log2(min(t_len, 4096)))))
  )
  n_fft = 1 << int(math.ceil(math.log2(t_len + ir_len)))
  bm_irs = _measure_carfac_bm_irs(
    cochlea_model, ir_len=ir_len, ohc_scale=ohc_scale
  )
  freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

  # Zero-phase Butterworth frequency masks for HF envelope extraction.
  lpf_env_mask = 1.0 / (1.0 + (freqs / _ENVELOPE_LPF_HZ) ** 4)
  hpf_tail_mask = ((freqs / _TAIL_LEAKAGE_HPF_HZ) ** 4) / (
    1.0 + (freqs / _TAIL_LEAKAGE_HPF_HZ) ** 4
  )

  rng = np.random.default_rng(config.seed)
  x_ears = np.zeros((t_len, n_ears), dtype=np.float64)

  for ear in range(n_ears):
    h_ear = np.fft.rfft(bm_irs[:, :, ear], n=n_fft, axis=0)
    lpf_inv = _ihc_lpf_inverse(cochlea_model, freqs, ear, fs)
    bm_f = np.zeros_like(h_ear)

    for ch in range(n_ch):
      cf = cfs[ch]
      if cf < config.min_warm_start_cf_hz:
        continue
      ch_nap = nap_c[:, ch, ear]
      f_lo = max(config.min_warm_start_cf_hz * 0.85, cf * 0.72)
      f_hi = min(0.47 * fs, max(f_lo + 40.0, cf * 1.38))
      ch_mask = ((freqs >= f_lo) & (freqs <= f_hi)).astype(np.float64)
      spec = np.fft.rfft(ch_nap, n=n_fft)
      if cf < config.hf_crossover_hz:
        bm_f[:, ch] = spec * lpf_inv * ch_mask
      else:
        # Remove low-frequency vowel tail leakage before envelope extraction.
        ch_hf = np.fft.irfft(spec * hpf_tail_mask, n=n_fft)[:t_len]
        env_spec = np.fft.rfft(np.maximum(ch_hf, 0.0), n=n_fft) * lpf_env_mask
        env = np.maximum(0.0, np.fft.irfft(env_spec, n=n_fft)[:t_len])
        carrier_bp = np.fft.irfft(
          np.fft.rfft(rng.standard_normal(t_len), n=n_fft) * ch_mask, n=n_fft
        )[:t_len]
        carrier_bp = carrier_bp / (np.std(carrier_bp) + 1e-12)
        bm_f[:, ch] = np.fft.rfft(env * carrier_bp, n=n_fft) * ch_mask

    num = np.sum(np.conj(h_ear) * bm_f, axis=1)
    den = np.sum(np.abs(h_ear) ** 2, axis=1)
    active_band = (freqs >= config.min_warm_start_cf_hz) & (
      freqs <= float(cfs[0])
    )
    med_den = float(np.median(den[active_band])) if np.any(active_band) else 1.0
    passband = (freqs >= config.min_warm_start_cf_hz * 0.85) & (
      freqs <= min(0.48 * fs, float(cfs[0]) * 1.15)
    )
    reg = np.where(passband, 0.02 * med_den, 1e6 * med_den)
    x_ears[:, ear] = np.fft.irfft(num / (den + reg), n=n_fft)[:t_len]

  norm_rms = float(np.sqrt(np.mean(x_ears**2)))
  if norm_rms < 1e-12:
    return np.zeros((t_len, n_ears), dtype=np.float32)

  x_unit = jnp.asarray(x_ears / norm_rms, dtype=jnp.float32)
  target_c_j = jnp.asarray(nap_c, dtype=jnp.float32)
  rest_j = jnp.asarray(rest_nap[None, :, :], dtype=jnp.float32)
  best_gain = _calibrate_scalar_gain(x_unit, target_c_j, rest_j, cochlea_model)
  x1 = x_ears * (best_gain / norm_rms)

  # Apply 1-shot nonlinear spectral tilt equalization via one forward pass.
  nap1 = (
    np.asarray(
      cochlea_model(jnp.asarray(x1, dtype=jnp.float32)), dtype=np.float64
    )
    - rest_nap[None, :, :]
  )
  order = np.argsort(cfs)
  f_w = np.fft.rfftfreq(t_len, d=1.0 / fs)
  x2 = np.zeros_like(x1)
  for ear in range(n_ears):
    rms_tgt = np.sqrt(np.mean(np.maximum(nap_c[:, :, ear], 0.0) ** 2, axis=0))
    rms_1 = np.sqrt(np.mean(np.maximum(nap1[:, :, ear], 0.0) ** 2, axis=0))
    ratio = np.clip((rms_tgt + 2e-3) / (rms_1 + 2e-3), 0.3, 3.5)
    eq_curve = (
      np.interp(f_w, cfs[order], ratio[order], left=1.0, right=ratio[order][-1])
      ** 0.8
    )
    eq_curve = np.where(f_w < config.min_warm_start_cf_hz * 0.8, 0.0, eq_curve)
    x2[:, ear] = np.fft.irfft(np.fft.rfft(x1[:, ear]) * eq_curve, n=t_len)

  x2_rms = float(np.sqrt(np.mean(x2**2)))
  if x2_rms < 1e-12:
    return np.zeros((t_len, n_ears), dtype=np.float32)

  x2_unit = jnp.asarray(x2 / x2_rms, dtype=jnp.float32)
  refined_gain = _calibrate_scalar_gain(
    x2_unit, target_c_j, rest_j, cochlea_model, center_rms=x2_rms
  )
  return np.asarray(x2 * (refined_gain / x2_rms), dtype=np.float32)


def _generic_dsp_warm_start(
  nap_3d: np.ndarray,
  rest_nap: np.ndarray,
  cochlea_model: cochlea.CochleaModel,
  config: InversionConfig,
) -> np.ndarray:
  """Fallback warm start for non-CARFAC ``CochleaModel`` implementations."""
  t_len, n_ch, n_ears = nap_3d.shape
  fs = float(cochlea_model.sample_rate)
  nap_c = nap_3d.astype(np.float64) - rest_nap[None, :, :]
  rng = np.random.default_rng(config.seed)

  centre_freq = getattr(cochlea_model, "centre_freq", None)
  if centre_freq is not None and len(centre_freq) == n_ch:
    cfs = np.asarray(centre_freq, dtype=np.float64)
  else:
    f_lo_grid = max(config.min_warm_start_cf_hz, 80.0)
    f_hi_grid = min(0.45 * fs, 8000.0)
    cfs = np.geomspace(f_hi_grid, f_lo_grid, n_ch)

  n_fft = 1 << int(math.ceil(math.log2(t_len)))
  freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
  lpf_env_mask = 1.0 / (1.0 + (freqs / _ENVELOPE_LPF_HZ) ** 4)

  x_ears = np.zeros((t_len, n_ears), dtype=np.float64)
  for ear in range(n_ears):
    for ch in range(n_ch):
      cf = cfs[ch]
      f_lo = max(60.0, cf * 0.75)
      f_hi = min(0.47 * fs, max(f_lo + 20.0, cf * 1.30))
      ch_mask = ((freqs >= f_lo) & (freqs <= f_hi)).astype(np.float64)
      env_spec = (
        np.fft.rfft(np.maximum(nap_c[:, ch, ear], 0.0), n=n_fft) * lpf_env_mask
      )
      env = np.maximum(0.0, np.fft.irfft(env_spec, n=n_fft)[:t_len])
      carrier = np.fft.irfft(
        np.fft.rfft(rng.standard_normal(t_len), n=n_fft) * ch_mask, n=n_fft
      )[:t_len]
      carrier = carrier / (np.std(carrier) + 1e-12)
      x_ears[:, ear] += env * carrier

  norm_rms = float(np.sqrt(np.mean(x_ears**2)))
  if norm_rms < 1e-12:
    return np.zeros((t_len, n_ears), dtype=np.float32)

  x_unit = jnp.asarray(x_ears / norm_rms, dtype=jnp.float32)
  target_c_j = jnp.asarray(nap_c, dtype=jnp.float32)
  rest_j = jnp.asarray(rest_nap[None, :, :], dtype=jnp.float32)
  best_gain = _calibrate_scalar_gain(x_unit, target_c_j, rest_j, cochlea_model)
  return np.asarray(x_unit * best_gain, dtype=np.float32)


def dsp_warm_start(
  target_nap: np.ndarray | jnp.ndarray,
  cochlea_model: cochlea.CochleaModel,
  config: InversionConfig | None = None,
) -> np.ndarray:
  """Computes a fast physics-based DSP initial waveform from a target NAP.

  Args:
    target_nap: Neural activity pattern of shape ``(samples, channels)`` or
      ``(samples, channels, ears)``.
    cochlea_model: Target cochlea model used to obtain channel frequencies and
      resting baselines.
    config: Optional inversion settings; defaults to ``InversionConfig()``.

  Returns:
    Warm-start waveform of shape ``(samples,)`` for 2D input or
    ``(samples, ears)`` for 3D input.
  """
  cfg = InversionConfig() if config is None else config
  nap_3d, is_2d = _normalize_target_nap(target_nap, cochlea_model)
  t_len, _, n_ears = nap_3d.shape

  rest_len = min(512, t_len)
  rest_nap = np.asarray(
    cochlea_model(jnp.zeros((rest_len, n_ears), dtype=jnp.float32))[-1],
    dtype=np.float64,
  )
  if float(np.max(np.abs(nap_3d - rest_nap[None, :, :]))) < _SILENCE_NAP_PEAK:
    zeros = np.zeros((t_len, n_ears), dtype=np.float32)
    return zeros[:, 0] if is_2d else zeros

  if isinstance(cochlea_model, carfac.CarfacCochleaModel):
    wave = _carfac_dsp_warm_start(nap_3d, rest_nap, cochlea_model, cfg)
  else:
    wave = _generic_dsp_warm_start(nap_3d, rest_nap, cochlea_model, cfg)
  return wave[:, 0] if is_2d else wave


def _smooth_time_boxcar(x: jnp.ndarray, window_samples: int) -> jnp.ndarray:
  """Applies a symmetric moving average along axis 0 for ``(T, C, E)`` arrays."""
  if window_samples <= 1:
    return x
  pad_left = window_samples // 2
  pad_right = window_samples - 1 - pad_left
  padded = jnp.pad(x, ((pad_left, pad_right), (0, 0), (0, 0)), mode="edge")
  cumsum = jnp.cumsum(padded, axis=0)
  cumsum = jnp.pad(cumsum, ((1, 0), (0, 0), (0, 0)))
  return (cumsum[window_samples:] - cumsum[:-window_samples]) / float(
    window_samples
  )


def _audible_band_mask(
  num_samples: int,
  sample_rate_hz: float,
  cochlea_model: cochlea.CochleaModel,
  config: InversionConfig,
) -> jnp.ndarray:
  """Builds a raised-cosine rFFT projection mask over the audible CARFAC band."""
  freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate_hz)
  if config.cutoff_high_hz is not None:
    f_hi = min(0.49 * sample_rate_hz, config.cutoff_high_hz)
  elif isinstance(cochlea_model, carfac.CarfacCochleaModel):
    max_cf = float(jnp.max(cochlea_model.channel_frequencies))
    f_hi = min(0.48 * sample_rate_hz, max_cf * 1.15)
  else:
    f_hi = 0.48 * sample_rate_hz

  f_lo = min(config.cutoff_low_hz, 0.5 * f_hi)
  f_lo_end = min(f_lo + 35.0, 0.8 * f_hi)
  f_hi_start = max(f_lo_end, f_hi * 0.9)

  mask = np.ones_like(freqs, dtype=np.float32)
  if f_lo > 0.0:
    mask[freqs < f_lo] = 0.0
    if f_lo_end > f_lo:
      tr_lo = (freqs >= f_lo) & (freqs < f_lo_end)
      mask[tr_lo] = 0.5 * (
        1.0 - np.cos(math.pi * (freqs[tr_lo] - f_lo) / (f_lo_end - f_lo))
      )
  if f_hi > f_hi_start:
    tr_hi = (freqs > f_hi_start) & (freqs <= f_hi)
    mask[tr_hi] = 0.5 * (
      1.0 + np.cos(math.pi * (freqs[tr_hi] - f_hi_start) / (f_hi - f_hi_start))
    )
  mask[freqs > f_hi] = 0.0
  return jnp.asarray(mask[:, np.newaxis], dtype=jnp.float32)


def invert_nap(
  target_nap: np.ndarray | jnp.ndarray,
  cochlea_model: cochlea.CochleaModel,
  config: InversionConfig | None = None,
  *,
  initial_waveform: np.ndarray | jnp.ndarray | None = None,
) -> NapInversionResult:
  """Inverts a neural activity pattern (NAP) back to an audio waveform.

  Uses a physics-based DSP warm start followed by scale-invariant Adam + L-BFGS
  optimization through the differentiable ``cochlea_model``.

  Args:
    target_nap: Target NAP of shape ``(samples, channels)`` or
      ``(samples, channels, ears)``.
    cochlea_model: Differentiable cochlea model to invert through. When
      performing cross-cochlea hearing-loss auralization (e.g. ``target_nap``
      came from an impaired cochlea and ``cochlea_model`` is healthy), pass
      ``InversionConfig(mode="perceptual")``.
    config: Optional ``InversionConfig`` controlling mode and optimization budget.
    initial_waveform: Optional warm-start waveform overriding
      ``dsp_warm_start``, shape ``(samples,)`` or ``(samples, ears)``.

  Returns:
    ``NapInversionResult`` holding the final waveform, warm start, forward
    reconstructed NAP, and optimization loss trace.
  """
  cfg = InversionConfig() if config is None else config
  nap_3d, is_2d = _normalize_target_nap(target_nap, cochlea_model)
  t_len, n_ch, n_ears = nap_3d.shape
  fs = float(cochlea_model.sample_rate)

  rest_len = min(512, t_len)
  rest_nap_j = jnp.asarray(
    cochlea_model(jnp.zeros((rest_len, n_ears), dtype=jnp.float32))[-1:],
    dtype=jnp.float32,
  )
  target_j = jnp.asarray(nap_3d, dtype=jnp.float32)

  if (
    initial_waveform is None
    and float(jnp.max(jnp.abs(target_j - rest_nap_j))) < _SILENCE_NAP_PEAK
  ):
    zeros_w = np.zeros((t_len, n_ears), dtype=np.float32)
    recon_nap = np.asarray(
      cochlea_model(jnp.asarray(zeros_w, dtype=jnp.float32)), dtype=np.float32
    )
    return NapInversionResult(
      waveform=zeros_w[:, 0] if is_2d else zeros_w,
      warm_start=zeros_w[:, 0] if is_2d else zeros_w,
      reconstructed_nap=recon_nap[:, :, 0] if is_2d else recon_nap,
      loss_history=(),
      final_loss=0.0,
      sample_rate_hz=fs,
    )

  if initial_waveform is not None:
    init_np = np.asarray(initial_waveform, dtype=np.float32)
    if init_np.ndim == 1:
      init_np = init_np[:, np.newaxis]
    if init_np.shape != (t_len, n_ears):
      raise ValueError(
        f"initial_waveform shape {init_np.shape} does not match expected "
        f"({t_len}, {n_ears})"
      )
    if not np.all(np.isfinite(init_np)):
      raise ValueError("initial_waveform must contain only finite values")
  elif cfg.use_warm_start:
    init_3d = dsp_warm_start(nap_3d, cochlea_model, cfg)
    init_np = np.asarray(init_3d, dtype=np.float32)
  else:
    init_np = np.zeros((t_len, n_ears), dtype=np.float32)

  band_mask_j = _audible_band_mask(t_len, fs, cochlea_model, cfg)

  def project_wave(wave_2d: jnp.ndarray) -> jnp.ndarray:
    spec = jnp.fft.rfft(wave_2d, axis=0) * band_mask_j
    return jnp.fft.irfft(spec, n=t_len, axis=0).astype(jnp.float32)

  nap_var = jnp.maximum(jnp.mean((target_j - rest_nap_j) ** 2), 1e-6)
  win_fast = max(3, int(round(_ENVELOPE_FAST_S * fs)) | 1)
  win_slow = max(5, int(round(_ENVELOPE_SLOW_S * fs)) | 1)
  target_pos = jnp.maximum(target_j - rest_nap_j, 0.0)
  target_env_f = _smooth_time_boxcar(target_pos, win_fast)
  target_env_s = _smooth_time_boxcar(target_pos, win_slow)
  env_var = jnp.maximum(jnp.mean(target_env_f**2), 1e-6)

  if isinstance(cochlea_model, carfac.CarfacCochleaModel):
    cfs = np.asarray(cochlea_model.channel_frequencies, dtype=np.float32)
    hf_mask_j = jnp.asarray((cfs >= 3500.0).astype(np.float32)[None, :, None])
  else:
    hf_mask_j = jnp.zeros((1, n_ch, 1), dtype=jnp.float32)
  ch_rms = jnp.sqrt(jnp.mean(target_j**2, axis=(0, 2), keepdims=True))
  ch_scale = jnp.maximum(ch_rms, 0.02)

  sai_fft_len = min(2048, 2 ** int(math.floor(math.log2(t_len))))
  use_sai = cfg.mode == "perceptual" and sai_fft_len >= 64
  if use_sai:
    sai_hypers = sai.SaiHypers(fft_len=sai_fft_len)
    target_sai = jax.lax.stop_gradient(
      sai.from_naps(jnp.transpose(target_j, (2, 0, 1)), sai_hypers)
    )
    sai_var = jnp.maximum(jnp.mean(target_sai**2), 1e-6)
  else:
    sai_hypers = None
    target_sai = None
    sai_var = jnp.float32(1.0)

  if jax.tree_util.treedef_is_leaf(jax.tree_util.tree_structure(cochlea_model)):
    fixed_cochlea = cochlea_model
  else:
    fixed_cochlea = jax.tree_util.tree_map(jax.lax.stop_gradient, cochlea_model)

  def loss_fn(wave_2d: jnp.ndarray) -> jnp.ndarray:
    projected = project_wave(wave_2d)
    pred = fixed_cochlea(projected)
    if cfg.mode == "exact":
      loss = jnp.mean((pred - target_j) ** 2) / nap_var
      if cfg.hf_env_weight > 0.0:
        pred_env = _smooth_time_boxcar(
          jnp.maximum(pred - rest_nap_j, 0.0), win_fast
        )
        hf_err = jnp.mean(
          hf_mask_j * ((pred_env - target_env_f) / ch_scale) ** 2
        )
        loss = loss + cfg.hf_env_weight * hf_err
      if cfg.cutoff_low_hz <= 0.0:
        loss = loss + waveform_losses.low_frequency_drift_penalty(
          projected, sample_rate=int(round(fs))
        )
      return loss

    pred_pos = jnp.maximum(pred - rest_nap_j, 0.0)
    pred_env_f = _smooth_time_boxcar(pred_pos, win_fast)
    pred_env_s = _smooth_time_boxcar(pred_pos, win_slow)
    env_loss = (
      jnp.mean((pred_env_f - target_env_f) ** 2)
      + jnp.mean((pred_env_s - target_env_s) ** 2)
    ) / (2.0 * env_var)
    nap_loss = jnp.mean((pred - target_j) ** 2) / nap_var
    if use_sai and sai_hypers is not None and target_sai is not None:
      pred_sai = sai.from_naps(jnp.transpose(pred, (2, 0, 1)), sai_hypers)
      sai_loss = jnp.mean((pred_sai - target_sai) ** 2) / sai_var
      return 0.65 * env_loss + 0.25 * sai_loss + 0.10 * nap_loss
    return 0.85 * env_loss + 0.15 * nap_loss

  current_w = project_wave(jnp.asarray(init_np, dtype=jnp.float32))
  best_w = current_w
  best_loss = float(jax.jit(loss_fn)(current_w))
  loss_history: list[float] = []

  if cfg.adam_steps > 0:
    rms0 = max(float(jnp.sqrt(jnp.mean(current_w**2))), 1e-5)
    schedule = optax.cosine_decay_schedule(
      init_value=cfg.adam_lr_scale * rms0,
      decay_steps=cfg.adam_steps,
      alpha=0.05,
    )
    adam_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    adam_state = adam_opt.init(current_w)
    adam_vg = jax.value_and_grad(loss_fn)

    @jax.jit
    def adam_step(
      w: jnp.ndarray, state: optax.OptState
    ) -> tuple[jnp.ndarray, optax.OptState, jnp.ndarray]:
      val, grads = adam_vg(w)
      updates, next_state = adam_opt.update(grads, state, w)
      return project_wave(optax.apply_updates(w, updates)), next_state, val

    for _ in range(cfg.adam_steps):
      current_w, adam_state, val = adam_step(current_w, adam_state)
      val_f = float(val)
      loss_history.append(val_f)
      if math.isfinite(val_f) and val_f < best_loss:
        best_loss = val_f
        best_w = current_w

  if cfg.lbfgs_steps > 0:
    # Normalize parameters to O(1) RMS so L-BFGS's identity Hessian matches scale.
    u_rms = float(jnp.sqrt(jnp.mean(best_w**2)))
    u_scale = 1e-2 if u_rms < 1e-8 else max(u_rms, 1e-5)
    u_curr = best_w / u_scale

    def loss_scaled(u: jnp.ndarray) -> jnp.ndarray:
      return loss_fn(u * u_scale)

    lbfgs_opt = optax.lbfgs(
      learning_rate=1.0, memory_size=cfg.lbfgs_memory_size
    )
    lbfgs_state = lbfgs_opt.init(u_curr)
    lbfgs_vg = jax.value_and_grad(loss_scaled)

    @jax.jit
    def lbfgs_step(
      u: jnp.ndarray, state: optax.OptState
    ) -> tuple[jnp.ndarray, optax.OptState, jnp.ndarray]:
      val, grads = lbfgs_vg(u)
      updates, next_state = lbfgs_opt.update(
        grads, state, u, value=val, grad=grads, value_fn=loss_scaled
      )
      return project_wave(optax.apply_updates(u, updates)), next_state, val

    for _ in range(cfg.lbfgs_steps):
      u_curr, lbfgs_state, val = lbfgs_step(u_curr, lbfgs_state)
      val_f = float(val)
      loss_history.append(val_f)
      if math.isfinite(val_f) and val_f < best_loss:
        best_loss = val_f
        best_w = u_curr * u_scale

  final_w_2d = project_wave(best_w)
  final_eval = float(jax.jit(loss_fn)(final_w_2d))
  if math.isfinite(final_eval):
    best_loss = min(best_loss, final_eval)

  recon_3d = np.asarray(fixed_cochlea(final_w_2d), dtype=np.float32)
  final_np = np.asarray(final_w_2d, dtype=np.float32)
  return NapInversionResult(
    waveform=final_np[:, 0] if is_2d else final_np,
    warm_start=init_np[:, 0] if is_2d else init_np,
    reconstructed_nap=recon_3d[:, :, 0] if is_2d else recon_3d,
    loss_history=tuple(loss_history),
    final_loss=best_loss,
    sample_rate_hz=fs,
  )
