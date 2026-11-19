"""Inverts CARFAC neural activity patterns (NAPs) back to audio waveforms.

Combines a physics-based basilar-membrane matched-filter warm start with
scale-invariant JAX gradient optimization (Adam + L-BFGS) through a
differentiable ``CochleaModel``. Supports both same-cochlea round-trip
reconstruction (``mode="exact"``) and cross-cochlea hearing-loss auralization
(``mode="perceptual"``), with fixed-window overlap-add chunking for long clips.
"""

import dataclasses
import functools
import logging
import math
from typing import Literal
from typing import cast

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

from carfac_inversion import neural

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
    perceptual_nap_weight: Weight on the direct sample-level NAP anchor in
      ``"perceptual"`` mode (the remainder weights envelopes and SAI).
    adam_lr_scale: Initial Adam step size as a fraction of the initial waveform
      RMS.
    lbfgs_memory_size: History size for L-BFGS inverse Hessian approximation.
    chunk_seconds: Fixed optimization window duration in seconds for
      overlap-add processing when input length exceeds one window, or ``None``
      to optimize the full clip in one graph.
    chunk_overlap_s: Cosine-squared crossfade overlap between adjacent windows
      in seconds when chunking is active.
    chunk_preroll_s: Frozen preceding-audio context in seconds prepended to
      each chunk so CARFAC traveling-wave and AGC states are warmed up before
      scoring the chunk window.
    neural_checkpoint: Optional path to a trained Phase 3 ``.npz``
      :class:`carfac_inversion.neural.NapResidualRefiner` checkpoint. When set,
      refines the DSP warm start in one feedforward pass before gradient
      optimization.
    seed: Random seed for high-frequency warm-start noise carriers.
  """

  mode: InversionMode = "exact"
  adam_steps: int = 45
  lbfgs_steps: int = 135
  use_warm_start: bool = True
  min_warm_start_cf_hz: float = 250.0
  hf_crossover_hz: float = 4200.0
  cutoff_low_hz: float = 35.0
  cutoff_high_hz: float | None = None
  hf_env_weight: float = 0.008
  perceptual_nap_weight: float = 0.30
  adam_lr_scale: float = 0.08
  lbfgs_memory_size: int = 25
  chunk_seconds: float | None = 1.5
  chunk_overlap_s: float = 0.128
  chunk_preroll_s: float = 0.128
  neural_checkpoint: str | None = None
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
    if not 0.0 <= self.perceptual_nap_weight <= 1.0:
      raise ValueError(
        "perceptual_nap_weight must be in [0, 1], got "
        f"{self.perceptual_nap_weight}"
      )
    if self.adam_lr_scale <= 0.0:
      raise ValueError(
        f"adam_lr_scale must be positive, got {self.adam_lr_scale}"
      )
    if self.lbfgs_memory_size < 1:
      raise ValueError(
        f"lbfgs_memory_size must be at least 1, got {self.lbfgs_memory_size}"
      )
    if self.chunk_seconds is not None and self.chunk_seconds <= 0.0:
      raise ValueError(
        f"chunk_seconds must be positive or None, got {self.chunk_seconds}"
      )
    if self.chunk_overlap_s < 0.0:
      raise ValueError(
        f"chunk_overlap_s must be non-negative, got {self.chunk_overlap_s}"
      )
    if self.chunk_preroll_s < 0.0:
      raise ValueError(
        f"chunk_preroll_s must be non-negative, got {self.chunk_preroll_s}"
      )
    if (
      self.chunk_seconds is not None
      and self.chunk_overlap_s >= self.chunk_seconds
    ):
      raise ValueError(
        "chunk_overlap_s must be strictly less than chunk_seconds, got "
        f"{self.chunk_overlap_s} >= {self.chunk_seconds}"
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
    loss_history: Objective value at each optimization step (concatenated across
      windows when chunking is active).
    final_loss: Objective value achieved on the final full-clip output.
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
  ohc_scale: float | np.ndarray = 1.0,
) -> np.ndarray:
  """Measures open-loop CARFAC basilar-membrane impulse responses."""
  num_ears = len(cochlea_model.hypers.ears)
  imp = jnp.zeros((ir_len, num_ears), dtype=jnp.float32)
  imp = imp.at[0, :].set(_IMPULSE_AMPLITUDE)
  w = conductive.apply(imp, cochlea_model.conductive_filter)
  w = middle_ear.apply(w, cochlea_model.middle_ear_filter)

  scale_arr = np.asarray(ohc_scale, dtype=np.float32)
  state_ears = []
  for ear_idx in range(num_ears):
    ear_hypers = cochlea_model.hypers.ears[ear_idx]
    ear_weights = cochlea_model.weights.ears[ear_idx]
    ear_state = cochlea_model.initial_state.ears[ear_idx]
    if scale_arr.ndim == 0:
      ear_scale: float | jnp.ndarray = float(scale_arr)
    elif scale_arr.ndim == 1:
      ear_scale = float(scale_arr[ear_idx])
    else:
      ear_scale = jnp.asarray(scale_arr[:, ear_idx], dtype=jnp.float32)
    undamping = ear_weights.car.ohc_health * ear_scale
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


class _StaticModelRef:
  """Identity-hashed wrapper allowing non-PyTree CochleaModels in static JIT."""

  def __init__(self, model: cochlea.CochleaModel) -> None:
    self.model = model

  def __hash__(self) -> int:
    return id(self.model)

  def __eq__(self, other: object) -> bool:
    return isinstance(other, _StaticModelRef) and self.model is other.model


def _split_cochlea(
  cochlea_model: cochlea.CochleaModel,
) -> tuple[cochlea.CochleaModel | None, _StaticModelRef | None]:
  """Splits a CochleaModel into dynamic PyTree or static non-PyTree form."""
  is_pytree = not jax.tree_util.treedef_is_leaf(
    jax.tree_util.tree_structure(cochlea_model)
  )
  if is_pytree:
    return (
      jax.tree_util.tree_map(jax.lax.stop_gradient, cochlea_model),
      None,
    )
  return None, _StaticModelRef(cochlea_model)


def _run_cochlea(
  wave: jnp.ndarray,
  dyn_cochlea: cochlea.CochleaModel | None,
  static_cochlea: _StaticModelRef | None,
) -> jnp.ndarray:
  """Evaluates either a dynamic PyTree or static non-PyTree CochleaModel."""
  if dyn_cochlea is not None:
    return dyn_cochlea(wave)
  assert static_cochlea is not None
  return static_cochlea.model(wave)


@functools.partial(jax.jit, static_argnames=("static_cochlea",))
def _best_gain_on_grid_jit(
  gains: jnp.ndarray,
  unit_waveform: jnp.ndarray,
  target_ch_rms: jnp.ndarray,
  rest_nap_j: jnp.ndarray,
  dyn_cochlea: cochlea.CochleaModel | None,
  static_cochlea: _StaticModelRef | None,
) -> jnp.ndarray:
  """Selects the gain minimizing per-channel positive NAP RMS mismatch."""

  def single_err(gain: jnp.ndarray) -> jnp.ndarray:
    pred_centered = (
      _run_cochlea(gain * unit_waveform, dyn_cochlea, static_cochlea)
      - rest_nap_j
    )
    pred_ch_rms = jnp.sqrt(
      jnp.mean(jnp.maximum(pred_centered, 0.0) ** 2, axis=(0, 2)) + 1e-8
    )
    return jnp.mean((pred_ch_rms - target_ch_rms) ** 2)

  errs = jax.vmap(single_err)(gains)
  return gains[jnp.argmin(errs)]


def _calibrate_scalar_gain(
  unit_waveform: jnp.ndarray,
  target_centered_j: jnp.ndarray,
  rest_nap_j: jnp.ndarray,
  cochlea_model: cochlea.CochleaModel,
  center_rms: float | None = None,
  reference_rms: float | None = None,
) -> float:
  """Finds the scalar gain minimizing per-channel positive NAP RMS mismatch."""
  target_ch_rms = jnp.sqrt(
    jnp.mean(jnp.maximum(target_centered_j, 0.0) ** 2, axis=(0, 2)) + 1e-8
  )
  dyn_cochlea, static_cochlea = _split_cochlea(cochlea_model)

  if center_rms is None:
    lo = 5e-5
    hi = 0.4
    # Extend the coarse bracket when linear deconvolution indicates a quiet
    # signal or a retuned high-gain/attenuated cochlea outside baseline bounds.
    if reference_rms is not None and reference_rms > 0.0:
      if reference_rms < 1e-3:
        lo = min(lo, max(1e-7, reference_rms * 0.02))
      if reference_rms > 0.05:
        hi = max(hi, min(2.0, reference_rms * 10.0))
    coarse = jnp.asarray(np.geomspace(lo, hi, 20), dtype=jnp.float32)
    best_gain = float(
      _best_gain_on_grid_jit(
        coarse,
        unit_waveform,
        target_ch_rms,
        rest_nap_j,
        dyn_cochlea,
        static_cochlea,
      )
    )
    fine = jnp.asarray(
      np.geomspace(best_gain * 0.65, best_gain * 1.55, 7), dtype=jnp.float32
    )
  else:
    fine = jnp.asarray(
      np.geomspace(max(center_rms * 0.5, 1e-8), center_rms * 2.0, 7),
      dtype=jnp.float32,
    )
  return float(
    _best_gain_on_grid_jit(
      fine,
      unit_waveform,
      target_ch_rms,
      rest_nap_j,
      dyn_cochlea,
      static_cochlea,
    )
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

  # Estimate per-ear AGC-damped proxy OHC scale from each ear's NAP drive.
  ear_nap_rms = np.sqrt(np.mean(nap_c**2, axis=(0, 1)))
  ohc_scale = np.minimum(
    1.0, 1.0 / (1.0 + 1.2 * np.maximum(0.0, ear_nap_rms - 0.3))
  )

  # Cap open-loop IR length by physical duration (~0.15 s) so high-Q / high-fs
  # models (e.g., 48 kHz) capture their longer impulse response tails.
  max_ir_len = min(
    8192, max(2048, 1 << int(math.ceil(math.log2(max(0.15 * fs, 1.0)))))
  )
  ir_len = min(
    max_ir_len,
    max(1024, 1 << int(math.ceil(math.log2(min(t_len, max_ir_len))))),
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
  x_mid = np.zeros((t_len, n_ears), dtype=np.float64)
  h_ears: list[np.ndarray] = []
  lpf_invs: list[np.ndarray] = []
  dens: list[np.ndarray] = []
  med_dens: list[float] = []

  # Pass 1: invert supra-apical channels (>= min_warm_start_cf_hz).
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
    x_mid[:, ear] = np.fft.irfft(num / (den + reg), n=n_fft)[:t_len]
    h_ears.append(h_ear)
    lpf_invs.append(lpf_inv)
    dens.append(den)
    med_dens.append(med_den)

  norm_rms = float(np.sqrt(np.mean(x_mid**2)))
  target_c_j = jnp.asarray(nap_c, dtype=jnp.float32)
  rest_j = jnp.asarray(rest_nap[None, :, :], dtype=jnp.float32)
  if norm_rms >= 1e-8:
    x_unit = jnp.asarray(x_mid / norm_rms, dtype=jnp.float32)
    best_gain = _calibrate_scalar_gain(
      x_unit,
      target_c_j,
      rest_j,
      cochlea_model,
      reference_rms=norm_rms,
    )
    x1 = x_mid * (best_gain / norm_rms)
    nap1 = (
      np.asarray(
        cochlea_model(jnp.asarray(x1, dtype=jnp.float32)), dtype=np.float64
      )
      - rest_nap[None, :, :]
    )
    mid_gain_scale = best_gain / norm_rms
  else:
    x1 = np.zeros((t_len, n_ears), dtype=np.float64)
    nap1 = np.zeros_like(nap_c)
    mid_gain_scale = 1.0

  # Pass 2: recover apical (< min_warm_start_cf_hz) drive from orthogonalized
  # closed-loop residual so quadratic ohc_nlf combination tones cancel even
  # when Pass 1 amplitude or phase differs slightly across sample rates.
  apical_floor = max(45.0, config.cutoff_low_hz)
  ap_chs = np.where(
    (cfs >= apical_floor) & (cfs < config.min_warm_start_cf_hz)
  )[0]
  x_comb = x1.copy()
  has_apical = [False] * n_ears
  added_any_apical = False

  if len(ap_chs) > 0:
    for ear in range(n_ears):
      tgt_ap = nap_c[:, ap_chs, ear]
      leak_ap = nap1[:, ap_chs, ear]
      dot_ap = np.sum(tgt_ap * leak_ap, axis=0)
      leak_pow = np.sum(leak_ap**2, axis=0) + 1e-12
      beta_ap = np.clip(dot_ap / leak_pow, 0.0, 1.5)
      orth_res = tgt_ap - beta_ap[None, :] * leak_ap

      rms_tgt_ap = float(np.sqrt(np.mean(tgt_ap**2))) + 1e-12
      rms_orth_ap = float(np.sqrt(np.mean(orth_res**2)))
      unexplained_frac = (
        1.0
        if norm_rms < 1e-8
        else float(np.clip((rms_orth_ap / rms_tgt_ap - 0.65) / 0.25, 0.0, 1.0))
      )
      if unexplained_frac <= 0.0:
        continue
      h_ear = h_ears[ear]
      lpf_inv = lpf_invs[ear]
      den = dens[ear]
      med_den = med_dens[ear]
      bm_ap = np.zeros_like(h_ear)
      for idx, ch in enumerate(ap_chs):
        cf = cfs[ch]
        f_lo = max(apical_floor * 0.85, cf * 0.72)
        f_hi = min(
          config.min_warm_start_cf_hz * 1.15, max(f_lo + 25.0, cf * 1.38)
        )
        ch_mask = ((freqs >= f_lo) & (freqs <= f_hi)).astype(np.float64)
        ch_tgt_r = float(np.sqrt(np.mean(tgt_ap[:, idx] ** 2))) + 1e-9
        ch_orth_r = float(np.sqrt(np.mean(orth_res[:, idx] ** 2)))
        ch_gate = (
          1.0
          if norm_rms < 1e-8
          else float(np.clip((ch_orth_r / ch_tgt_r - 0.55) / 0.35, 0.0, 1.0))
        )
        bm_ap[:, ch] = (
          np.fft.rfft(orth_res[:, idx], n=n_fft) * lpf_inv * ch_mask * ch_gate
        )
      num_ap = np.sum(np.conj(h_ear) * bm_ap, axis=1)
      reg_ap = np.maximum(0.04 * den, 0.001 * med_den)
      pass_ap = (freqs >= apical_floor * 0.85) & (
        freqs <= config.min_warm_start_cf_hz * 1.2
      )
      x_ap = np.fft.irfft(
        np.where(pass_ap, num_ap / (den + reg_ap), 0.0), n=n_fft
      )[:t_len]
      x_comb[:, ear] += unexplained_frac * x_ap * mid_gain_scale
      has_apical[ear] = unexplained_frac >= 0.3
      added_any_apical = True

  comb_rms = float(np.sqrt(np.mean(x_comb**2)))
  if comb_rms < 1e-12:
    return np.zeros((t_len, n_ears), dtype=np.float32)

  if norm_rms < 1e-8:
    comb_gain = _calibrate_scalar_gain(
      jnp.asarray(x_comb / comb_rms, dtype=jnp.float32),
      target_c_j,
      rest_j,
      cochlea_model,
      reference_rms=comb_rms,
    )
    x_comb = x_comb * (comb_gain / comb_rms)

  # Apply 1-shot nonlinear spectral tilt equalization via closed-loop CARFAC.
  if added_any_apical or norm_rms < 1e-8:
    nap_eq = (
      np.asarray(
        cochlea_model(jnp.asarray(x_comb, dtype=jnp.float32)), dtype=np.float64
      )
      - rest_nap[None, :, :]
    )
  else:
    nap_eq = nap1
  order = np.argsort(cfs)
  f_w = np.fft.rfftfreq(t_len, d=1.0 / fs)
  x2 = np.zeros_like(x_comb)
  for ear in range(n_ears):
    rms_tgt = np.sqrt(np.mean(np.maximum(nap_c[:, :, ear], 0.0) ** 2, axis=0))
    rms_cur = np.sqrt(np.mean(np.maximum(nap_eq[:, :, ear], 0.0) ** 2, axis=0))
    ratio = np.clip((rms_tgt + 2e-3) / (rms_cur + 2e-3), 0.3, 3.5)
    eq_curve = (
      np.interp(f_w, cfs[order], ratio[order], left=1.0, right=ratio[order][-1])
      ** 0.8
    )
    low_cut = (
      apical_floor * 0.85
      if has_apical[ear]
      else config.min_warm_start_cf_hz * 0.8
    )
    eq_curve = np.where(f_w < low_cut, 0.0, eq_curve)
    x2[:, ear] = np.fft.irfft(np.fft.rfft(x_comb[:, ear]) * eq_curve, n=t_len)

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
  f_lo_end = min(f_lo + 18.0, 0.8 * f_hi)
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


def _project_wave(
  wave_2d: jnp.ndarray,
  band_mask_j: jnp.ndarray,
  num_samples: int,
) -> jnp.ndarray:
  """Projects a 2D waveform onto the audible frequency band via rFFT."""
  spec = jnp.fft.rfft(wave_2d, axis=0) * band_mask_j
  return jnp.fft.irfft(spec, n=num_samples, axis=0).astype(jnp.float32)


def _window_loss_impl(
  wave_2d: jnp.ndarray,
  dyn_cochlea: cochlea.CochleaModel | None,
  prefix_2d: jnp.ndarray,
  target_j: jnp.ndarray,
  rest_nap_j: jnp.ndarray,
  band_mask_j: jnp.ndarray,
  target_env_f: jnp.ndarray,
  target_env_s: jnp.ndarray,
  target_sai: jnp.ndarray,
  hf_mask_j: jnp.ndarray,
  ch_scale: jnp.ndarray,
  nap_var: jnp.ndarray,
  env_var: jnp.ndarray,
  sai_var: jnp.ndarray,
  hf_env_weight: jnp.ndarray,
  perceptual_nap_weight: jnp.ndarray,
  *,
  static_cochlea: _StaticModelRef | None,
  mode: InversionMode,
  use_hf_env: bool,
  apply_drift_loss: bool,
  use_sai: bool,
  sai_fft_len: int,
  win_fast: int,
  win_slow: int,
  fs_int: int,
  num_samples: int,
  preroll_samples: int,
) -> jnp.ndarray:
  """Evaluates the inversion objective for a single window (or full clip)."""
  projected = _project_wave(wave_2d, band_mask_j, num_samples)
  if preroll_samples > 0:
    full_wave = jnp.concatenate(
      [jax.lax.stop_gradient(prefix_2d), projected], axis=0
    )
    pred = _run_cochlea(full_wave, dyn_cochlea, static_cochlea)[
      preroll_samples:
    ]
  else:
    pred = _run_cochlea(projected, dyn_cochlea, static_cochlea)

  if mode == "exact":
    loss = jnp.mean((pred - target_j) ** 2) / nap_var
    if use_hf_env:
      pred_env = _smooth_time_boxcar(
        jnp.maximum(pred - rest_nap_j, 0.0), win_fast
      )
      hf_denom = (
        jnp.maximum(jnp.sum(hf_mask_j), 1.0)
        * float(pred.shape[0])
        * float(pred.shape[2])
      )
      hf_scale = jnp.maximum(ch_scale, 0.04)
      hf_err = (
        jnp.sum(hf_mask_j * ((pred_env - target_env_f) / hf_scale) ** 2)
        / hf_denom
      )
      loss = loss + hf_env_weight * hf_err
    if apply_drift_loss:
      loss = loss + waveform_losses.low_frequency_drift_penalty(
        projected, sample_rate=fs_int
      )
    return loss

  pred_pos = jnp.maximum(pred - rest_nap_j, 0.0)
  pred_env_f = _smooth_time_boxcar(pred_pos, win_fast)
  pred_env_s = _smooth_time_boxcar(pred_pos, win_slow)
  # Blend global envelope MSE with per-channel normalized MSE so 1-4 kHz
  # speech bands stay weighted under high-frequency hearing loss.
  global_env_loss = (
    jnp.mean((pred_env_f - target_env_f) ** 2)
    + jnp.mean((pred_env_s - target_env_s) ** 2)
  ) / (2.0 * env_var)
  ch_env_loss = 0.5 * (
    jnp.mean(((pred_env_f - target_env_f) / ch_scale) ** 2)
    + jnp.mean(((pred_env_s - target_env_s) / ch_scale) ** 2)
  )
  env_loss = 0.70 * global_env_loss + 0.30 * ch_env_loss
  nap_loss = jnp.mean((pred - target_j) ** 2) / nap_var
  if use_sai:
    sai_hypers = sai.SaiHypers(fft_len=sai_fft_len)
    pred_sai = sai.from_naps(jnp.transpose(pred, (2, 0, 1)), sai_hypers)
    sai_loss = jnp.mean((pred_sai - target_sai) ** 2) / sai_var
    remaining = jnp.maximum(0.0, 1.0 - perceptual_nap_weight)
    env_w = remaining * (0.65 / 0.90)
    sai_w = remaining * (0.25 / 0.90)
    return (
      env_w * env_loss + sai_w * sai_loss + perceptual_nap_weight * nap_loss
    )
  return (1.0 - perceptual_nap_weight) * env_loss + (
    perceptual_nap_weight * nap_loss
  )


_ADAM_TRANSFORM = optax.chain(
  optax.clip_by_global_norm(1.0), optax.scale_by_adam()
)

_STATIC_WINDOW_ARGS = (
  "static_cochlea",
  "mode",
  "use_hf_env",
  "apply_drift_loss",
  "use_sai",
  "sai_fft_len",
  "win_fast",
  "win_slow",
  "fs_int",
  "num_samples",
  "preroll_samples",
)


@functools.partial(jax.jit, static_argnames=_STATIC_WINDOW_ARGS)
def _eval_window_loss_jit(
  wave_2d: jnp.ndarray,
  dyn_cochlea: cochlea.CochleaModel | None,
  prefix_2d: jnp.ndarray,
  target_j: jnp.ndarray,
  rest_nap_j: jnp.ndarray,
  band_mask_j: jnp.ndarray,
  target_env_f: jnp.ndarray,
  target_env_s: jnp.ndarray,
  target_sai: jnp.ndarray,
  hf_mask_j: jnp.ndarray,
  ch_scale: jnp.ndarray,
  nap_var: jnp.ndarray,
  env_var: jnp.ndarray,
  sai_var: jnp.ndarray,
  hf_env_weight: jnp.ndarray,
  perceptual_nap_weight: jnp.ndarray,
  *,
  static_cochlea: _StaticModelRef | None,
  mode: InversionMode,
  use_hf_env: bool,
  apply_drift_loss: bool,
  use_sai: bool,
  sai_fft_len: int,
  win_fast: int,
  win_slow: int,
  fs_int: int,
  num_samples: int,
  preroll_samples: int,
) -> jnp.ndarray:
  """Module-level JIT loss evaluation reusing XLA across chunks and clips."""
  return _window_loss_impl(
    wave_2d,
    dyn_cochlea,
    prefix_2d,
    target_j,
    rest_nap_j,
    band_mask_j,
    target_env_f,
    target_env_s,
    target_sai,
    hf_mask_j,
    ch_scale,
    nap_var,
    env_var,
    sai_var,
    hf_env_weight,
    perceptual_nap_weight,
    static_cochlea=static_cochlea,
    mode=mode,
    use_hf_env=use_hf_env,
    apply_drift_loss=apply_drift_loss,
    use_sai=use_sai,
    sai_fft_len=sai_fft_len,
    win_fast=win_fast,
    win_slow=win_slow,
    fs_int=fs_int,
    num_samples=num_samples,
    preroll_samples=preroll_samples,
  )


@functools.partial(
  jax.jit, static_argnames=_STATIC_WINDOW_ARGS + ("adam_steps",)
)
def _adam_scan_jit(
  init_w: jnp.ndarray,
  adam_lr_scale: jnp.ndarray,
  dyn_cochlea: cochlea.CochleaModel | None,
  prefix_2d: jnp.ndarray,
  target_j: jnp.ndarray,
  rest_nap_j: jnp.ndarray,
  band_mask_j: jnp.ndarray,
  target_env_f: jnp.ndarray,
  target_env_s: jnp.ndarray,
  target_sai: jnp.ndarray,
  hf_mask_j: jnp.ndarray,
  ch_scale: jnp.ndarray,
  nap_var: jnp.ndarray,
  env_var: jnp.ndarray,
  sai_var: jnp.ndarray,
  hf_env_weight: jnp.ndarray,
  perceptual_nap_weight: jnp.ndarray,
  *,
  static_cochlea: _StaticModelRef | None,
  mode: InversionMode,
  use_hf_env: bool,
  apply_drift_loss: bool,
  use_sai: bool,
  sai_fft_len: int,
  win_fast: int,
  win_slow: int,
  fs_int: int,
  num_samples: int,
  preroll_samples: int,
  adam_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Runs the entire cosine-decayed Adam phase inside ``jax.lax.scan``."""

  def loss_w(w: jnp.ndarray) -> jnp.ndarray:
    return _window_loss_impl(
      w,
      dyn_cochlea,
      prefix_2d,
      target_j,
      rest_nap_j,
      band_mask_j,
      target_env_f,
      target_env_s,
      target_sai,
      hf_mask_j,
      ch_scale,
      nap_var,
      env_var,
      sai_var,
      hf_env_weight,
      perceptual_nap_weight,
      static_cochlea=static_cochlea,
      mode=mode,
      use_hf_env=use_hf_env,
      apply_drift_loss=apply_drift_loss,
      use_sai=use_sai,
      sai_fft_len=sai_fft_len,
      win_fast=win_fast,
      win_slow=win_slow,
      fs_int=fs_int,
      num_samples=num_samples,
      preroll_samples=preroll_samples,
    )

  w0 = _project_wave(init_w, band_mask_j, num_samples)
  raw_rms = jnp.sqrt(jnp.mean(w0**2))
  rms0 = jnp.where(
    raw_rms < 1e-9, jnp.float32(1e-5), jnp.maximum(raw_rms, 1e-7)
  )
  init_lr = adam_lr_scale * rms0
  opt_state0 = _ADAM_TRANSFORM.init(w0)
  best_l0 = loss_w(w0)
  vg_fn = jax.value_and_grad(loss_w)

  def scan_body(carry, step_idx):
    curr_w, opt_state, best_w, best_l = carry
    cosine = 0.5 * (
      1.0 + jnp.cos(math.pi * step_idx.astype(jnp.float32) / max(1, adam_steps))
    )
    step_lr = init_lr * (0.05 + 0.95 * cosine)
    val, grads = vg_fn(curr_w)
    updates, next_state = _ADAM_TRANSFORM.update(grads, opt_state, curr_w)
    upd_arr = cast(jnp.ndarray, updates)
    # Scale by -step_lr because _ADAM_TRANSFORM omits the learning-rate factor.
    next_w = _project_wave(curr_w - step_lr * upd_arr, band_mask_j, num_samples)
    better = jnp.isfinite(val) & (val < best_l)
    best_w = jnp.where(better, curr_w, best_w)
    best_l = jnp.where(better, val, best_l)
    return (next_w, next_state, best_w, best_l), val

  (final_w, _, best_w, best_l), losses = jax.lax.scan(
    scan_body,
    (w0, opt_state0, w0, best_l0),
    jnp.arange(adam_steps),
  )
  end_val = loss_w(final_w)
  end_better = jnp.isfinite(end_val) & (end_val < best_l)
  best_w = jnp.where(end_better, final_w, best_w)
  best_l = jnp.where(end_better, end_val, best_l)
  return best_w, best_l, losses


@functools.partial(
  jax.jit,
  static_argnames=_STATIC_WINDOW_ARGS + ("lbfgs_steps", "lbfgs_memory_size"),
)
def _lbfgs_scan_jit(
  init_w: jnp.ndarray,
  dyn_cochlea: cochlea.CochleaModel | None,
  prefix_2d: jnp.ndarray,
  target_j: jnp.ndarray,
  rest_nap_j: jnp.ndarray,
  band_mask_j: jnp.ndarray,
  target_env_f: jnp.ndarray,
  target_env_s: jnp.ndarray,
  target_sai: jnp.ndarray,
  hf_mask_j: jnp.ndarray,
  ch_scale: jnp.ndarray,
  nap_var: jnp.ndarray,
  env_var: jnp.ndarray,
  sai_var: jnp.ndarray,
  hf_env_weight: jnp.ndarray,
  perceptual_nap_weight: jnp.ndarray,
  *,
  static_cochlea: _StaticModelRef | None,
  mode: InversionMode,
  use_hf_env: bool,
  apply_drift_loss: bool,
  use_sai: bool,
  sai_fft_len: int,
  win_fast: int,
  win_slow: int,
  fs_int: int,
  num_samples: int,
  preroll_samples: int,
  lbfgs_steps: int,
  lbfgs_memory_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Runs the RMS-normalized L-BFGS phase inside ``jax.lax.scan``."""
  # Project before scaling so out-of-band energy does not skew u_scale.
  w_in = _project_wave(init_w, band_mask_j, num_samples)
  u_rms = jnp.sqrt(jnp.mean(w_in**2))
  u_scale = jnp.where(u_rms < 1e-9, jnp.float32(1e-2), jnp.maximum(u_rms, 1e-7))

  def loss_scaled(u: jnp.ndarray) -> jnp.ndarray:
    return _window_loss_impl(
      u * u_scale,
      dyn_cochlea,
      prefix_2d,
      target_j,
      rest_nap_j,
      band_mask_j,
      target_env_f,
      target_env_s,
      target_sai,
      hf_mask_j,
      ch_scale,
      nap_var,
      env_var,
      sai_var,
      hf_env_weight,
      perceptual_nap_weight,
      static_cochlea=static_cochlea,
      mode=mode,
      use_hf_env=use_hf_env,
      apply_drift_loss=apply_drift_loss,
      use_sai=use_sai,
      sai_fft_len=sai_fft_len,
      win_fast=win_fast,
      win_slow=win_slow,
      fs_int=fs_int,
      num_samples=num_samples,
      preroll_samples=preroll_samples,
    )

  u0 = w_in / u_scale
  lbfgs_opt = optax.lbfgs(learning_rate=1.0, memory_size=lbfgs_memory_size)
  state0 = lbfgs_opt.init(u0)
  best_l0 = loss_scaled(u0)
  vg_fn = jax.value_and_grad(loss_scaled)

  def scan_body(carry, _):
    curr_u, opt_state, best_u, best_l = carry
    val, grads = vg_fn(curr_u)
    updates, next_state = lbfgs_opt.update(
      grads, opt_state, curr_u, value=val, grad=grads, value_fn=loss_scaled
    )
    # Preserve exact L-BFGS secant updates; loss_scaled already projects inputs.
    next_u = cast(jnp.ndarray, optax.apply_updates(curr_u, updates))
    better = jnp.isfinite(val) & (val < best_l)
    best_u = jnp.where(better, curr_u, best_u)
    best_l = jnp.where(better, val, best_l)
    return (next_u, next_state, best_u, best_l), val

  (final_u, _, best_u, best_l), losses = jax.lax.scan(
    scan_body,
    (u0, state0, u0, best_l0),
    None,
    length=lbfgs_steps,
  )
  end_val = loss_scaled(final_u)
  end_better = jnp.isfinite(end_val) & (end_val < best_l)
  best_u = jnp.where(end_better, final_u, best_u)
  best_l = jnp.where(end_better, end_val, best_l)
  best_w = _project_wave(best_u * u_scale, band_mask_j, num_samples)
  return best_w, best_l, losses


def _optimize_window(
  init_2d: np.ndarray,
  target_j: jnp.ndarray,
  prefix_2d: jnp.ndarray,
  rest_nap_j: jnp.ndarray,
  band_mask_j: jnp.ndarray,
  hf_mask_j: jnp.ndarray,
  dyn_cochlea: cochlea.CochleaModel | None,
  static_cochlea: _StaticModelRef | None,
  config: InversionConfig,
  *,
  fs_int: int,
  win_fast: int,
  win_slow: int,
  sai_fft_len: int,
  use_sai: bool,
) -> tuple[jnp.ndarray, list[float]]:
  """Optimizes one waveform window (or full clip when unchunked)."""
  num_samples = int(target_j.shape[0])
  nap_var = jnp.maximum(jnp.mean((target_j - rest_nap_j) ** 2), 1e-6)
  target_pos = jnp.maximum(target_j - rest_nap_j, 0.0)
  target_env_f = _smooth_time_boxcar(target_pos, win_fast)
  target_env_s = _smooth_time_boxcar(target_pos, win_slow)
  env_var = jnp.maximum(jnp.mean(target_env_f**2), 1e-6)

  ch_rms = jnp.sqrt(jnp.mean(target_j**2, axis=(0, 2), keepdims=True))
  ch_scale = jnp.maximum(ch_rms, 0.02)

  if use_sai:
    sai_hypers = sai.SaiHypers(fft_len=sai_fft_len)
    target_sai = jax.lax.stop_gradient(
      sai.from_naps(jnp.transpose(target_j, (2, 0, 1)), sai_hypers)
    )
    sai_var = jnp.maximum(jnp.mean(target_sai**2), 1e-6)
  else:
    target_sai = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    sai_var = jnp.float32(1.0)

  hf_env_w_j = jnp.float32(config.hf_env_weight)
  perc_nap_w_j = jnp.float32(config.perceptual_nap_weight)
  use_hf_env = config.hf_env_weight > 0.0
  apply_drift_loss = config.cutoff_low_hz <= 0.0
  preroll_samples = int(prefix_2d.shape[0])

  best_w = _project_wave(
    jnp.asarray(init_2d, dtype=jnp.float32), band_mask_j, num_samples
  )
  loss_history: list[float] = []

  common_args = (
    dyn_cochlea,
    prefix_2d,
    target_j,
    rest_nap_j,
    band_mask_j,
    target_env_f,
    target_env_s,
    target_sai,
    hf_mask_j,
    ch_scale,
    nap_var,
    env_var,
    sai_var,
    hf_env_w_j,
    perc_nap_w_j,
  )

  if config.adam_steps > 0:
    best_w, _, adam_losses = _adam_scan_jit(
      best_w,
      jnp.float32(config.adam_lr_scale),
      *common_args,
      static_cochlea=static_cochlea,
      mode=config.mode,
      use_hf_env=use_hf_env,
      apply_drift_loss=apply_drift_loss,
      use_sai=use_sai,
      sai_fft_len=sai_fft_len,
      win_fast=win_fast,
      win_slow=win_slow,
      fs_int=fs_int,
      num_samples=num_samples,
      preroll_samples=preroll_samples,
      adam_steps=config.adam_steps,
    )
    loss_history.extend(
      float(x) for x in np.asarray(adam_losses, dtype=np.float64)
    )

  if config.lbfgs_steps > 0:
    best_w, _, lbfgs_losses = _lbfgs_scan_jit(
      best_w,
      *common_args,
      static_cochlea=static_cochlea,
      mode=config.mode,
      use_hf_env=use_hf_env,
      apply_drift_loss=apply_drift_loss,
      use_sai=use_sai,
      sai_fft_len=sai_fft_len,
      win_fast=win_fast,
      win_slow=win_slow,
      fs_int=fs_int,
      num_samples=num_samples,
      preroll_samples=preroll_samples,
      lbfgs_steps=config.lbfgs_steps,
      lbfgs_memory_size=config.lbfgs_memory_size,
    )
    loss_history.extend(
      float(x) for x in np.asarray(lbfgs_losses, dtype=np.float64)
    )

  return _project_wave(best_w, band_mask_j, num_samples), loss_history


def invert_nap(
  target_nap: np.ndarray | jnp.ndarray,
  cochlea_model: cochlea.CochleaModel,
  config: InversionConfig | None = None,
  *,
  initial_waveform: np.ndarray | jnp.ndarray | None = None,
) -> NapInversionResult:
  """Inverts a neural activity pattern (NAP) back to an audio waveform.

  Uses a physics-based DSP warm start followed by scale-invariant Adam + L-BFGS
  optimization through the differentiable ``cochlea_model``. When input duration
  exceeds ``config.chunk_seconds``, runs fixed-window overlap-add optimization
  with frozen preceding-audio pre-roll so XLA compiles once and runtime scales
  linearly with duration.

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
  fs_int = int(round(fs))

  rest_len = min(512, t_len)
  rest_nap_j = jnp.asarray(
    cochlea_model(jnp.zeros((rest_len, n_ears), dtype=jnp.float32))[-1:],
    dtype=jnp.float32,
  )
  rest_nap_np = np.asarray(rest_nap_j[0], dtype=np.float32)
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

  if initial_waveform is None and cfg.neural_checkpoint is not None:
    init_np = neural.refine_waveform(
      nap_3d,
      init_np,
      cochlea_model,
      cfg.neural_checkpoint,
      chunk_seconds=cfg.chunk_seconds,
      chunk_overlap_s=cfg.chunk_overlap_s,
      chunk_preroll_s=cfg.chunk_preroll_s,
      expected_mode=cfg.mode,
    )

  dyn_cochlea, static_cochlea = _split_cochlea(cochlea_model)

  if isinstance(cochlea_model, carfac.CarfacCochleaModel):
    cfs = np.asarray(cochlea_model.channel_frequencies, dtype=np.float32)
    hf_mask_j = jnp.asarray((cfs >= 3500.0).astype(np.float32)[None, :, None])
  else:
    hf_mask_j = jnp.zeros((1, n_ch, 1), dtype=jnp.float32)

  win_fast = max(3, int(round(_ENVELOPE_FAST_S * fs)) | 1)
  win_slow = max(5, int(round(_ENVELOPE_SLOW_S * fs)) | 1)

  chunk_samples = (
    int(round(cfg.chunk_seconds * fs))
    if cfg.chunk_seconds is not None
    else t_len
  )
  use_ola = cfg.chunk_seconds is not None and t_len > chunk_samples

  if not use_ola:
    band_mask_j = _audible_band_mask(t_len, fs, cochlea_model, cfg)
    sai_fft_len = min(2048, 2 ** int(math.floor(math.log2(t_len))))
    use_sai = cfg.mode == "perceptual" and sai_fft_len >= 64
    empty_prefix = jnp.zeros((0, n_ears), dtype=jnp.float32)
    final_w_2d, loss_history = _optimize_window(
      init_np,
      target_j,
      empty_prefix,
      rest_nap_j,
      band_mask_j,
      hf_mask_j,
      dyn_cochlea,
      static_cochlea,
      cfg,
      fs_int=fs_int,
      win_fast=win_fast,
      win_slow=win_slow,
      sai_fft_len=sai_fft_len,
      use_sai=use_sai,
    )
  else:
    win_len = max(64, chunk_samples)
    fade_len = min(int(round(cfg.chunk_overlap_s * fs)), win_len // 2)
    hop = max(1, win_len - fade_len)
    pre_roll = min(int(round(cfg.chunk_preroll_s * fs)), hop)

    band_mask_j = _audible_band_mask(win_len, fs, cochlea_model, cfg)
    sai_fft_len = min(2048, 2 ** int(math.floor(math.log2(win_len))))
    use_sai = cfg.mode == "perceptual" and sai_fft_len >= 64

    num_chunks = max(1, math.ceil(max(0, t_len - win_len) / hop) + 1)
    padded_len = (num_chunks - 1) * hop + win_len

    target_pad = np.broadcast_to(
      rest_nap_np[None, :, :], (padded_len, n_ch, n_ears)
    ).copy()
    target_pad[:t_len] = nap_3d
    init_pad = np.zeros((padded_len, n_ears), dtype=np.float32)
    init_pad[:t_len] = init_np

    if fade_len > 0:
      ramp = (
        np.sin(0.5 * math.pi * np.arange(fade_len, dtype=np.float32) / fade_len)
        ** 2
      )[:, None]
    else:
      ramp = np.zeros((0, 1), dtype=np.float32)

    out_pad = np.zeros((padded_len, n_ears), dtype=np.float32)
    loss_history = []
    for chunk_idx in range(num_chunks):
      start = chunk_idx * hop
      end = start + win_len
      if chunk_idx == 0 or pre_roll == 0:
        prefix = jnp.zeros((pre_roll, n_ears), dtype=jnp.float32)
      else:
        prefix = jnp.asarray(
          out_pad[start - pre_roll : start], dtype=jnp.float32
        )

      target_win = jnp.asarray(target_pad[start:end], dtype=jnp.float32)
      init_win = init_pad[start:end]
      win_w, win_hist = _optimize_window(
        init_win,
        target_win,
        prefix,
        rest_nap_j,
        band_mask_j,
        hf_mask_j,
        dyn_cochlea,
        static_cochlea,
        cfg,
        fs_int=fs_int,
        win_fast=win_fast,
        win_slow=win_slow,
        sai_fft_len=sai_fft_len,
        use_sai=use_sai,
      )
      loss_history.extend(win_hist)
      win_np = np.asarray(win_w, dtype=np.float32)

      if fade_len > 0 and chunk_idx > 0:
        out_pad[start : start + fade_len] = (
          out_pad[start : start + fade_len] * (1.0 - ramp)
          + win_np[:fade_len] * ramp
        )
        out_pad[start + fade_len : end] = win_np[fade_len:]
      else:
        out_pad[start:end] = win_np

    full_mask_j = _audible_band_mask(t_len, fs, cochlea_model, cfg)
    final_w_2d = _project_wave(
      jnp.asarray(out_pad[:t_len], dtype=jnp.float32), full_mask_j, t_len
    )

  recon_3d = np.asarray(cochlea_model(final_w_2d), dtype=np.float32)
  # Compute full-clip final loss against the blended output.
  full_mask_j = _audible_band_mask(t_len, fs, cochlea_model, cfg)
  full_sai_len = min(2048, 2 ** int(math.floor(math.log2(t_len))))
  full_use_sai = cfg.mode == "perceptual" and full_sai_len >= 64
  full_nap_var = jnp.maximum(jnp.mean((target_j - rest_nap_j) ** 2), 1e-6)
  full_pos = jnp.maximum(target_j - rest_nap_j, 0.0)
  full_env_f = _smooth_time_boxcar(full_pos, win_fast)
  full_env_s = _smooth_time_boxcar(full_pos, win_slow)
  full_env_var = jnp.maximum(jnp.mean(full_env_f**2), 1e-6)
  full_ch_scale = jnp.maximum(
    jnp.sqrt(jnp.mean(target_j**2, axis=(0, 2), keepdims=True)), 0.02
  )
  if full_use_sai:
    full_sai_target = sai.from_naps(
      jnp.transpose(target_j, (2, 0, 1)), sai.SaiHypers(fft_len=full_sai_len)
    )
    full_sai_var = jnp.maximum(jnp.mean(full_sai_target**2), 1e-6)
  else:
    full_sai_target = jnp.zeros((1, 1, 1, 1), dtype=jnp.float32)
    full_sai_var = jnp.float32(1.0)

  final_loss = float(
    _eval_window_loss_jit(
      final_w_2d,
      dyn_cochlea,
      jnp.zeros((0, n_ears), dtype=jnp.float32),
      target_j,
      rest_nap_j,
      full_mask_j,
      full_env_f,
      full_env_s,
      full_sai_target,
      hf_mask_j,
      full_ch_scale,
      full_nap_var,
      full_env_var,
      full_sai_var,
      jnp.float32(cfg.hf_env_weight),
      jnp.float32(cfg.perceptual_nap_weight),
      static_cochlea=static_cochlea,
      mode=cfg.mode,
      use_hf_env=cfg.hf_env_weight > 0.0,
      apply_drift_loss=cfg.cutoff_low_hz <= 0.0,
      use_sai=full_use_sai,
      sai_fft_len=full_sai_len,
      win_fast=win_fast,
      win_slow=win_slow,
      fs_int=fs_int,
      num_samples=t_len,
      preroll_samples=0,
    )
  )

  final_np = np.asarray(final_w_2d, dtype=np.float32)
  return NapInversionResult(
    waveform=final_np[:, 0] if is_2d else final_np,
    warm_start=init_np[:, 0] if is_2d else init_np,
    reconstructed_nap=recon_3d[:, :, 0] if is_2d else recon_3d,
    loss_history=tuple(loss_history),
    final_loss=final_loss,
    sample_rate_hz=fs,
  )
