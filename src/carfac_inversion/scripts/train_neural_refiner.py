"""Training script for the Phase 3 dual-domain CARFAC neural residual refiner.

Runs a two-stage training curriculum on cached ``(target_wav, ws_wav)`` pairs:
  1. Stage 1: Multi-resolution STFT, HF (3.8-7.8 kHz) spectral log-magnitude,
     scale-preserving L1/log-RMS, SI-SDR, and <20 Hz drift losses.
  2. Stage 2: Adds end-to-end differentiable CARFAC NAP + HF-envelope
     cycle-consistency fine-tuning so predictions align directly on the
     cochlear manifold.
"""

import dataclasses
import functools
import gc
import math
import pathlib
import time
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app
from absl import flags
from hp_acoustic import audio_constants
from hp_acoustic.losses import waveform_losses
from hp_acoustic.models import carfac
from hp_acoustic.models import sai

from carfac_inversion import inversion
from carfac_inversion import neural
from carfac_inversion import tpu

FLAGS = flags.FLAGS

_TRAIN_CACHE: flags.FlagHolder[str | None]
_VAL_CACHE: flags.FlagHolder[str | None]
_OUTPUT_CHECKPOINT: flags.FlagHolder[str | None]
_OBJECTIVE: flags.FlagHolder[str]
_INIT_CHECKPOINT: flags.FlagHolder[str | None]
_STAGE1_STEPS: flags.FlagHolder[int]
_STAGE2_STEPS: flags.FlagHolder[int]
_PERCEPTUAL_STEPS: flags.FlagHolder[int]
_BATCH_SIZE: flags.FlagHolder[int]
_STAGE2_BATCH_SIZE: flags.FlagHolder[int]
_PERCEPTUAL_BATCH_SIZE: flags.FlagHolder[int]
_STAGE1_LR: flags.FlagHolder[float]
_STAGE2_LR: flags.FlagHolder[float]
_PERCEPTUAL_LR: flags.FlagHolder[float]
_EVAL_EVERY: flags.FlagHolder[int]
_CARFAC_WEIGHT: flags.FlagHolder[float]
_PERCEPTUAL_NAP_WEIGHT: flags.FlagHolder[float]
_SAI_FFT_LEN: flags.FlagHolder[int]
_DRIFT_WEIGHT: flags.FlagHolder[float]
_HF_DIFF_WEIGHT: flags.FlagHolder[float]
_SEED: flags.FlagHolder[int]
_STEPS_PER_LOOP: flags.FlagHolder[int]
_COMPILATION_CACHE_DIR: flags.FlagHolder[str | None]
_MATMUL_PRECISION: flags.FlagHolder[str]

_MRSTFT_SPECS = (
  (2048, True, 0.25),
  (2048, False, 0.5),
  (1024, True, 0.25),
  (1024, False, 0.5),
  (512, True, 0.25),
  (512, False, 0.5),
  (128, True, 0.25),
  (128, False, 0.5),
)


@dataclasses.dataclass(frozen=True)
class TrainingResult:
  """Summary returned after Phase 3 training completes."""

  best_stage1_val_loss: float
  best_stage2_val_loss: float
  best_stage1_sisdr_db: float
  best_stage2_sisdr_db: float
  stage1_checkpoint_path: str | None
  final_checkpoint_path: str
  history: tuple[dict[str, float], ...]
  best_val_nap_corr: float = float("nan")


def define_flags(flag_values: flags.FlagValues = FLAGS) -> None:
  """Registers training CLI flags on ``flag_values``."""
  global _TRAIN_CACHE
  global _VAL_CACHE
  global _OUTPUT_CHECKPOINT
  global _STAGE1_STEPS
  global _STAGE2_STEPS
  global _BATCH_SIZE
  global _STAGE2_BATCH_SIZE
  global _STAGE1_LR
  global _STAGE2_LR
  global _EVAL_EVERY
  global _CARFAC_WEIGHT
  global _SEED
  global _OBJECTIVE
  global _INIT_CHECKPOINT
  global _PERCEPTUAL_STEPS
  global _PERCEPTUAL_BATCH_SIZE
  global _PERCEPTUAL_LR
  global _PERCEPTUAL_NAP_WEIGHT
  global _SAI_FFT_LEN
  global _DRIFT_WEIGHT
  global _HF_DIFF_WEIGHT
  global _STEPS_PER_LOOP
  global _COMPILATION_CACHE_DIR
  global _MATMUL_PRECISION

  _TRAIN_CACHE = flags.DEFINE_string(
    "train_cache",
    None,
    "Path to .npz training cache with target_wavs and ws_wavs.",
    required=True,
    flag_values=flag_values,
  )
  _VAL_CACHE = flags.DEFINE_string(
    "val_cache",
    None,
    "Path to .npz validation cache with target_wavs and ws_wavs.",
    required=True,
    flag_values=flag_values,
  )
  _OUTPUT_CHECKPOINT = flags.DEFINE_string(
    "output_checkpoint",
    None,
    "Destination .npz path for the trained neural refiner checkpoint.",
    required=True,
    flag_values=flag_values,
  )
  _STAGE1_STEPS = flags.DEFINE_integer(
    "stage1_steps",
    1400,
    "Number of Stage 1 spectral + waveform optimization steps.",
    flag_values=flag_values,
  )
  _STAGE2_STEPS = flags.DEFINE_integer(
    "stage2_steps",
    350,
    "Number of Stage 2 differentiable CARFAC cycle-consistency steps.",
    flag_values=flag_values,
  )
  _BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    8,
    "Mini-batch size for Stage 1 training.",
    flag_values=flag_values,
  )
  _STAGE2_BATCH_SIZE = flags.DEFINE_integer(
    "stage2_batch_size",
    4,
    "Mini-batch size for Stage 2 CARFAC fine-tuning.",
    flag_values=flag_values,
  )
  _STAGE1_LR = flags.DEFINE_float(
    "stage1_lr",
    4e-4,
    "Peak learning rate for Stage 1 cosine schedule.",
    flag_values=flag_values,
  )
  _STAGE2_LR = flags.DEFINE_float(
    "stage2_lr",
    1.5e-4,
    "Peak learning rate for Stage 2 cosine schedule.",
    flag_values=flag_values,
  )
  _EVAL_EVERY = flags.DEFINE_integer(
    "eval_every",
    150,
    "Validation cadence in training steps.",
    flag_values=flag_values,
  )
  _CARFAC_WEIGHT = flags.DEFINE_float(
    "carfac_weight",
    15.0,
    "Weight applied to variance-normalized CARFAC NAP cycle loss in Stage 2.",
    flag_values=flag_values,
  )
  _SEED = flags.DEFINE_integer(
    "seed",
    0,
    "PRNG seed for initialization and batch sampling.",
    flag_values=flag_values,
  )
  _OBJECTIVE = flags.DEFINE_enum(
    "objective",
    "exact",
    ("exact", "perceptual"),
    "Training objective: 'exact' (paired healthy) or 'perceptual' (self-supervised Phase 3b).",
    flag_values=flag_values,
  )
  _INIT_CHECKPOINT = flags.DEFINE_string(
    "init_checkpoint",
    None,
    "Optional checkpoint .npz used to initialize weights before perceptual fine-tuning.",
    flag_values=flag_values,
  )
  _PERCEPTUAL_STEPS = flags.DEFINE_integer(
    "perceptual_steps",
    1500,
    "Optimizer steps when --objective=perceptual.",
    flag_values=flag_values,
  )
  _PERCEPTUAL_BATCH_SIZE = flags.DEFINE_integer(
    "perceptual_batch_size",
    4,
    "Minibatch size when --objective=perceptual.",
    flag_values=flag_values,
  )
  _PERCEPTUAL_LR = flags.DEFINE_float(
    "perceptual_lr",
    2e-4,
    "Peak learning rate when --objective=perceptual.",
    flag_values=flag_values,
  )
  _PERCEPTUAL_NAP_WEIGHT = flags.DEFINE_float(
    "perceptual_nap_weight",
    0.30,
    "Weight on raw NAP MSE inside the Phase 3b perceptual cycle loss.",
    flag_values=flag_values,
  )
  _SAI_FFT_LEN = flags.DEFINE_integer(
    "sai_fft_len",
    2048,
    "SAI frame FFT length for --objective=perceptual.",
    flag_values=flag_values,
  )
  _DRIFT_WEIGHT = flags.DEFINE_float(
    "drift_weight",
    0.5,
    "Weight on sub-35 Hz infrasound drift penalty in --objective=perceptual.",
    flag_values=flag_values,
  )
  _HF_DIFF_WEIGHT = flags.DEFINE_float(
    "hf_diff_weight",
    0.0,
    "Optional high-frequency difference regularizer weight in --objective=perceptual.",
    flag_values=flag_values,
  )
  _STEPS_PER_LOOP = flags.DEFINE_integer(
    "steps_per_loop",
    1,
    "Optimizer steps fused per device jax.lax.scan call (K >= 1). Values > 1 "
    "eliminate host dispatch overhead on TPU.",
    flag_values=flag_values,
  )
  _COMPILATION_CACHE_DIR = flags.DEFINE_string(
    "compilation_cache_dir",
    None,
    "Optional directory or GCS URI for persistent XLA compilation cache.",
    flag_values=flag_values,
  )
  _MATMUL_PRECISION = flags.DEFINE_enum(
    "matmul_precision",
    "high",
    ("default", "high", "highest", "float32"),
    "JAX default matmul precision policy ('high' preserves TPU accuracy while "
    "keeping CARFAC cochlea states and losses in float32).",
    flag_values=flag_values,
  )


def _si_sdr_db(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
  """Computes per-example scale-invariant SDR in dB."""
  p = pred - jnp.mean(pred, axis=1, keepdims=True)
  t = target - jnp.mean(target, axis=1, keepdims=True)
  t_pow = jnp.sum(t**2, axis=1, keepdims=True) + 1e-12
  alpha = jnp.sum(p * t, axis=1, keepdims=True) / t_pow
  proj = alpha * t
  noise = p - proj
  num = jnp.sum(proj**2, axis=1) + 1e-8 * jnp.sum(t**2, axis=1) + 1e-12
  den = jnp.sum(noise**2, axis=1) + 1e-8 * jnp.sum(t**2, axis=1) + 1e-12
  return 10.0 * jnp.log10(num / den)


def _hf_log_mag_loss(
  target_norm: jnp.ndarray,
  pred_norm: jnp.ndarray,
  sample_rate_hz: int,
) -> jnp.ndarray:
  """Computes 3.8-7.8 kHz STFT log-magnitude MSE for HF naturalness."""
  n_fft = min(512, target_norm.shape[1])
  hop = max(n_fft // 4, 1)
  freqs, _, z_t = jax.scipy.signal.stft(
    target_norm,
    fs=sample_rate_hz,
    window="hann",
    nperseg=n_fft,
    noverlap=n_fft - hop,
    nfft=n_fft,
    boundary=None,
    padded=False,
    axis=1,
  )
  _, _, z_p = jax.scipy.signal.stft(
    pred_norm,
    fs=sample_rate_hz,
    window="hann",
    nperseg=n_fft,
    noverlap=n_fft - hop,
    nfft=n_fft,
    boundary=None,
    padded=False,
    axis=1,
  )
  hf_mask = ((freqs >= 3800.0) & (freqs <= 7800.0)).astype(jnp.float32)
  hf_mask = hf_mask[None, :, None]
  log_t = jnp.log(jnp.abs(z_t) + 1e-4)
  log_p = jnp.log(jnp.abs(z_p) + 1e-4)
  denom = jnp.maximum(jnp.sum(hf_mask), 1.0) * float(
    target_norm.shape[0] * z_t.shape[-1]
  )
  return jnp.sum(hf_mask * (log_t - log_p) ** 2) / denom


def compute_refiner_losses(
  pred: jnp.ndarray,
  target: jnp.ndarray,
  *,
  sample_rate_hz: int = audio_constants.SOURCE_SAMPLE_RATE,
  cochlea_model: carfac.CarfacCochleaModel | None = None,
  target_nap_raw: jnp.ndarray | None = None,
  rest_nap_raw: jnp.ndarray | None = None,
  hf_chan_mask: jnp.ndarray | None = None,
  carfac_weight: float = 0.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  """Computes normalized spectral, waveform, SPL-scale, and CARFAC losses."""
  tgt_rms = jnp.sqrt(jnp.mean(target**2, axis=1, keepdims=True) + 1e-16) + 1e-7
  pred_rms = jnp.sqrt(jnp.mean(pred**2, axis=1, keepdims=True) + 1e-16) + 1e-7
  tgt_n = target / tgt_rms
  pred_n = pred / tgt_rms

  stft_loss = waveform_losses.multiresolution_stft(
    tgt_n,
    pred_n,
    specs=_MRSTFT_SPECS,
    sample_rate=sample_rate_hz,
    eps=1e-5,
  )
  hf_loss = _hf_log_mag_loss(tgt_n, pred_n, sample_rate_hz)
  wave_l1 = jnp.mean(jnp.abs(pred_n - tgt_n))
  log_rms = jnp.mean((jnp.log(pred_rms) - jnp.log(tgt_rms)) ** 2)
  sisdr = jnp.mean(_si_sdr_db(pred, target))
  drift = waveform_losses.low_frequency_drift_penalty(
    pred_n[:, :, None], sample_rate=sample_rate_hz
  )

  total = (
    0.15 * stft_loss
    + 1.5 * hf_loss
    + 3.0 * wave_l1
    + 2.5 * log_rms
    - 0.35 * sisdr
    + 1.0 * drift
  )
  metrics = {
    "total": total,
    "stft": stft_loss,
    "hf_spec": hf_loss,
    "wave_l1": wave_l1,
    "log_rms": log_rms,
    "sisdr_db": sisdr,
    "drift": drift,
  }

  if (
    carfac_weight > 0.0
    and cochlea_model is not None
    and target_nap_raw is not None
    and rest_nap_raw is not None
  ):
    pred_nap = jax.vmap(lambda w: cochlea_model(w[:, None])[:, :, 0])(pred)
    tgt_c = target_nap_raw - rest_nap_raw[None, None, :]
    pred_c = pred_nap - rest_nap_raw[None, None, :]
    nap_var = jnp.mean(tgt_c**2, axis=(1, 2)) + 1e-6
    nap_mse = jnp.mean(jnp.mean((pred_c - tgt_c) ** 2, axis=(1, 2)) / nap_var)

    # Boxcar-smoothed (>3.5 kHz) HF positive envelope consistency.
    hf_err = jnp.float32(0.0)
    if hf_chan_mask is not None:
      tgt_pos = jnp.maximum(tgt_c, 0.0)
      pred_pos = jnp.maximum(pred_c, 0.0)
      ch_scale = jnp.maximum(
        jnp.sqrt(jnp.mean(tgt_pos**2, axis=1, keepdims=True)), 0.04
      )
      hf_denom = (
        jnp.maximum(jnp.sum(hf_chan_mask), 1.0)
        * float(pred.shape[0])
        * float(pred.shape[1])
      )
      hf_err = (
        jnp.sum(
          hf_chan_mask[None, None, :] * ((pred_pos - tgt_pos) / ch_scale) ** 2
        )
        / hf_denom
      )

    total = total + carfac_weight * (nap_mse + 0.15 * hf_err)
    metrics["carfac_nap"] = nap_mse
    metrics["carfac_hf"] = hf_err
    metrics["total"] = total

  return total, metrics


def _resolve_sai_fft_len(
  num_samples: int, requested_fft_len: int
) -> int | None:
  """Selects a valid SAI FFT window <= ``num_samples``, or ``None`` if too short."""
  fft_len = requested_fft_len
  while fft_len >= 512 and num_samples < fft_len:
    fft_len //= 2
  return fft_len if num_samples >= fft_len >= 512 else None


def compute_perceptual_refiner_losses(
  pred: jnp.ndarray,
  target_nap_raw: jnp.ndarray,
  *,
  cochlea_model: carfac.CarfacCochleaModel,
  rest_nap_raw: jnp.ndarray,
  sample_rate_hz: int = 16000,
  perceptual_nap_weight: float = 0.30,
  sai_fft_len: int = 2048,
  drift_weight: float = 0.5,
  hf_diff_weight: float = 0.0,
  ch_scale_rel_floor: float = 0.03,
  sample_weights: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  """Self-supervised differentiable CARFAC perceptual cycle loss (Phase 3b).

  Optimizes a healthy-cochlea forward synthesis ``pred`` directly against an
  arbitrary (possibly impaired) target NAP on matching multi-scale cochlear
  envelopes (2 ms & 8 ms), Stabilized Auditory Image (SAI) fine timing, and raw
  NAP, without requiring a ground-truth reference waveform.
  """
  pred_nap = jax.vmap(lambda w: cochlea_model(w[:, None])[:, :, 0])(pred)
  t_pos = jnp.maximum(target_nap_raw - rest_nap_raw[None, None, :], 0.0)
  p_pos = jnp.maximum(pred_nap - rest_nap_raw[None, None, :], 0.0)

  win_fast = max(1, int(round(0.002 * sample_rate_hz)))
  win_slow = max(1, int(round(0.008 * sample_rate_hz)))
  t_ef = jax.vmap(
    lambda x: inversion._smooth_time_boxcar(x[:, :, None], win_fast)[:, :, 0]
  )(t_pos)
  t_es = jax.vmap(
    lambda x: inversion._smooth_time_boxcar(x[:, :, None], win_slow)[:, :, 0]
  )(t_pos)
  p_ef = jax.vmap(
    lambda x: inversion._smooth_time_boxcar(x[:, :, None], win_fast)[:, :, 0]
  )(p_pos)
  p_es = jax.vmap(
    lambda x: inversion._smooth_time_boxcar(x[:, :, None], win_slow)[:, :, 0]
  )(p_pos)

  env_var = jnp.maximum(
    jnp.var(t_ef, axis=(1, 2)) + jnp.var(t_es, axis=(1, 2)), 1e-12
  )
  nap_var = jnp.maximum(jnp.var(target_nap_raw, axis=(1, 2)), 1e-12)
  raw_ch = jnp.sqrt(
    jnp.mean(t_ef**2, axis=1, keepdims=True)
    + jnp.mean(t_es**2, axis=1, keepdims=True)
  )
  # Floor weak high-CF channels relative to peak channel energy so silent or
  # synaptopathic (resting-shifted) HF channels do not dominate gradients.
  ch_floor = jnp.maximum(
    1e-4, ch_scale_rel_floor * jnp.max(raw_ch, axis=2, keepdims=True)
  )
  ch_scale = jnp.maximum(raw_ch, ch_floor)

  g_env = (
    jnp.mean((p_ef - t_ef) ** 2, axis=(1, 2))
    + jnp.mean((p_es - t_es) ** 2, axis=(1, 2))
  ) / (2.0 * env_var)
  c_env = 0.5 * (
    jnp.mean(((p_ef - t_ef) / ch_scale) ** 2, axis=(1, 2))
    + jnp.mean(((p_es - t_es) / ch_scale) ** 2, axis=(1, 2))
  )
  env_per_ex = 0.70 * g_env + 0.30 * c_env
  nap_per_ex = jnp.mean((pred_nap - target_nap_raw) ** 2, axis=(1, 2)) / nap_var

  eff_fft = _resolve_sai_fft_len(pred_nap.shape[1], sai_fft_len)
  if eff_fft is not None:
    sai_h = sai.SaiHypers(fft_len=eff_fft)
    t_sai = jax.vmap(lambda x: sai.from_naps(x[None, :, :], sai_h))(
      target_nap_raw
    )
    p_sai = jax.vmap(lambda x: sai.from_naps(x[None, :, :], sai_h))(pred_nap)
    sai_var = jnp.maximum(
      jnp.var(t_sai, axis=tuple(range(1, t_sai.ndim))), 1e-12
    )
    sai_per_ex = (
      jnp.mean((p_sai - t_sai) ** 2, axis=tuple(range(1, p_sai.ndim))) / sai_var
    )
    remaining = max(0.0, 1.0 - perceptual_nap_weight)
    env_w = remaining * (0.65 / 0.90)
    sai_w = remaining * (0.25 / 0.90)
    perc_per_ex = (
      env_w * env_per_ex
      + sai_w * sai_per_ex
      + perceptual_nap_weight * nap_per_ex
    )
  else:
    sai_per_ex = jnp.zeros_like(env_per_ex)
    remaining = max(0.0, 1.0 - perceptual_nap_weight)
    perc_per_ex = remaining * env_per_ex + perceptual_nap_weight * nap_per_ex

  n_t = pred.shape[1]
  freqs = jnp.fft.rfftfreq(n_t, d=1.0 / sample_rate_hz)
  spec_pow = jnp.abs(jnp.fft.rfft(pred, axis=1)) ** 2
  lf_mask = (freqs < 35.0).astype(jnp.float32)[None, :]
  drift_per_ex = jnp.sum(spec_pow * lf_mask, axis=1) / (
    jnp.sum(spec_pow, axis=1) + 1e-12
  )
  hf_diff_per_ex = jnp.mean((pred[:, 1:] - pred[:, :-1]) ** 2, axis=1) / (
    jnp.mean(pred**2, axis=1) + 1e-12
  )

  p_flat = pred_nap.reshape((pred_nap.shape[0], -1))
  t_flat = target_nap_raw.reshape((target_nap_raw.shape[0], -1))
  p_c = p_flat - jnp.mean(p_flat, axis=1, keepdims=True)
  t_c = t_flat - jnp.mean(t_flat, axis=1, keepdims=True)
  nap_corr_per_ex = jnp.sum(p_c * t_c, axis=1) / jnp.sqrt(
    jnp.maximum(jnp.sum(p_c**2, axis=1) * jnp.sum(t_c**2, axis=1), 1e-12)
  )

  if sample_weights is None:
    w = jnp.full((pred.shape[0],), 1.0 / pred.shape[0], dtype=jnp.float32)
  else:
    w = sample_weights / jnp.maximum(jnp.sum(sample_weights), 1e-8)

  def _wmean(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x * w)

  perc_l = _wmean(perc_per_ex)
  env_l = _wmean(env_per_ex)
  sai_l = _wmean(sai_per_ex)
  nap_l = _wmean(nap_per_ex)
  drift_l = _wmean(drift_per_ex)
  hf_diff_l = _wmean(hf_diff_per_ex)
  nap_corr_m = _wmean(nap_corr_per_ex)
  total = perc_l + drift_weight * drift_l + hf_diff_weight * hf_diff_l
  metrics = {
    "total": total,
    "perc": perc_l,
    "env": env_l,
    "sai": sai_l,
    "nap": nap_l,
    "drift": drift_l,
    "hf_diff": hf_diff_l,
    "nap_corr": nap_corr_m,
  }
  return total, metrics


def _batched_cochlea_run(
  wavs: np.ndarray,
  cochlea_model: carfac.CarfacCochleaModel,
  batch_size: int = 16,
) -> np.ndarray:
  """Runs healthy CARFAC in fixed-size JIT chunks to avoid shape recompilation."""
  jit_fn = jax.jit(jax.vmap(lambda x: cochlea_model(x[:, None])[:, :, 0]))
  n_ex, t_len = wavs.shape
  out = np.zeros((n_ex, t_len, cochlea_model.num_channels), dtype=np.float32)
  for s in range(0, n_ex, batch_size):
    e = min(n_ex, s + batch_size)
    chunk = wavs[s:e]
    if len(chunk) < batch_size:
      padded = np.pad(chunk, ((0, batch_size - len(chunk)), (0, 0)))
      res = np.asarray(jit_fn(jnp.asarray(padded)), dtype=np.float32)[
        : len(chunk)
      ]
    else:
      res = np.asarray(jit_fn(jnp.asarray(chunk)), dtype=np.float32)
    out[s:e] = res
  return out


def _project_naps_canon(
  raw_naps: np.ndarray,
  proj_w: np.ndarray,
  rest_nap: np.ndarray,
  batch_size: int = 16,
) -> np.ndarray:
  """Projects raw cochlear NAPs to canonical log-compressed CF bins."""
  proj_j = jnp.asarray(proj_w, dtype=jnp.float32)
  rest_j = jnp.asarray(rest_nap, dtype=jnp.float32)
  jit_proj = jax.jit(
    lambda n: neural._compress_nap(
      jnp.maximum(n - rest_j[None, None, :], 0.0) @ proj_j
    )
  )
  n_ex, t_len, _ = raw_naps.shape
  n_bins = proj_w.shape[1]
  out = np.zeros((n_ex, t_len, n_bins), dtype=np.float32)
  for s in range(0, n_ex, batch_size):
    e = min(n_ex, s + batch_size)
    out[s:e] = np.asarray(
      jit_proj(jnp.asarray(raw_naps[s:e])), dtype=np.float32
    )
  return out


def _load_perceptual_cache(
  cache_path: str,
  cochlea_model: carfac.CarfacCochleaModel,
  proj_w: np.ndarray,
  rest_nap: np.ndarray,
  *,
  batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Loads `(ws_wavs, canon_tgt_naps, canon_ws_naps, raw_tgt_naps)` from cache."""
  with np.load(cache_path) as d:
    ws_w = np.asarray(d["ws_wavs"], dtype=np.float32)
    if "target_naps" in d:
      raw_tgt = np.asarray(d["target_naps"], dtype=np.float32)
      if raw_tgt.shape != (
        ws_w.shape[0],
        ws_w.shape[1],
        cochlea_model.num_channels,
      ):
        raise ValueError(
          f"Cache {cache_path} target_naps shape {raw_tgt.shape} does not match "
          f"ws_wavs {ws_w.shape} x {cochlea_model.num_channels} channels."
        )
    elif "target_wavs" in d:
      tgt_w = np.asarray(d["target_wavs"], dtype=np.float32)
      raw_tgt = _batched_cochlea_run(
        tgt_w, cochlea_model, batch_size=batch_size
      )
    else:
      raise KeyError(
        f"Cache {cache_path} must contain 'target_naps' or 'target_wavs'."
      )

  raw_ws = _batched_cochlea_run(ws_w, cochlea_model, batch_size=batch_size)
  canon_tgt = _project_naps_canon(
    raw_tgt, proj_w, rest_nap, batch_size=batch_size
  )
  canon_ws = _project_naps_canon(
    raw_ws, proj_w, rest_nap, batch_size=batch_size
  )
  return ws_w, canon_tgt, canon_ws, raw_tgt


def _iter_sampled_batches(
  arrays: tuple[np.ndarray, ...],
  np_rng: np.random.Generator,
  batch_size: int,
  num_batches: int,
) -> Iterator[tuple[np.ndarray, ...]]:
  """Yields ``num_batches`` host mini-batches sampled uniformly from ``arrays``."""
  n_items = len(arrays[0])
  replace = n_items < batch_size
  for _ in range(num_batches):
    idx = np_rng.choice(n_items, size=batch_size, replace=replace)
    yield tuple(arr[idx] for arr in arrays)


def _extract_last_step_metrics(
  metrics_dev: Mapping[str, jnp.ndarray],
) -> dict[str, float]:
  """Copies ``[K]`` device metrics to host once and returns final-step scalars."""
  return {k: float(np.asarray(v)[-1]) for k, v in metrics_dev.items()}


def _validate_batch_size(
  name: str,
  batch_size: int,
  mesh: jax.sharding.Mesh,
) -> None:
  """Ensures training mini-batch size divides cleanly across local devices."""
  n_local = max(1, len(mesh.local_devices))
  if batch_size < 1 or batch_size % n_local != 0:
    raise ValueError(
      f"{name}={batch_size} must be a positive multiple of local device count "
      f"{n_local}."
    )


def _place_eval_batch(
  batch_tuple: tuple[np.ndarray, ...],
  mesh: jax.sharding.Mesh,
) -> tuple[jnp.ndarray, ...]:
  """Places an evaluation batch sharded across ``mesh`` (or replicated on tail)."""
  n_local = max(1, len(mesh.local_devices))
  if len(batch_tuple[0]) % n_local == 0:
    return cast(
      tuple[jnp.ndarray, ...],
      tpu.shard_batch(batch_tuple, mesh, stacked=False),
    )
  return tpu.replicate_tree(tuple(jnp.asarray(x) for x in batch_tuple), mesh)


def _run_training_stage(
  *,
  params: Mapping[str, Any],
  opt_state: optax.OptState,
  mesh: jax.sharding.Mesh,
  single_step_fn: Callable[
    [Mapping[str, Any], optax.OptState, Any],
    tuple[Mapping[str, Any], optax.OptState, dict[str, jnp.ndarray]],
  ],
  multi_step_fn: Callable[
    [Mapping[str, Any], optax.OptState, Any],
    tuple[Mapping[str, Any], optax.OptState, dict[str, jnp.ndarray]],
  ],
  train_arrays: tuple[np.ndarray, ...],
  np_rng: np.random.Generator,
  total_steps: int,
  batch_size: int,
  steps_per_loop: int,
  eval_every: int,
  eval_fn: Callable[[Mapping[str, Any]], dict[str, float]],
  best_loss_init: float,
  best_params_host_init: Mapping[str, Any],
  on_eval: Callable[[int, dict[str, float], dict[str, float]], None],
  on_best: Callable[[Mapping[str, Any], dict[str, float]], None] | None = None,
) -> tuple[Mapping[str, Any], optax.OptState, float, Mapping[str, Any]]:
  """Runs ``total_steps`` using fused ``jax.lax.scan`` loops and DMA prefetching.

  Donated ``params`` and ``opt_state`` buffers remain purely on device between
  evaluations; whenever validation improves, a detached host NumPy copy of
  ``params`` is snapshotted so buffer donation never aliases live references.
  """
  eval_cadence = max(1, eval_every)
  eff_k = (
    min(steps_per_loop, total_steps, eval_cadence) if steps_per_loop > 1 else 1
  )
  best_loss = best_loss_init
  best_params_host = best_params_host_init
  step = 0

  def _handle_eval(curr_step: int, metrics_dev: dict[str, jnp.ndarray]) -> None:
    nonlocal best_loss, best_params_host
    tr_m = _extract_last_step_metrics(metrics_dev)
    val_m = eval_fn(params)
    on_eval(curr_step, tr_m, val_m)
    if math.isfinite(val_m["total"]) and val_m["total"] < best_loss:
      best_loss = float(val_m["total"])
      best_params_host = tpu.to_host_tree(params)
      if on_best is not None:
        on_best(best_params_host, val_m)

  def _run_chunks(
    n_chunks: int,
    chunk_k: int,
    step_fn: Callable[
      [Mapping[str, Any], optax.OptState, Any],
      tuple[Mapping[str, Any], optax.OptState, dict[str, jnp.ndarray]],
    ],
    *,
    force_final_eval: bool,
  ) -> None:
    nonlocal params, opt_state, step
    if n_chunks <= 0:
      return
    with tpu.ThreadedDevicePrefetcher(
      _iter_sampled_batches(
        train_arrays, np_rng, batch_size, n_chunks * chunk_k
      ),
      mesh,
      steps_per_loop=chunk_k,
    ) as prefetcher:
      for chunk_idx, dev_chunk in enumerate(prefetcher, start=1):
        params, opt_state, m_dev = step_fn(params, opt_state, dev_chunk)
        step += chunk_k
        is_eval = (step % eval_cadence == 0) or (
          force_final_eval and step == total_steps
        )
        if is_eval:
          _handle_eval(step, m_dev)
        elif chunk_k > 1 and chunk_idx % 4 == 0:
          jax.tree_util.tree_leaves(m_dev)[0].block_until_ready()

  def _run_segment(n_steps: int, *, eval_at_end: bool) -> None:
    n_k = (n_steps // eff_k) if eff_k > 1 else 0
    n_rem = n_steps - n_k * eff_k
    _run_chunks(
      n_k, eff_k, multi_step_fn, force_final_eval=(eval_at_end and n_rem == 0)
    )
    _run_chunks(n_rem, 1, single_step_fn, force_final_eval=eval_at_end)

  if eff_k == 1 or (eval_cadence % eff_k == 0):
    _run_segment(total_steps, eval_at_end=True)
  else:
    milestones = list(range(eval_cadence, total_steps, eval_cadence))
    if not milestones or milestones[-1] != total_steps:
      milestones.append(total_steps)
    prev = 0
    for target_step in milestones:
      _run_segment(target_step - prev, eval_at_end=(target_step == total_steps))
      prev = target_step

  return params, opt_state, best_loss, best_params_host


def _evaluate_perceptual_split(
  eval_batch_fn: Callable[..., dict[str, jnp.ndarray]],
  mesh: jax.sharding.Mesh,
  ws_wavs: np.ndarray,
  canon_tgt_naps: np.ndarray,
  canon_ws_naps: np.ndarray,
  raw_tgt_naps: np.ndarray,
  params: Mapping[str, Any],
  *,
  batch_size: int = 4,
) -> dict[str, float]:
  """Evaluates perceptual validation metrics over fixed-shape padded batches."""
  n_ex = len(ws_wavs)
  sums: dict[str, float] = {}
  total_weight = 0.0
  for s in range(0, n_ex, batch_size):
    e = min(n_ex, s + batch_size)
    cnt = e - s
    weights = np.zeros((batch_size,), dtype=np.float32)
    weights[:cnt] = 1.0
    if cnt < batch_size:
      pad_idx = np.concatenate(
        [np.arange(s, e), np.full((batch_size - cnt,), s)]
      )
    else:
      pad_idx = np.arange(s, e)
    b_ws, b_tc, b_wc, b_rtgt, b_w = tpu.shard_batch(
      (
        ws_wavs[pad_idx],
        canon_tgt_naps[pad_idx],
        canon_ws_naps[pad_idx],
        raw_tgt_naps[pad_idx],
        weights,
      ),
      mesh,
      stacked=False,
    )
    metrics = eval_batch_fn(params, b_ws, b_tc, b_wc, b_rtgt, b_w)
    for k, v in metrics.items():
      sums[k] = sums.get(k, 0.0) + float(np.asarray(v)) * cnt
    total_weight += cnt
  return {k: v / max(1.0, total_weight) for k, v in sums.items()}


def _run_perceptual_training(
  *,
  train_cache: str,
  val_cache: str,
  output_checkpoint: str,
  init_checkpoint: str | None,
  steps: int,
  batch_size: int,
  lr: float,
  eval_every: int,
  perceptual_nap_weight: float,
  sai_fft_len: int,
  drift_weight: float,
  hf_diff_weight: float,
  seed: int,
  model_config: neural.NeuralRefinerConfig | None,
  steps_per_loop: int = 1,
  compilation_cache_dir: str | None = None,
  matmul_precision: str = "high",
) -> TrainingResult:
  """Trains a Phase 3b refiner end-to-end under differentiable CARFAC perceptual loss."""
  if steps_per_loop < 1:
    raise ValueError(f"steps_per_loop must be >= 1, got {steps_per_loop}.")
  t0 = time.perf_counter()
  backend, n_devices = tpu.configure_runtime(
    compilation_cache_dir=compilation_cache_dir,
    matmul_precision=matmul_precision,
  )
  mesh = tpu.make_data_mesh()
  _validate_batch_size("perceptual_batch_size", batch_size, mesh)

  init_params: Mapping[str, Any] | None = None
  if init_checkpoint is not None:
    _, init_params, init_cfg = neural.load_refiner_checkpoint(init_checkpoint)
    if model_config is not None:
      arch_keys = (
        "dim",
        "dilations",
        "n_coch_bins",
        "cf_min_hz",
        "cf_max_hz",
        "n_fft",
        "hop_length",
        "unet_base_dim",
        "sample_rate_hz",
      )
      for key in arch_keys:
        if getattr(init_cfg, key) != getattr(model_config, key):
          raise ValueError(
            f"init_checkpoint {key}={getattr(init_cfg, key)!r} does not match "
            f"model_config {key}={getattr(model_config, key)!r}."
          )
      base_cfg = model_config
    else:
      base_cfg = init_cfg
  else:
    base_cfg = (
      neural.NeuralRefinerConfig() if model_config is None else model_config
    )

  cfg = dataclasses.replace(base_cfg, target_mode="perceptual")
  model = neural.NapResidualRefiner(config=cfg)
  fs = int(round(cfg.sample_rate_hz))

  with np.load(train_cache) as d_tr:
    t_len = int(d_tr["ws_wavs"].shape[1])
  with tpu.host_cpu_context():
    cochlea_model = carfac.build(fs=fs)
    rest_nap = np.asarray(
      cochlea_model(jnp.zeros((min(512, t_len), 1), dtype=jnp.float32))[
        -1, :, 0
      ],
      dtype=np.float32,
    )
    proj_w = neural.cochlea_cf_projection(cochlea_model, cfg)
    cochlea_host = jax.tree_util.tree_map(
      jax.lax.stop_gradient, tpu.to_host_tree(cochlea_model)
    )
  rest_nap_j = tpu.replicate_tree(
    jnp.asarray(rest_nap, dtype=jnp.float32), mesh
  )
  rep_cochlea = tpu.replicate_tree(cochlea_host, mesh)

  print(
    f"Loading Phase 3b perceptual caches ({fs} Hz, T={t_len}, "
    f"backend={backend}, devices={n_devices}, K={steps_per_loop})...",
    flush=True,
  )
  tr_ws, tr_tgt_c, tr_ws_c, tr_raw_tgt = _load_perceptual_cache(
    train_cache, rep_cochlea, proj_w, rest_nap
  )
  va_ws, va_tgt_c, va_ws_c, va_raw_tgt = _load_perceptual_cache(
    val_cache, rep_cochlea, proj_w, rest_nap
  )
  n_train = len(tr_ws)
  print(
    f"Loaded {n_train} train, {len(va_ws)} val perceptual examples "
    f"in {time.perf_counter() - t0:.1f}s.",
    flush=True,
  )

  if init_params is None:
    with tpu.host_cpu_context():
      raw_params = model.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, t_len), dtype=jnp.float32),
        jnp.zeros((1, t_len, cfg.n_coch_bins), dtype=jnp.float32),
        jnp.zeros((1, t_len, cfg.n_coch_bins), dtype=jnp.float32),
      )
      init_host = tpu.to_host_tree(raw_params)
    params_host: Mapping[str, Any] = init_host
  else:
    params_host = tpu.to_host_tree(init_params)
  params = cast(Mapping[str, Any], tpu.replicate_tree(params_host, mesh))

  warmup = min(50, max(1, steps // 10))
  schedule = optax.warmup_cosine_decay_schedule(
    init_value=lr * 0.1,
    peak_value=lr,
    warmup_steps=warmup,
    decay_steps=max(steps, warmup + 1),
    end_value=lr * 0.05,
  )
  tx = optax.chain(
    optax.clip_by_global_norm(2.0),
    optax.adamw(learning_rate=schedule, weight_decay=1e-4),
  )
  with tpu.host_cpu_context():
    opt_host = tpu.to_host_tree(tx.init(params_host))
  opt_state = tpu.replicate_tree(opt_host, mesh)

  def _loss_on_batch(
    pr: Mapping[str, Any],
    b_ws: jnp.ndarray,
    b_tc: jnp.ndarray,
    b_wc: jnp.ndarray,
    b_rtgt: jnp.ndarray,
    sample_weights: jnp.ndarray | None = None,
  ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    pred = cast(jnp.ndarray, model.apply(pr, b_ws, b_tc, b_wc))
    return compute_perceptual_refiner_losses(
      pred,
      b_rtgt,
      cochlea_model=rep_cochlea,
      rest_nap_raw=rest_nap_j,
      sample_rate_hz=fs,
      perceptual_nap_weight=perceptual_nap_weight,
      sai_fft_len=sai_fft_len,
      drift_weight=drift_weight,
      hf_diff_weight=hf_diff_weight,
      sample_weights=sample_weights,
    )

  def step_impl(
    p: Mapping[str, Any],
    os_state: optax.OptState,
    batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
  ) -> tuple[Mapping[str, Any], optax.OptState, dict[str, jnp.ndarray]]:
    b_ws, b_tc, b_wc, b_rtgt = batch
    (_, metrics), grads = jax.value_and_grad(_loss_on_batch, has_aux=True)(
      p, b_ws, b_tc, b_wc, b_rtgt
    )
    updates, new_os = tx.update(grads, os_state, p)
    next_p = cast(Mapping[str, Any], optax.apply_updates(p, updates))
    return next_p, new_os, metrics

  single_step_fn, multi_step_fn = tpu.make_multi_step_fns(
    step_impl, donate_argnums=(0, 1)
  )

  @jax.jit
  def eval_batch(
    p: Mapping[str, Any],
    b_ws: jnp.ndarray,
    b_tc: jnp.ndarray,
    b_wc: jnp.ndarray,
    b_rtgt: jnp.ndarray,
    b_weights: jnp.ndarray,
  ) -> dict[str, jnp.ndarray]:
    _, metrics = _loss_on_batch(p, b_ws, b_tc, b_wc, b_rtgt, b_weights)
    return metrics

  def _eval_val(p: Mapping[str, Any]) -> dict[str, float]:
    return _evaluate_perceptual_split(
      eval_batch,
      mesh,
      va_ws,
      va_tgt_c,
      va_ws_c,
      va_raw_tgt,
      p,
      batch_size=batch_size,
    )

  np_rng = np.random.default_rng(seed)
  best_val_nap_corr = -1.0
  history: list[dict[str, float]] = []
  out_path = pathlib.Path(output_checkpoint)

  # Log step-0 baseline before any update, without overwriting best with unadapted exact weights.
  val0 = _eval_val(params)
  init_val_loss = float(val0["total"])
  print(
    f"[Phase3b] init val loss={val0['total']:.4f} "
    f"(perc={val0['perc']:.4f}, env={val0['env']:.4f}, sai={val0['sai']:.4f}, "
    f"nap={val0['nap']:.4f}, nap_corr={val0['nap_corr']:.4f}, drift={val0['drift']:.4f})",
    flush=True,
  )
  best_host = tpu.to_host_tree(params)
  if steps == 0:
    neural.save_refiner_checkpoint(out_path, best_host, cfg)
    tpu.delete_tree((params, opt_state))
    return TrainingResult(
      best_stage1_val_loss=init_val_loss,
      best_stage2_val_loss=init_val_loss,
      best_stage1_sisdr_db=0.0,
      best_stage2_sisdr_db=0.0,
      stage1_checkpoint_path=init_checkpoint,
      final_checkpoint_path=str(out_path),
      history=(),
      best_val_nap_corr=float(val0["nap_corr"]),
    )

  best_val_loss = float("inf")

  logged_best = float("inf")

  def _on_eval_and_log(
    step_idx: int,
    tr_m: dict[str, float],
    m_val: dict[str, float],
  ) -> None:
    nonlocal logged_best
    history.append(
      {
        "stage": 3.0,
        "step": float(step_idx),
        "train_loss": float(tr_m["total"]),
        "train_nap_corr": float(tr_m["nap_corr"]),
        "val_loss": float(m_val["total"]),
        "val_perc": float(m_val["perc"]),
        "val_env": float(m_val["env"]),
        "val_sai": float(m_val["sai"]),
        "val_nap": float(m_val["nap"]),
        "val_drift": float(m_val["drift"]),
        "val_nap_corr": float(m_val["nap_corr"]),
      }
    )
    if math.isfinite(m_val["total"]) and m_val["total"] < logged_best:
      logged_best = float(m_val["total"])
    print(
      f"[Phase3b] step {step_idx:4d}/{steps} | train={tr_m['total']:.4f} "
      f"| val={m_val['total']:.4f} (env={m_val['env']:.4f}, sai={m_val['sai']:.4f}, "
      f"nap={m_val['nap']:.4f}, drift={m_val['drift']:.4f}, nap_corr={m_val['nap_corr']:.4f}) "
      f"| best={logged_best:.4f} | t={time.perf_counter() - t0:.1f}s",
      flush=True,
    )

  def _on_best(
    best_p_host: Mapping[str, Any],
    m_val: dict[str, float],
  ) -> None:
    nonlocal best_val_nap_corr
    best_val_nap_corr = float(m_val["nap_corr"])
    neural.save_refiner_checkpoint(out_path, best_p_host, cfg)

  params, opt_state, best_val_loss, _ = _run_training_stage(
    params=params,
    opt_state=opt_state,
    mesh=mesh,
    single_step_fn=single_step_fn,
    multi_step_fn=multi_step_fn,
    train_arrays=(tr_ws, tr_tgt_c, tr_ws_c, tr_raw_tgt),
    np_rng=np_rng,
    total_steps=steps,
    batch_size=batch_size,
    steps_per_loop=steps_per_loop,
    eval_every=eval_every,
    eval_fn=_eval_val,
    best_loss_init=best_val_loss,
    best_params_host_init=best_host,
    on_eval=_on_eval_and_log,
    on_best=_on_best,
  )
  tpu.delete_tree((params, opt_state))

  return TrainingResult(
    best_stage1_val_loss=init_val_loss,
    best_stage2_val_loss=best_val_loss,
    best_stage1_sisdr_db=0.0,
    best_stage2_sisdr_db=0.0,
    stage1_checkpoint_path=init_checkpoint,
    final_checkpoint_path=str(out_path),
    history=tuple(history),
    best_val_nap_corr=best_val_nap_corr,
  )


def _precompute_cochlear_features(
  target_wavs: np.ndarray,
  ws_wavs: np.ndarray,
  cochlea_model: carfac.CarfacCochleaModel,
  proj_w: np.ndarray,
  rest_nap: np.ndarray,
  *,
  batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Precomputes raw target NAP and canonically projected NAPs via batched JIT."""
  n_ex, t_len = target_wavs.shape
  n_ch = cochlea_model.num_channels
  n_bins = proj_w.shape[1]
  proj_j = jnp.asarray(proj_w, dtype=jnp.float32)
  rest_j = jnp.asarray(rest_nap, dtype=jnp.float32)

  @jax.jit
  def _forward_pair(
    tgt_b: jnp.ndarray, ws_b: jnp.ndarray
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t_raw = jax.vmap(lambda w: cochlea_model(w[:, None])[:, :, 0])(tgt_b)
    w_raw = jax.vmap(lambda w: cochlea_model(w[:, None])[:, :, 0])(ws_b)
    t_canon = jnp.einsum("btc,ck->btk", t_raw - rest_j[None, None, :], proj_j)
    w_canon = jnp.einsum("btc,ck->btk", w_raw - rest_j[None, None, :], proj_j)
    return t_raw, t_canon, w_canon

  raw_tgt_naps = np.zeros((n_ex, t_len, n_ch), dtype=np.float32)
  canon_tgt_naps = np.zeros((n_ex, t_len, n_bins), dtype=np.float32)
  canon_ws_naps = np.zeros((n_ex, t_len, n_bins), dtype=np.float32)

  for start in range(0, n_ex, batch_size):
    end = min(n_ex, start + batch_size)
    k = end - start
    tgt_chunk = target_wavs[start:end]
    ws_chunk = ws_wavs[start:end]
    if k < batch_size:
      pad = batch_size - k
      tgt_chunk = np.pad(tgt_chunk, ((0, pad), (0, 0)))
      ws_chunk = np.pad(ws_chunk, ((0, pad), (0, 0)))
    t_raw, t_can, w_can = _forward_pair(
      jnp.asarray(tgt_chunk, dtype=jnp.float32),
      jnp.asarray(ws_chunk, dtype=jnp.float32),
    )
    raw_tgt_naps[start:end] = np.asarray(t_raw[:k], dtype=np.float32)
    canon_tgt_naps[start:end] = np.asarray(t_can[:k], dtype=np.float32)
    canon_ws_naps[start:end] = np.asarray(w_can[:k], dtype=np.float32)

  return raw_tgt_naps, canon_tgt_naps, canon_ws_naps


def _evaluate_split(
  eval_batch_fn: Callable[..., dict[str, jnp.ndarray]],
  mesh: jax.sharding.Mesh,
  ws_wavs: np.ndarray,
  tgt_wavs: np.ndarray,
  canon_tgt_naps: np.ndarray,
  canon_ws_naps: np.ndarray,
  raw_tgt_naps: np.ndarray,
  params: Mapping[str, Any],
  *,
  batch_size: int = 8,
) -> dict[str, float]:
  """Evaluates mean metrics across all examples in ``tgt_wavs``."""
  n_ex = len(tgt_wavs)
  sums: dict[str, float] = {}
  total_count = 0
  for start in range(0, n_ex, batch_size):
    end = min(n_ex, start + batch_size)
    k = end - start
    ws_b, tc_b, wc_b, tgt_b, tr_b = _place_eval_batch(
      (
        ws_wavs[start:end],
        canon_tgt_naps[start:end],
        canon_ws_naps[start:end],
        tgt_wavs[start:end],
        raw_tgt_naps[start:end],
      ),
      mesh,
    )
    metrics = eval_batch_fn(params, ws_b, tc_b, wc_b, tgt_b, tr_b)
    for key, val in metrics.items():
      sums[key] = sums.get(key, 0.0) + float(np.asarray(val)) * k
    total_count += k
  return {k: v / max(total_count, 1) for k, v in sums.items()}


def run_training(
  *,
  train_cache: str,
  val_cache: str,
  output_checkpoint: str,
  stage1_steps: int = 1400,
  stage2_steps: int = 350,
  batch_size: int = 8,
  stage2_batch_size: int = 4,
  stage1_lr: float = 4e-4,
  stage2_lr: float = 1.5e-4,
  eval_every: int = 150,
  carfac_weight: float = 15.0,
  seed: int = 0,
  model_config: neural.NeuralRefinerConfig | None = None,
  objective: str = "exact",
  init_checkpoint: str | None = None,
  perceptual_steps: int = 1500,
  perceptual_batch_size: int = 4,
  perceptual_lr: float = 2e-4,
  perceptual_nap_weight: float = 0.30,
  sai_fft_len: int = 2048,
  drift_weight: float = 0.5,
  hf_diff_weight: float = 0.0,
  steps_per_loop: int = 1,
  compilation_cache_dir: str | None = None,
  matmul_precision: str = "high",
) -> TrainingResult:
  """Trains a Phase 3 (`exact`) or Phase 3b (`perceptual`) residual refiner."""
  if objective == "perceptual":
    return _run_perceptual_training(
      train_cache=train_cache,
      val_cache=val_cache,
      output_checkpoint=output_checkpoint,
      init_checkpoint=init_checkpoint,
      steps=perceptual_steps,
      batch_size=perceptual_batch_size,
      lr=perceptual_lr,
      eval_every=eval_every,
      perceptual_nap_weight=perceptual_nap_weight,
      sai_fft_len=sai_fft_len,
      drift_weight=drift_weight,
      hf_diff_weight=hf_diff_weight,
      seed=seed,
      model_config=model_config,
      steps_per_loop=steps_per_loop,
      compilation_cache_dir=compilation_cache_dir,
      matmul_precision=matmul_precision,
    )
  if objective != "exact":
    raise ValueError(
      f"Unsupported objective {objective!r}; expected 'exact' or 'perceptual'."
    )
  if steps_per_loop < 1:
    raise ValueError(f"steps_per_loop must be >= 1, got {steps_per_loop}.")

  t0 = time.perf_counter()
  backend, n_devices = tpu.configure_runtime(
    compilation_cache_dir=compilation_cache_dir,
    matmul_precision=matmul_precision,
  )
  mesh = tpu.make_data_mesh()
  if stage1_steps > 0:
    _validate_batch_size("batch_size", batch_size, mesh)
  if stage2_steps > 0:
    _validate_batch_size("stage2_batch_size", stage2_batch_size, mesh)

  cfg = neural.NeuralRefinerConfig() if model_config is None else model_config
  model = neural.NapResidualRefiner(config=cfg)
  fs = int(round(cfg.sample_rate_hz))

  with np.load(train_cache) as d_tr:
    tr_tgt_w = np.asarray(d_tr["target_wavs"], dtype=np.float32)
    tr_ws_w = np.asarray(d_tr["ws_wavs"], dtype=np.float32)
    tr_has_naps = "target_naps" in d_tr
  with np.load(val_cache) as d_va:
    va_tgt_w = np.asarray(d_va["target_wavs"], dtype=np.float32)
    va_ws_w = np.asarray(d_va["ws_wavs"], dtype=np.float32)
    va_has_naps = "target_naps" in d_va

  n_train, t_len = tr_tgt_w.shape
  with tpu.host_cpu_context():
    cochlea_model = carfac.build(fs=fs)
    rest_nap = np.asarray(
      cochlea_model(jnp.zeros((min(512, t_len), 1), dtype=jnp.float32))[
        -1, :, 0
      ],
      dtype=np.float32,
    )
    proj_w = neural.cochlea_cf_projection(cochlea_model, cfg)
    cfs = np.asarray(cochlea_model.channel_frequencies, dtype=np.float32)
    cochlea_host = jax.tree_util.tree_map(
      jax.lax.stop_gradient, tpu.to_host_tree(cochlea_model)
    )
  hf_chan_mask = tpu.replicate_tree(
    jnp.asarray((cfs >= 3500.0).astype(np.float32)), mesh
  )
  rest_nap_j = tpu.replicate_tree(
    jnp.asarray(rest_nap, dtype=jnp.float32), mesh
  )
  fixed_cochlea = tpu.replicate_tree(cochlea_host, mesh)

  print(
    f"Precomputing cochlear features ({n_train} train, {len(va_tgt_w)} val, "
    f"backend={backend}, devices={n_devices}, K={steps_per_loop})...",
    flush=True,
  )
  if tr_has_naps:
    _, tr_can_tgt, tr_can_ws, tr_raw_nap = _load_perceptual_cache(
      train_cache, fixed_cochlea, proj_w, rest_nap
    )
  else:
    tr_raw_nap, tr_can_tgt, tr_can_ws = _precompute_cochlear_features(
      tr_tgt_w, tr_ws_w, fixed_cochlea, proj_w, rest_nap
    )
  if va_has_naps:
    _, va_can_tgt, va_can_ws, va_raw_nap = _load_perceptual_cache(
      val_cache, fixed_cochlea, proj_w, rest_nap
    )
  else:
    va_raw_nap, va_can_tgt, va_can_ws = _precompute_cochlear_features(
      va_tgt_w, va_ws_w, fixed_cochlea, proj_w, rest_nap
    )
  print(
    f"Feature precompute done in {time.perf_counter() - t0:.1f}s", flush=True
  )

  np_rng = np.random.default_rng(seed)
  with tpu.host_cpu_context():
    rng = jax.random.PRNGKey(seed)
    raw_params = model.init(
      rng,
      jnp.zeros((1, t_len), dtype=jnp.float32),
      jnp.zeros((1, t_len, cfg.n_coch_bins), dtype=jnp.float32),
      jnp.zeros((1, t_len, cfg.n_coch_bins), dtype=jnp.float32),
    )
    init_params_host = tpu.to_host_tree(raw_params)
  params = cast(Mapping[str, Any], tpu.replicate_tree(init_params_host, mesh))
  n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
  print(
    f"Initialized NapResidualRefiner with {n_params:,} parameters", flush=True
  )

  def _eval_and_loss(
    p: Mapping[str, Any],
    ws_b: jnp.ndarray,
    tc_b: jnp.ndarray,
    wc_b: jnp.ndarray,
    tgt_b: jnp.ndarray,
    tr_b: jnp.ndarray,
    *,
    use_carfac: bool,
  ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    pred = cast(jnp.ndarray, model.apply(p, ws_b, tc_b, wc_b))
    return compute_refiner_losses(
      pred,
      tgt_b,
      sample_rate_hz=fs,
      cochlea_model=fixed_cochlea if use_carfac else None,
      target_nap_raw=tr_b if use_carfac else None,
      rest_nap_raw=rest_nap_j if use_carfac else None,
      hf_chan_mask=hf_chan_mask if use_carfac else None,
      carfac_weight=carfac_weight if use_carfac else 0.0,
    )

  @functools.partial(jax.jit, static_argnames=("use_carfac",))
  def eval_step(
    p: Mapping[str, Any],
    ws_b: jnp.ndarray,
    tc_b: jnp.ndarray,
    wc_b: jnp.ndarray,
    tgt_b: jnp.ndarray,
    tr_b: jnp.ndarray,
    *,
    use_carfac: bool,
  ) -> dict[str, jnp.ndarray]:
    _, metrics = _eval_and_loss(
      p, ws_b, tc_b, wc_b, tgt_b, tr_b, use_carfac=use_carfac
    )
    return metrics

  def make_train_fns(tx: optax.GradientTransformation, *, use_carfac: bool):
    def step_impl(
      p: Mapping[str, Any],
      opt_state: optax.OptState,
      batch: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
      ],
    ) -> tuple[Mapping[str, Any], optax.OptState, dict[str, jnp.ndarray]]:
      ws_b, tc_b, wc_b, tgt_b, tr_b = batch
      (_, metrics), grads = jax.value_and_grad(
        functools.partial(_eval_and_loss, use_carfac=use_carfac),
        has_aux=True,
      )(p, ws_b, tc_b, wc_b, tgt_b, tr_b)
      updates, next_opt = tx.update(grads, opt_state, p)
      next_p = cast(Mapping[str, Any], optax.apply_updates(p, updates))
      return next_p, next_opt, metrics

    return tpu.make_multi_step_fns(step_impl, donate_argnums=(0, 1))

  out_path = pathlib.Path(output_checkpoint)
  stage1_path = out_path.with_name(f"{out_path.stem}_stage1.npz")
  history: list[dict[str, float]] = []
  train_arrays = (tr_ws_w, tr_can_tgt, tr_can_ws, tr_tgt_w, tr_raw_nap)

  # Baseline val evaluation (step 0: should match warm start cleanly).
  val_b = min(batch_size, len(va_tgt_w))
  eval_s1_fn = functools.partial(eval_step, use_carfac=False)
  init_val = _evaluate_split(
    eval_s1_fn,
    mesh,
    va_ws_w,
    va_tgt_w,
    va_can_tgt,
    va_can_ws,
    va_raw_nap,
    params,
    batch_size=val_b,
  )
  print(
    f"[step 0 / warm-start baseline] val_loss={init_val['total']:.4f}, "
    f"val_sisdr={init_val['sisdr_db']:.2f} dB, stft={init_val['stft']:.4f}, "
    f"hf={init_val['hf_spec']:.4f}, log_rms={init_val['log_rms']:.4f}",
    flush=True,
  )
  best_s1_loss = float(init_val["total"])
  best_s1_sisdr = float(init_val["sisdr_db"])
  best_s1_params_host = init_params_host

  if stage1_steps > 0:
    sched1 = optax.warmup_cosine_decay_schedule(
      init_value=stage1_lr * 0.1,
      peak_value=stage1_lr,
      warmup_steps=min(150, max(1, stage1_steps // 10)),
      decay_steps=max(2, stage1_steps),
      end_value=stage1_lr * 0.05,
    )
    tx1 = optax.chain(
      optax.clip_by_global_norm(2.0),
      optax.adamw(learning_rate=sched1, weight_decay=3e-4),
    )
    with tpu.host_cpu_context():
      opt_state1_host = tpu.to_host_tree(tx1.init(init_params_host))
    opt_state1 = tpu.replicate_tree(opt_state1_host, mesh)
    s1_single, s1_multi = make_train_fns(tx1, use_carfac=False)
    t_s1 = time.perf_counter()

    def _s1_eval(p: Mapping[str, Any]) -> dict[str, float]:
      return _evaluate_split(
        eval_s1_fn,
        mesh,
        va_ws_w,
        va_tgt_w,
        va_can_tgt,
        va_can_ws,
        va_raw_nap,
        p,
        batch_size=val_b,
      )

    def _s1_on_eval(
      step_idx: int,
      tr_m: dict[str, float],
      val_m: dict[str, float],
    ) -> None:
      elapsed = time.perf_counter() - t_s1
      print(
        f"[S1 {step_idx:4d}/{stage1_steps} | {elapsed:5.1f}s] "
        f"tr={tr_m['total']:.4f} (sdr={tr_m['sisdr_db']:.2f}) | "
        f"va={val_m['total']:.4f} sdr={val_m['sisdr_db']:.2f}dB "
        f"stft={val_m['stft']:.4f} hf={val_m['hf_spec']:.4f} "
        f"lrms={val_m['log_rms']:.4f}",
        flush=True,
      )
      history.append({"stage": 1.0, "step": float(step_idx), **val_m})

    def _s1_on_best(
      _best_p: Mapping[str, Any],
      val_m: dict[str, float],
    ) -> None:
      nonlocal best_s1_sisdr
      best_s1_sisdr = float(val_m["sisdr_db"])

    params, opt_state1, best_s1_loss, best_s1_params_host = _run_training_stage(
      params=params,
      opt_state=opt_state1,
      mesh=mesh,
      single_step_fn=s1_single,
      multi_step_fn=s1_multi,
      train_arrays=train_arrays,
      np_rng=np_rng,
      total_steps=stage1_steps,
      batch_size=batch_size,
      steps_per_loop=steps_per_loop,
      eval_every=eval_every,
      eval_fn=_s1_eval,
      best_loss_init=best_s1_loss,
      best_params_host_init=best_s1_params_host,
      on_eval=_s1_on_eval,
      on_best=_s1_on_best,
    )

    neural.save_refiner_checkpoint(stage1_path, best_s1_params_host, cfg)
    print(
      f"Saved Stage 1 best checkpoint -> {stage1_path} (val={best_s1_loss:.4f})",
      flush=True,
    )
    tpu.delete_tree((params, opt_state1))
    gc.collect()
    params = cast(
      Mapping[str, Any], tpu.replicate_tree(best_s1_params_host, mesh)
    )

  best_s2_loss = best_s1_loss
  best_s2_sisdr = best_s1_sisdr
  best_s2_params_host = best_s1_params_host

  if stage2_steps > 0:
    eval_s2_fn = functools.partial(eval_step, use_carfac=True)
    s2_b = min(stage2_batch_size, len(va_tgt_w))
    s2_init_val = _evaluate_split(
      eval_s2_fn,
      mesh,
      va_ws_w,
      va_tgt_w,
      va_can_tgt,
      va_can_ws,
      va_raw_nap,
      params,
      batch_size=s2_b,
    )
    print(
      f"[S2 start] val_total={s2_init_val['total']:.4f} "
      f"sdr={s2_init_val['sisdr_db']:.2f}dB nap={s2_init_val['carfac_nap']:.4f}",
      flush=True,
    )
    best_s2_loss = float(s2_init_val["total"])
    best_s2_sisdr = float(s2_init_val["sisdr_db"])

    sched2 = optax.cosine_decay_schedule(
      init_value=stage2_lr,
      decay_steps=max(2, stage2_steps),
      alpha=0.1,
    )
    tx2 = optax.chain(
      optax.clip_by_global_norm(2.0),
      optax.adamw(learning_rate=sched2, weight_decay=3e-4),
    )
    with tpu.host_cpu_context():
      opt_state2_host = tpu.to_host_tree(tx2.init(best_s1_params_host))
    opt_state2 = tpu.replicate_tree(opt_state2_host, mesh)
    s2_single, s2_multi = make_train_fns(tx2, use_carfac=True)
    t_s2 = time.perf_counter()
    s2_eval_cadence = max(50, min(eval_every, max(50, stage2_steps // 4)))

    def _s2_eval(p: Mapping[str, Any]) -> dict[str, float]:
      return _evaluate_split(
        eval_s2_fn,
        mesh,
        va_ws_w,
        va_tgt_w,
        va_can_tgt,
        va_can_ws,
        va_raw_nap,
        p,
        batch_size=s2_b,
      )

    def _s2_on_eval(
      step_idx: int,
      tr_m: dict[str, float],
      val_m: dict[str, float],
    ) -> None:
      elapsed = time.perf_counter() - t_s2
      print(
        f"[S2 {step_idx:4d}/{stage2_steps} | {elapsed:5.1f}s] "
        f"tr={tr_m['total']:.4f} nap={tr_m['carfac_nap']:.4f} | "
        f"va={val_m['total']:.4f} sdr={val_m['sisdr_db']:.2f}dB "
        f"nap={val_m['carfac_nap']:.4f} hf={val_m['hf_spec']:.4f} "
        f"lrms={val_m['log_rms']:.4f}",
        flush=True,
      )
      history.append({"stage": 2.0, "step": float(step_idx), **val_m})

    def _s2_on_best(
      _best_p: Mapping[str, Any],
      val_m: dict[str, float],
    ) -> None:
      nonlocal best_s2_sisdr
      best_s2_sisdr = float(val_m["sisdr_db"])

    params, opt_state2, best_s2_loss, best_s2_params_host = _run_training_stage(
      params=params,
      opt_state=opt_state2,
      mesh=mesh,
      single_step_fn=s2_single,
      multi_step_fn=s2_multi,
      train_arrays=train_arrays,
      np_rng=np_rng,
      total_steps=stage2_steps,
      batch_size=stage2_batch_size,
      steps_per_loop=steps_per_loop,
      eval_every=s2_eval_cadence,
      eval_fn=_s2_eval,
      best_loss_init=best_s2_loss,
      best_params_host_init=best_s2_params_host,
      on_eval=_s2_on_eval,
      on_best=_s2_on_best,
    )
    tpu.delete_tree(opt_state2)

  tpu.delete_tree(params)
  neural.save_refiner_checkpoint(out_path, best_s2_params_host, cfg)
  print(
    f"Saved final checkpoint -> {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)",
    flush=True,
  )
  return TrainingResult(
    best_stage1_val_loss=best_s1_loss,
    best_stage2_val_loss=best_s2_loss,
    best_stage1_sisdr_db=best_s1_sisdr,
    best_stage2_sisdr_db=best_s2_sisdr,
    stage1_checkpoint_path=str(stage1_path) if stage1_steps > 0 else None,
    final_checkpoint_path=str(out_path),
    history=tuple(history),
  )


def main(argv: Sequence[str]) -> None:
  """Entrypoint for the neural refiner training CLI."""
  del argv
  train_cache = _TRAIN_CACHE.value
  val_cache = _VAL_CACHE.value
  output_checkpoint = _OUTPUT_CHECKPOINT.value
  if train_cache is None or val_cache is None or output_checkpoint is None:
    raise ValueError(
      "--train_cache, --val_cache, and --output_checkpoint must all be provided."
    )
  run_training(
    train_cache=train_cache,
    val_cache=val_cache,
    output_checkpoint=output_checkpoint,
    stage1_steps=_STAGE1_STEPS.value,
    stage2_steps=_STAGE2_STEPS.value,
    batch_size=_BATCH_SIZE.value,
    stage2_batch_size=_STAGE2_BATCH_SIZE.value,
    stage1_lr=_STAGE1_LR.value,
    stage2_lr=_STAGE2_LR.value,
    eval_every=_EVAL_EVERY.value,
    carfac_weight=_CARFAC_WEIGHT.value,
    seed=_SEED.value,
    objective=_OBJECTIVE.value,
    init_checkpoint=_INIT_CHECKPOINT.value,
    perceptual_steps=_PERCEPTUAL_STEPS.value,
    perceptual_batch_size=_PERCEPTUAL_BATCH_SIZE.value,
    perceptual_lr=_PERCEPTUAL_LR.value,
    perceptual_nap_weight=_PERCEPTUAL_NAP_WEIGHT.value,
    sai_fft_len=_SAI_FFT_LEN.value,
    drift_weight=_DRIFT_WEIGHT.value,
    hf_diff_weight=_HF_DIFF_WEIGHT.value,
    steps_per_loop=_STEPS_PER_LOOP.value,
    compilation_cache_dir=_COMPILATION_CACHE_DIR.value,
    matmul_precision=_MATMUL_PRECISION.value,
  )


if __name__ == "__main__":
  define_flags()
  app.run(main)
