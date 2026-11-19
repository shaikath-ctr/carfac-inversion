"""Phase 5 TPU training script for clinical conditioning & HASQI/HASPI prescriptions."""

import csv
import dataclasses
import gc
import json
import pathlib
import time
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
from hp_acoustic.io import audio_io
from hp_acoustic.models import carfac
from jax_haspi_hasqi import haspi as haspi_lib
from jax_haspi_hasqi import hasqi as hasqi_lib

from carfac_inversion import arcfac_metrics
from carfac_inversion import clinical
from carfac_inversion import nap_inverter
from carfac_inversion import neural
from carfac_inversion import perceptual_losses
from carfac_inversion import tpu
from carfac_inversion import vocoder_losses
from carfac_inversion.scripts import benchmark_suite
from carfac_inversion.scripts import train_phase4

FLAGS = flags.FLAGS

_OUTPUT_DIR: flags.FlagHolder[str | None]
_DATA_DIR: flags.FlagHolder[str | None]
_INIT_CHECKPOINT: flags.FlagHolder[str | None]
_ARCHITECTURE: flags.FlagHolder[str]
_WIDTH: flags.FlagHolder[int]
_NUM_BLOCKS: flags.FlagHolder[int]
_KERNEL_SIZE: flags.FlagHolder[int]
_COARSE_DIM: flags.FlagHolder[int]
_COARSE_HOP: flags.FlagHolder[int]
_LOOKAHEAD: flags.FlagHolder[int]
_MAX_LOG_GAIN: flags.FlagHolder[float]
_SAMPLE_RATE_HZ: flags.FlagHolder[int]
_SECONDS: flags.FlagHolder[float]
_MASK_SECONDS: flags.FlagHolder[float]
_OUTPUT_HIGHPASS_HZ: flags.FlagHolder[float]
_WARMUP_WEIGHT: flags.FlagHolder[float]
_LAM: flags.FlagHolder[float]
_STFT_WEIGHT: flags.FlagHolder[float]
_HASQI_WEIGHT: flags.FlagHolder[float]
_HASPI_WEIGHT: flags.FlagHolder[float]
_UCL_WEIGHT: flags.FlagHolder[float]
_CLIP_NORM: flags.FlagHolder[float]
_LR: flags.FlagHolder[float]
_LR_SCHEDULE: flags.FlagHolder[str]
_STEPS: flags.FlagHolder[int]
_BATCH_SIZE: flags.FlagHolder[int]
_GRAD_ACCUM_STEPS: flags.FlagHolder[int]
_STEPS_PER_LOOP: flags.FlagHolder[int]
_SHARE_PRESCRIBE: flags.FlagHolder[float]
_SHARE_AURALIZE: flags.FlagHolder[float]
_EVAL_EVERY: flags.FlagHolder[int]
_EVAL_CLIPS: flags.FlagHolder[int]
_SCORE_SECONDS: flags.FlagHolder[float]
_SEED: flags.FlagHolder[int]
_COMPILATION_CACHE_DIR: flags.FlagHolder[str | None]
_MATMUL_PRECISION: flags.FlagHolder[str]

_TRAIN_BISGAARD_CODES = ("N1", "N2", "N3", "N4", "N5", "N6", "S1", "S2", "S3")

_Phase5Batch = tuple[
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
]


def define_flags(flag_values: flags.FlagValues = FLAGS) -> None:
  """Registers Phase 5 CLI flags cleanly."""
  global _OUTPUT_DIR
  global _DATA_DIR
  global _INIT_CHECKPOINT
  global _ARCHITECTURE
  global _WIDTH
  global _NUM_BLOCKS
  global _KERNEL_SIZE
  global _COARSE_DIM
  global _COARSE_HOP
  global _LOOKAHEAD
  global _MAX_LOG_GAIN
  global _SAMPLE_RATE_HZ
  global _SECONDS
  global _MASK_SECONDS
  global _OUTPUT_HIGHPASS_HZ
  global _WARMUP_WEIGHT
  global _LAM
  global _STFT_WEIGHT
  global _HASQI_WEIGHT
  global _HASPI_WEIGHT
  global _UCL_WEIGHT
  global _CLIP_NORM
  global _LR
  global _LR_SCHEDULE
  global _STEPS
  global _BATCH_SIZE
  global _GRAD_ACCUM_STEPS
  global _STEPS_PER_LOOP
  global _SHARE_PRESCRIBE
  global _SHARE_AURALIZE
  global _EVAL_EVERY
  global _EVAL_CLIPS
  global _SCORE_SECONDS
  global _SEED
  global _COMPILATION_CACHE_DIR
  global _MATMUL_PRECISION

  _OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Directory for config, checkpoints, training log, and clinical evals.",
    required=True,
    flag_values=flag_values,
  )
  _DATA_DIR = flags.DEFINE_string(
    "data_dir",
    None,
    "ProcessedClips shard root directory (falls back to synthetic if None).",
    flag_values=flag_values,
  )
  _INIT_CHECKPOINT = flags.DEFINE_string(
    "init_checkpoint",
    None,
    "Optional Phase 4 or Phase 5 checkpoint .npz for warm-starting.",
    flag_values=flag_values,
  )
  _ARCHITECTURE = flags.DEFINE_enum(
    "p4_architecture",
    "cochlea_hybrid",
    ("tcn", "cochlea_hybrid"),
    "Causal inverter backbone architecture.",
    flag_values=flag_values,
  )
  _WIDTH = flags.DEFINE_integer(
    "width", 64, "Causal TCN channel width.", flag_values=flag_values
  )
  _NUM_BLOCKS = flags.DEFINE_integer(
    "num_blocks", 13, "Number of dilated TCN blocks.", flag_values=flag_values
  )
  _KERNEL_SIZE = flags.DEFINE_integer(
    "kernel_size", 3, "Causal convolution kernel size.", flag_values=flag_values
  )
  _COARSE_DIM = flags.DEFINE_integer(
    "coarse_dim",
    96,
    "Coarse frame branch hidden width.",
    flag_values=flag_values,
  )
  _COARSE_HOP = flags.DEFINE_integer(
    "coarse_hop", 64, "Coarse frame hop in samples.", flag_values=flag_values
  )
  _LOOKAHEAD = flags.DEFINE_integer(
    "lookahead", 256, "Causal lookahead samples.", flag_values=flag_values
  )
  _MAX_LOG_GAIN = flags.DEFINE_float(
    "max_log_gain",
    4.5,
    "Upper clamp on log_gain_t for prescription headroom.",
    flag_values=flag_values,
  )
  _SAMPLE_RATE_HZ = flags.DEFINE_integer(
    "sample_rate_hz",
    32000,
    "Target sample rate in Hz.",
    flag_values=flag_values,
  )
  _SECONDS = flags.DEFINE_float(
    "seconds",
    1.0,
    "Training clip duration in seconds.",
    flag_values=flag_values,
  )
  _MASK_SECONDS = flags.DEFINE_float(
    "mask_seconds",
    0.25,
    "Warmup mask duration in seconds.",
    flag_values=flag_values,
  )
  _OUTPUT_HIGHPASS_HZ = flags.DEFINE_float(
    "output_highpass_hz",
    20.0,
    "Causal high-pass cutoff.",
    flag_values=flag_values,
  )
  _WARMUP_WEIGHT = flags.DEFINE_float(
    "warmup_weight",
    0.01,
    "Warmup energy penalty weight.",
    flag_values=flag_values,
  )
  _LAM = flags.DEFINE_float(
    "lam", 0.01, "Healthy-anchor waveform L1 weight.", flag_values=flag_values
  )
  _STFT_WEIGHT = flags.DEFINE_float(
    "stft_weight",
    0.1,
    "Healthy-anchor MR-STFT weight.",
    flag_values=flag_values,
  )
  _HASQI_WEIGHT = flags.DEFINE_float(
    "hasqi_weight",
    0.05,
    "Differentiable HASQI v2 weight for prescription and healthy clips.",
    flag_values=flag_values,
  )
  _HASPI_WEIGHT = flags.DEFINE_float(
    "haspi_weight",
    0.05,
    "Differentiable HASPI v2 weight for prescription and healthy clips.",
    flag_values=flag_values,
  )
  _UCL_WEIGHT = flags.DEFINE_float(
    "ucl_weight",
    1.0,
    "Soft quadratic comfort limit penalty above 0.95 full scale.",
    flag_values=flag_values,
  )
  _CLIP_NORM = flags.DEFINE_float(
    "clip_norm", 1.0, "Global gradient norm clipping.", flag_values=flag_values
  )
  _LR = flags.DEFINE_float(
    "lr", 1e-3, "Peak Adam learning rate.", flag_values=flag_values
  )
  _LR_SCHEDULE = flags.DEFINE_enum(
    "lr_schedule", "cosine", ("constant", "cosine"), "LR schedule."
  )
  _STEPS = flags.DEFINE_integer(
    "steps", 600, "Total Phase 5 optimization steps.", flag_values=flag_values
  )
  _BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 32, "Global batch size.", flag_values=flag_values
  )
  _GRAD_ACCUM_STEPS = flags.DEFINE_integer(
    "grad_accum_steps",
    2,
    "Gradient accumulation micro-steps.",
    flag_values=flag_values,
  )
  _STEPS_PER_LOOP = flags.DEFINE_integer(
    "steps_per_loop",
    20,
    "Steps fused per jax.lax.scan chunk.",
    flag_values=flag_values,
  )
  _SHARE_PRESCRIBE = flags.DEFINE_float(
    "share_prescribe",
    0.70,
    "Fraction of batch dedicated to inverse hearing-aid prescription.",
    flag_values=flag_values,
  )
  _SHARE_AURALIZE = flags.DEFINE_float(
    "share_auralize",
    0.15,
    "Fraction of batch dedicated to clinical impairment auralization.",
    flag_values=flag_values,
  )
  _EVAL_EVERY = flags.DEFINE_integer(
    "eval_every", 100, "Evaluation cadence in steps.", flag_values=flag_values
  )
  _EVAL_CLIPS = flags.DEFINE_integer(
    "eval_clips", 2, "Held-out evaluation clips.", flag_values=flag_values
  )
  _SCORE_SECONDS = flags.DEFINE_float(
    "score_seconds",
    2.0,
    "Held-out clip length in seconds.",
    flag_values=flag_values,
  )
  _SEED = flags.DEFINE_integer("seed", 5, "PRNG seed.", flag_values=flag_values)
  _COMPILATION_CACHE_DIR = flags.DEFINE_string(
    "compilation_cache_dir",
    "/tmp/carfac_inversion_xla_cache",
    "Persistent XLA cache directory.",
    flag_values=flag_values,
  )
  _MATMUL_PRECISION = flags.DEFINE_enum(
    "matmul_precision",
    "high",
    ("default", "high", "highest", "float32"),
    "JAX default matmul precision.",
    flag_values=flag_values,
  )


def _sample_clinical_profile(
  rng: np.random.Generator, cfs: np.ndarray, sample_rate_hz: float
) -> clinical.ClinicalListenerProfile:
  """Draws a randomized clinical audiogram profile (Bisgaard + ISO 7029)."""
  if rng.random() < 0.5:
    code = _TRAIN_BISGAARD_CODES[
      int(rng.integers(0, len(_TRAIN_BISGAARD_CODES)))
    ]
    aud6 = clinical.bisgaard_audiogram_6(code)
    aud6 = np.clip(aud6 + rng.uniform(-3.0, 3.0, size=(6,)), 0.0, 110.0)
    name = f"bisgaard:{code}"
  else:
    age = float(rng.uniform(25.0, 80.0))
    sex = "male" if rng.random() < 0.5 else "female"
    pct = float(rng.uniform(0.10, 0.95))
    aud6 = clinical.iso7029_audiogram_6(age, sex=sex, percentile=pct)
    name = f"iso7029:{age:.0f}:{sex}:{pct:.2f}"
  return clinical.profile_from_audiogram_6(
    name, aud6, sample_rate_hz=sample_rate_hz, channel_freqs_hz=cfs
  )


def _draw_clinical_task_batch(
  rng: np.random.Generator,
  batch_size: int,
  num_channels: int,
  cfs: np.ndarray,
  sample_rate_hz: float,
  *,
  share_prescribe: float,
  share_auralize: float,
  lam: float,
  stft_weight: float,
) -> tuple[
  np.ndarray,
  np.ndarray,
  np.ndarray,
  np.ndarray,
  np.ndarray,
  np.ndarray,
  np.ndarray,
]:
  """Samples source/target ear states, conditioning, audiograms, and weights."""
  src_ohc = np.ones((batch_size, num_channels), dtype=np.float32)
  src_shift = np.zeros((batch_size, num_channels, 3), dtype=np.float32)
  eval_ohc = np.ones((batch_size, num_channels), dtype=np.float32)
  eval_shift = np.zeros((batch_size, num_channels, 3), dtype=np.float32)
  cond = np.zeros((batch_size, num_channels, 3), dtype=np.float32)
  hl6 = np.zeros((batch_size, 6), dtype=np.float32)
  loss_w = np.zeros((batch_size, 3), dtype=np.float32)

  draws = rng.uniform(size=batch_size)
  for i in range(batch_size):
    u = float(draws[i])
    if u < share_prescribe:
      prof = _sample_clinical_profile(rng, cfs, sample_rate_hz)
      eval_ohc[i] = prof.ohc_health
      eval_shift[i] = prof.v_half_shift
      cond[i] = prof.conditioning_features()
      hl6[i] = prof.audiogram_6_db_hl
      loss_w[i, 2] = 1.0
    elif u < share_prescribe + share_auralize:
      prof = _sample_clinical_profile(rng, cfs, sample_rate_hz)
      src_ohc[i] = prof.ohc_health
      src_shift[i] = prof.v_half_shift
    else:
      loss_w[i, 0] = lam
      loss_w[i, 1] = stft_weight
      loss_w[i, 2] = 1.0
  return src_ohc, src_shift, eval_ohc, eval_shift, cond, hl6, loss_w


def _iter_phase5_batches(
  *,
  wave_iter: Iterator[np.ndarray],
  rng: np.random.Generator,
  num_channels: int,
  cfs: np.ndarray,
  sample_rate_hz: float,
  share_prescribe: float,
  share_auralize: float,
  lam: float,
  stft_weight: float,
) -> Iterator[tuple[np.ndarray, ...]]:
  for wav in wave_iter:
    yield (
      wav,
      *_draw_clinical_task_batch(
        rng,
        wav.shape[0],
        num_channels,
        cfs,
        sample_rate_hz,
        share_prescribe=share_prescribe,
        share_auralize=share_auralize,
        lam=lam,
        stft_weight=stft_weight,
      ),
    )


def _evaluate_phase5(
  *,
  apply_fn: Any,
  params: Mapping[str, Any],
  eval_clips: Sequence[tuple[str, np.ndarray]],
  healthy_cochlea: carfac.CarfacCochleaModel,
  n3_profile: clinical.ClinicalListenerProfile,
  s2_profile: clinical.ClinicalListenerProfile,
  n3_cochlea: carfac.CarfacCochleaModel,
  s2_cochlea: carfac.CarfacCochleaModel,
  sample_rate_hz: int,
  lookahead: int,
  mask_samples: int,
  out_root: pathlib.Path,
  step_idx: int,
  mesh: jax.sharding.Mesh | None = None,
) -> dict[str, Any]:
  """Evaluates healthy round-trip, clinical prescriptions vs NAL-R, and N3 auralize."""
  forward_jit = jax.jit(lambda w, cm: cm(w[:, None])[:, :, 0])

  def _place(arr: np.ndarray) -> jnp.ndarray:
    arr_j = jnp.asarray(arr, dtype=jnp.float32)
    return tpu.replicate_tree(arr_j, mesh) if mesh is not None else arr_j

  level1 = benchmark_suite.CARFAC_RMS_ONE_DB_SPL
  zero_hl = np.zeros((6,), dtype=np.float64)
  rt_nap_vals: list[float] = []
  rt_sdr_vals: list[float] = []
  pres_rel_vals: list[float] = []
  pres_heal_vals: list[float] = []
  pres_hasqi_vals: list[float] = []
  pres_haspi_vals: list[float] = []
  nalr_rel_vals: list[float] = []
  nalr_heal_vals: list[float] = []
  nalr_hasqi_vals: list[float] = []
  nalr_haspi_vals: list[float] = []
  aur_kept_vals: list[float] = []

  render_dir = out_root / "renders" / f"step{step_idx}"
  render_dir.mkdir(parents=True, exist_ok=True)
  scores_dir = out_root / "scores"
  scores_dir.mkdir(parents=True, exist_ok=True)

  for clip_id, raw_x in eval_clips:
    x = benchmark_suite._scale_to_spl(raw_x, 65.0)
    mask = min(mask_samples, len(x) // 4)
    x_pad = np.concatenate([x, np.zeros(lookahead, dtype=np.float32)])
    n_hx_pad = np.asarray(
      forward_jit(_place(x_pad), healthy_cochlea), dtype=np.float32
    )
    n_hx = n_hx_pad[: len(x)]
    hx_norm = max(float(np.mean(np.abs(n_hx[mask:]))), 1e-8)

    # 1) Healthy round-trip anchor.
    z_cond = _place(
      np.zeros((1, len(n3_profile.channel_freqs_hz), 3), np.float32)
    )
    y_rt = np.asarray(
      apply_fn(params, _place(n_hx_pad[None]), z_cond)[0], dtype=np.float32
    )[lookahead : lookahead + len(x)]
    n_rt = np.asarray(
      forward_jit(_place(y_rt), healthy_cochlea), dtype=np.float32
    )
    with tpu.host_cpu_context():
      rt_sc = arcfac_metrics.score_arcfac(
        x,
        y_rt,
        n_target=n_hx,
        n_healthy_ref=n_hx,
        n_healthy_est=n_rt,
        sample_rate_hz=sample_rate_hz,
        delay=0,
        mask_samples=mask,
      )
      rt_m = benchmark_suite._score_pair(
        ref=x,
        est=y_rt,
        target_nap=n_hx,
        recon_nap=n_rt,
        healthy_ref_nap=n_hx,
        sample_rate_hz=sample_rate_hz,
        hearing_loss_db=zero_hl,
      )
    rt_nap_vals.append(float(rt_sc["nap"]["round_trip_rel"]))
    rt_sdr_vals.append(float(rt_m["si_sdr"]))

    # 2) Clinical hearing-aid prescription (Bisgaard N3 & S2) vs NAL-R.
    for p_name, prof, p_cm in (
      ("N3", n3_profile, n3_cochlea),
      ("S2", s2_profile, s2_cochlea),
    ):
      c_feat = _place(prof.conditioning_features()[None, :, :])
      y_aid = np.asarray(
        apply_fn(params, _place(n_hx_pad[None]), c_feat)[0], dtype=np.float32
      )[lookahead : lookahead + len(x)]
      y_aid = np.clip(y_aid, -0.99, 0.99).astype(np.float32)
      y_nalr = clinical.apply_nalr(
        x, prof.audiogram_6_db_hl, float(sample_rate_hz)
      )

      n_unaided = np.asarray(
        forward_jit(_place(x_pad), p_cm), dtype=np.float32
      )[: len(x)]
      n_aid = np.asarray(forward_jit(_place(y_aid), p_cm), dtype=np.float32)
      n_nalr = np.asarray(forward_jit(_place(y_nalr), p_cm), dtype=np.float32)

      unaided_gap = max(
        float(np.mean(np.abs(n_unaided[mask:] - n_hx[mask:]))), 1e-8
      )
      aid_err = float(np.mean(np.abs(n_aid[mask:] - n_hx[mask:])))
      nalr_err = float(np.mean(np.abs(n_nalr[mask:] - n_hx[mask:])))
      pres_rel_vals.append(aid_err / hx_norm)
      pres_heal_vals.append(1.0 - aid_err / unaided_gap)
      nalr_rel_vals.append(nalr_err / hx_norm)
      nalr_heal_vals.append(1.0 - nalr_err / unaided_gap)

      aud_hl = np.asarray(prof.audiogram_6_db_hl, dtype=np.float64)
      with tpu.host_cpu_context():
        p_aid, _ = haspi_lib.haspi_v2(
          x, sample_rate_hz, y_aid, sample_rate_hz, aud_hl, level1=level1
        )
        q_aid, _, _, _ = hasqi_lib.hasqi_v2(
          x, sample_rate_hz, y_aid, sample_rate_hz, aud_hl, level1=level1
        )
        p_nalr, _ = haspi_lib.haspi_v2(
          x, sample_rate_hz, y_nalr, sample_rate_hz, aud_hl, level1=level1
        )
        q_nalr, _, _, _ = hasqi_lib.hasqi_v2(
          x, sample_rate_hz, y_nalr, sample_rate_hz, aud_hl, level1=level1
        )
      pres_haspi_vals.append(float(p_aid))
      pres_hasqi_vals.append(float(q_aid))
      nalr_haspi_vals.append(float(p_nalr))
      nalr_hasqi_vals.append(float(q_nalr))
      audio_io.save_audio_to_file(
        y_aid,
        str(render_dir / f"{clip_id}-prescribe-{p_name}.wav"),
        sample_rate_hz,
      )

      if p_name == "N3":
        # 3) Clinical cross-cochlea auralization check on Bisgaard N3.
        n_aur_in_pad = np.asarray(
          forward_jit(_place(x_pad), p_cm), dtype=np.float32
        )
        y_aur = np.asarray(
          apply_fn(params, _place(n_aur_in_pad[None]), z_cond)[0],
          dtype=np.float32,
        )[lookahead : lookahead + len(x)]
        n_aur_h = np.asarray(
          forward_jit(_place(y_aur), healthy_cochlea), dtype=np.float32
        )
        with tpu.host_cpu_context():
          aur_sc = arcfac_metrics.score_arcfac(
            x,
            y_aur,
            n_target=n_aur_in_pad[: len(x)],
            n_healthy_ref=n_hx,
            n_healthy_est=n_aur_h,
            sample_rate_hz=sample_rate_hz,
            delay=0,
            mask_samples=mask,
          )
        aur_kept_vals.append(float(aur_sc["nap"].get("kept", float("nan"))))

  m_pres_rel = float(np.nanmean(pres_rel_vals))
  m_pres_q = float(np.nanmean(pres_hasqi_vals))
  m_rt_rel = float(np.nanmean(rt_nap_vals))
  summary = {
    "nap_rt_rel": m_rt_rel,
    "rt_si_sdr_db": float(np.nanmean(rt_sdr_vals)),
    "prescribe_nap_rel": m_pres_rel,
    "prescribe_nap_healed_pct": 100.0 * float(np.nanmean(pres_heal_vals)),
    "prescribe_haspi": float(np.nanmean(pres_haspi_vals)),
    "prescribe_hasqi": m_pres_q,
    "nalr_nap_rel": float(np.nanmean(nalr_rel_vals)),
    "nalr_nap_healed_pct": 100.0 * float(np.nanmean(nalr_heal_vals)),
    "nalr_haspi": float(np.nanmean(nalr_haspi_vals)),
    "nalr_hasqi": float(np.nanmean(nalr_hasqi_vals)),
    "auralize_nap_kept": float(np.nanmean(aur_kept_vals)),
    "selection_loss": m_pres_rel + 0.5 * (1.0 - m_pres_q) + 0.5 * m_rt_rel,
  }
  with (scores_dir / f"step{step_idx}.json").open("w") as f:
    json.dump({"step": step_idx, "summary": summary}, f, indent=1)
  return summary


def run_phase5_training(
  *,
  output_dir: str,
  data_dir: str | None = None,
  init_checkpoint: str | None = None,
  architecture: str = "cochlea_hybrid",
  width: int = 64,
  num_blocks: int = 13,
  kernel_size: int = 3,
  coarse_dim: int = 96,
  coarse_hop: int = 64,
  lookahead: int = 256,
  max_log_gain: float = 4.5,
  sample_rate_hz: int = 32000,
  seconds: float = 1.0,
  mask_seconds: float = 0.25,
  output_highpass_hz: float = 20.0,
  warmup_weight: float = 0.01,
  lam: float = 0.01,
  stft_weight: float = 0.1,
  hasqi_weight: float = 0.05,
  haspi_weight: float = 0.05,
  ucl_weight: float = 1.0,
  clip_norm: float = 1.0,
  lr: float = 1e-3,
  lr_schedule: str = "cosine",
  steps: int = 600,
  batch_size: int = 32,
  grad_accum_steps: int = 2,
  steps_per_loop: int = 20,
  share_prescribe: float = 0.70,
  share_auralize: float = 0.15,
  eval_every: int = 100,
  eval_clips_count: int = 2,
  score_seconds: float = 2.0,
  seed: int = 5,
  compilation_cache_dir: str | None = None,
  matmul_precision: str = "high",
) -> dict[str, Any]:
  """Trains conditioned Phase 5 clinical prescription & auralization models."""
  if steps < 1:
    raise ValueError(f"steps must be >= 1, got {steps}.")
  if share_prescribe + share_auralize > 1.0:
    raise ValueError("share_prescribe + share_auralize must be <= 1.0.")

  backend, n_devices = tpu.configure_runtime(
    compilation_cache_dir=compilation_cache_dir,
    matmul_precision=matmul_precision,
  )
  mesh = tpu.make_data_mesh()
  if (
    batch_size < n_devices * grad_accum_steps
    or batch_size % (n_devices * grad_accum_steps) != 0
  ):
    raise ValueError(
      f"batch_size={batch_size} must be divisible by n_devices*grad_accum_steps="
      f"{n_devices * grad_accum_steps}."
    )
  micro_batch_size = batch_size // grad_accum_steps

  out_root = pathlib.Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)
  log_path = out_root / "log.csv"
  with log_path.open("w", newline="") as f:
    csv.writer(f).writerow(
      (
        "step",
        "loss",
        "nap_l1",
        "hasqi",
        "haspi",
        "stft",
        "ucl",
        "finite",
        "wall_s",
      )
    )

  num_samples = max(2048, int(round(sample_rate_hz * seconds)))
  mask_samples = min(
    int(round(sample_rate_hz * mask_seconds)), num_samples // 2
  )
  start_idx = max(mask_samples, lookahead)
  percep_len = min(int(round(0.35 * sample_rate_hz)), num_samples - start_idx)
  nominal_rms_scale = 10.0 ** (
    (65.0 - benchmark_suite.CARFAC_RMS_ONE_DB_SPL) / 20.0
  )

  with tpu.host_cpu_context():
    healthy_host = neural.cochlea_to_host(carfac.build(fs=sample_rate_hz))
  cfs = np.asarray(healthy_host.channel_frequencies, dtype=np.float32)
  num_channels = int(healthy_host.num_channels)
  healthy_dev = neural.replicate_cochlea(healthy_host, mesh)

  n3_prof = clinical.build_clinical_profile(
    "bisgaard:N3", sample_rate_hz=float(sample_rate_hz), channel_freqs_hz=cfs
  )
  s2_prof = clinical.build_clinical_profile(
    "bisgaard:S2", sample_rate_hz=float(sample_rate_hz), channel_freqs_hz=cfs
  )
  with tpu.host_cpu_context():
    n3_dev = neural.replicate_cochlea(
      neural.cochlea_to_host(
        clinical.build_cochlea_from_profile(n3_prof, base_cochlea=healthy_host)
      ),
      mesh,
    )
    s2_dev = neural.replicate_cochlea(
      neural.cochlea_to_host(
        clinical.build_cochlea_from_profile(s2_prof, base_cochlea=healthy_host)
      ),
      mesh,
    )

  eval_clips = benchmark_suite.load_speech_clips(
    data_dir,
    sample_rate_hz=sample_rate_hz,
    max_clips=eval_clips_count,
    clip_seconds=score_seconds,
    split="test",
  )
  if data_dir is not None:
    wave_iter = train_phase4._build_grain_waveform_iter(
      data_dir=data_dir,
      sample_rate_hz=sample_rate_hz,
      seconds=float(num_samples) / float(sample_rate_hz),
      batch_size=batch_size,
      seed=seed,
    )
  else:
    wave_iter = train_phase4._build_synthetic_waveform_iter(
      sample_rate_hz=sample_rate_hz,
      num_samples=num_samples,
      batch_size=batch_size,
      seed=seed,
    )

  if init_checkpoint:
    _, warm_p, warm_cfg = nap_inverter.load_direct_checkpoint(init_checkpoint)
    cfg = dataclasses.replace(
      warm_cfg,
      conditioned=True,
      cond_dim=3,
      max_log_gain=max_log_gain,
    )
    model = nap_inverter.CochleaDirectInverter(config=cfg)
    with tpu.host_cpu_context():
      raw_init = model.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, min(1024, num_samples), num_channels), dtype=jnp.float32),
        cond=jnp.zeros((1, num_channels, 3), dtype=jnp.float32),
      )
      init_params_host = tpu.to_host_tree(
        nap_inverter.merge_warmstart_params(raw_init, warm_p)
      )
  else:
    calib_x_np = next(wave_iter)
    if float(np.sqrt(np.mean(calib_x_np**2))) < 1e-6:
      calib_x_np = next(wave_iter)
    healthy_ear_fn = jax.jit(
      jax.vmap(lambda w: healthy_dev(w[:, None])[:, :, 0])
    )
    calib_nap = healthy_ear_fn(tpu.shard_batch(calib_x_np, mesh, stacked=False))
    calib_in_scale = max(float(jnp.sqrt(jnp.mean(calib_nap**2))), 1e-6)
    calib_out_scale = max(float(np.sqrt(np.mean(calib_x_np**2))), 1e-6)
    cfg = nap_inverter.DirectInverterConfig(
      architecture=architecture,
      width=width,
      kernel_size=kernel_size,
      num_blocks=num_blocks,
      coarse_dim=coarse_dim,
      coarse_hop=coarse_hop,
      input_scale=calib_in_scale,
      output_scale=calib_out_scale,
      lookahead=lookahead,
      output_highpass_hz=output_highpass_hz,
      fs=float(sample_rate_hz),
      conditioned=True,
      cond_dim=3,
      max_log_gain=max_log_gain,
    )
    model = nap_inverter.CochleaDirectInverter(config=cfg)
    with tpu.host_cpu_context():
      raw_init = model.init(
        jax.random.PRNGKey(seed),
        jnp.zeros((1, min(1024, num_samples), num_channels), dtype=jnp.float32),
        cond=jnp.zeros((1, num_channels, 3), dtype=jnp.float32),
      )
      init_params_host = tpu.to_host_tree(raw_init)

  params = cast(Mapping[str, Any], tpu.replicate_tree(init_params_host, mesh))
  n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
  batch_iter = _iter_phase5_batches(
    wave_iter=wave_iter,
    rng=np.random.default_rng(seed + 1),
    num_channels=num_channels,
    cfs=cfs,
    sample_rate_hz=float(sample_rate_hz),
    share_prescribe=share_prescribe,
    share_auralize=share_auralize,
    lam=lam,
    stft_weight=stft_weight,
  )

  ear_vmap_fn = jax.vmap(
    lambda w, o, s: train_phase4._apply_ear_with_profile(healthy_dev, w, o, s),
    in_axes=(0, 0, 0),
  )
  stft_res = vocoder_losses.stft_resolutions_for_rate(float(sample_rate_hz))
  level1 = benchmark_suite.CARFAC_RMS_ONE_DB_SPL

  with (out_root / "config.json").open("w") as f:
    json.dump(
      {
        "cfg": dataclasses.asdict(cfg),
        "params": int(n_params),
        "steps": steps,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "hasqi_weight": hasqi_weight,
        "haspi_weight": haspi_weight,
        "share_prescribe": share_prescribe,
        "share_auralize": share_auralize,
        "backend": backend,
        "n_devices": n_devices,
      },
      f,
      indent=1,
      default=str,
    )

  if lr_schedule == "cosine" and steps > 1:
    sched: optax.Schedule | float = optax.warmup_cosine_decay_schedule(
      init_value=lr * 0.1,
      peak_value=lr,
      warmup_steps=min(100, max(1, steps // 10)),
      decay_steps=max(2, steps),
      end_value=lr * 0.05,
    )
  else:
    sched = lr
  tx = (
    optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(sched))
    if clip_norm > 0.0
    else optax.adam(sched)
  )
  with tpu.host_cpu_context():
    opt_state_host = tpu.to_host_tree(tx.init(init_params_host))
  opt_state = tpu.replicate_tree(opt_state_host, mesh)

  def _step_loss(
    p: Mapping[str, Any], micro: _Phase5Batch
  ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    x_b, src_ohc, src_shift, eval_ohc, eval_shift, cond_b, hl6_b, lw_b = micro
    tgt_nap = jax.lax.stop_gradient(ear_vmap_fn(x_b, src_ohc, src_shift))
    y_b = cast(jnp.ndarray, model.apply(p, tgt_nap, cond=cond_b))
    nap_y = ear_vmap_fn(y_b, eval_ohc, eval_shift)

    tgt_delayed = vocoder_losses.delay_samples(tgt_nap, lookahead, axis=1)
    x_delayed = vocoder_losses.delay_samples(x_b, lookahead, axis=1)

    nap_steady = jnp.mean(
      jnp.abs(nap_y[:, start_idx:] - tgt_delayed[:, start_idx:]), axis=(1, 2)
    )
    if start_idx > lookahead:
      nap_onset = jnp.mean(
        jnp.abs(
          nap_y[:, lookahead:start_idx] - tgt_delayed[:, lookahead:start_idx]
        ),
        axis=(1, 2),
      )
      nap_total = nap_steady + 0.25 * nap_onset
    else:
      nap_total = nap_steady

    abs_floor = max(0.1 * float(nominal_rms_scale), 1e-7)
    pow_floor = abs_floor**2
    lam_b = lw_b[:, 0]
    stft_b = lw_b[:, 1]
    percep_b = lw_b[:, 2]

    wave_per_clip = jnp.mean(
      jnp.abs(y_b[:, lookahead:] - x_delayed[:, lookahead:]), axis=1
    ) / jnp.maximum(
      jnp.mean(jnp.abs(x_delayed[:, lookahead:]), axis=1), abs_floor
    )

    look_e = (
      jnp.mean(y_b[:, :lookahead] ** 2, axis=1)
      if lookahead > 0
      else jnp.zeros((y_b.shape[0],), dtype=jnp.float32)
    )
    norm_e = jnp.maximum(
      jnp.maximum(
        jnp.mean(y_b[:, start_idx:] ** 2, axis=1),
        jnp.mean(x_delayed[:, start_idx:] ** 2, axis=1),
      ),
      pow_floor,
    )
    is_healthy_anchor = (stft_b > 0.0).astype(jnp.float32)
    excess_e = is_healthy_anchor * jnp.mean(
      jnp.maximum(y_b[:, :start_idx] ** 2 - x_delayed[:, :start_idx] ** 2, 0.0),
      axis=1,
    )
    warm_per_clip = (look_e + excess_e) / norm_e

    ucl_per_clip = jnp.mean(jax.nn.relu(jnp.abs(y_b) - 0.95) ** 2, axis=1)
    total_per_clip = (
      nap_total
      + lam_b * wave_per_clip
      + warmup_weight * warm_per_clip
      + float(ucl_weight) * ucl_per_clip
    )

    if stft_weight > 0.0:
      stft_per_clip = jax.vmap(
        lambda a, b: vocoder_losses.multi_resolution_stft_loss(
          a[None, :], b[None, :], resolutions=stft_res
        )[0]
      )(y_b[:, lookahead:], x_delayed[:, lookahead:])
      total_per_clip = total_per_clip + stft_b * stft_per_clip
      stft_mean = jnp.sum(
        jnp.where(stft_b > 0.0, stft_per_clip, 0.0)
      ) / jnp.maximum(jnp.sum(stft_b > 0.0), 1.0)
    else:
      stft_mean = jnp.float32(0.0)

    total = jnp.mean(total_per_clip)
    if (hasqi_weight > 0.0 or haspi_weight > 0.0) and percep_len > 256:
      p_loss, p_aux = perceptual_losses.batched_hasqi_haspi_loss(
        x_delayed[:, start_idx : start_idx + percep_len],
        y_b[:, start_idx : start_idx + percep_len],
        hl6_b,
        mask=percep_b,
        sample_rate_hz=float(sample_rate_hz),
        level1=level1,
        hasqi_weight=hasqi_weight,
        haspi_weight=haspi_weight,
      )
      total = total + p_loss
      q_mean = p_aux["hasqi"]
      p_mean = p_aux["haspi"]
    else:
      q_mean = jnp.float32(0.0)
      p_mean = jnp.float32(0.0)

    return total, {
      "loss": cast(jnp.ndarray, total),
      "nap_l1": cast(jnp.ndarray, jnp.mean(nap_steady)),
      "hasqi": q_mean,
      "haspi": p_mean,
      "stft": stft_mean,
      "ucl": cast(jnp.ndarray, jnp.mean(ucl_per_clip)),
    }

  def _train_step(
    p: Mapping[str, Any], os_s: optax.OptState, batch: _Phase5Batch
  ) -> tuple[Mapping[str, Any], optax.OptState, dict[str, jnp.ndarray]]:
    if grad_accum_steps == 1:
      (_, avg_m), grads = jax.value_and_grad(_step_loss, has_aux=True)(p, batch)
    else:
      micros = train_phase4._split_microbatches(batch, grad_accum_steps)
      zero_g = jax.tree.map(jnp.zeros_like, p)
      zero_m = {
        k: jnp.zeros((), dtype=jnp.float32)
        for k in ("loss", "nap_l1", "hasqi", "haspi", "stft", "ucl")
      }

      def _accum(
        carry: tuple[Mapping[str, Any], dict[str, jnp.ndarray]],
        micro: _Phase5Batch,
      ) -> tuple[tuple[Mapping[str, Any], dict[str, jnp.ndarray]], None]:
        g_acc, m_acc = carry
        (_, aux), g_micro = jax.value_and_grad(_step_loss, has_aux=True)(
          p, micro
        )
        return (
          jax.tree.map(jnp.add, g_acc, g_micro),
          {k: m_acc[k] + aux[k] for k in m_acc},
        ), None

      (g_sum, m_sum), _ = jax.lax.scan(_accum, (zero_g, zero_m), micros)
      inv_a = jnp.float32(1.0 / float(grad_accum_steps))
      grads = jax.tree.map(lambda g: g * inv_a, g_sum)
      avg_m = {k: v * inv_a for k, v in m_sum.items()}
    updates, next_os = tx.update(grads, os_s, p)
    cand_p = cast(Mapping[str, Any], optax.apply_updates(p, updates))
    fin = jnp.isfinite(avg_m["loss"]) & jnp.all(
      jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads)])
    )
    out_p = jax.tree.map(lambda a, b: jnp.where(fin, a, b), cand_p, p)
    out_os = jax.tree.map(lambda a, b: jnp.where(fin, a, b), next_os, os_s)
    return out_p, out_os, {**avg_m, "finite": fin.astype(jnp.float32)}

  eff_k = (
    min(steps_per_loop, steps, max(1, eval_every)) if steps_per_loop > 1 else 1
  )
  single_fn = jax.jit(_train_step, donate_argnums=(0, 1))

  def _scan_body(
    carry: tuple[Mapping[str, Any], optax.OptState], batch: _Phase5Batch
  ) -> tuple[tuple[Mapping[str, Any], optax.OptState], dict[str, jnp.ndarray]]:
    np_p, np_os, met = _train_step(carry[0], carry[1], batch)
    return (np_p, np_os), met

  multi_fn = jax.jit(
    lambda p, os_s, chunk: jax.lax.scan(_scan_body, (p, os_s), chunk),
    donate_argnums=(0, 1),
  )
  apply_jit = jax.jit(lambda p, nap, cond_in: model.apply(p, nap, cond=cond_in))

  ckpt_path = out_root / "checkpoint.npz"
  step0_scores = _evaluate_phase5(
    apply_fn=apply_jit,
    params=params,
    eval_clips=eval_clips,
    healthy_cochlea=healthy_dev,
    n3_profile=n3_prof,
    s2_profile=s2_prof,
    n3_cochlea=n3_dev,
    s2_cochlea=s2_dev,
    sample_rate_hz=sample_rate_hz,
    lookahead=lookahead,
    mask_samples=mask_samples,
    out_root=out_root,
    step_idx=0,
    mesh=mesh,
  )
  print(f"[Phase 5 step 0] {json.dumps(step0_scores)}", flush=True)
  best_sel_loss = float(step0_scores["selection_loss"])
  best_scores = step0_scores
  best_params_host = init_params_host
  nap_inverter.save_direct_checkpoint(ckpt_path, best_params_host, cfg)

  print(
    f"[Phase 5] params={n_params:,} fs={sample_rate_hz}Hz T={num_samples} "
    f"B={batch_size} (accum={grad_accum_steps}x{micro_batch_size}) K={eff_k} "
    f"percep=({hasqi_weight}/{haspi_weight}) backend={backend} devices={n_devices}",
    flush=True,
  )

  step = 0
  skipped = 0
  last_metrics: dict[str, float] = {}
  with tpu.ThreadedDevicePrefetcher(
    batch_iter, mesh, steps_per_loop=eff_k, max_prefetch=2
  ) as prefetcher:
    while step < steps:
      rem = steps - step
      next_eval = max(1, eval_every - (step % max(1, eval_every)))
      t0 = time.perf_counter()
      m_dev: dict[str, jnp.ndarray]
      if eff_k > 1 and rem >= eff_k and next_eval >= eff_k:
        chunk = cast(_Phase5Batch, next(prefetcher))
        (params, opt_state), m_dev_raw = multi_fn(params, opt_state, chunk)
        m_dev = cast(dict[str, jnp.ndarray], m_dev_raw)
        c_len = eff_k
      else:
        sh = cast(
          _Phase5Batch, tpu.shard_batch(next(batch_iter), mesh, stacked=False)
        )
        params, opt_state, m_one = single_fn(params, opt_state, sh)
        m_dev = {
          k: v[None] for k, v in cast(dict[str, jnp.ndarray], m_one).items()
        }
        c_len = 1
      step += c_len
      fin_arr = np.asarray(m_dev["finite"])
      if float(np.min(fin_arr)) < 0.5:
        skipped += int(np.sum(fin_arr < 0.5))
        if skipped >= 20:
          raise FloatingPointError(
            f"{skipped} consecutive non-finite steps at step {step}."
          )
      else:
        skipped = 0
      dt = time.perf_counter() - t0
      last_metrics = {k: float(np.asarray(v)[-1]) for k, v in m_dev.items()}
      with log_path.open("a", newline="") as f:
        csv.writer(f).writerow(
          (
            step,
            f"{last_metrics['loss']:.6g}",
            f"{last_metrics['nap_l1']:.6g}",
            f"{last_metrics['hasqi']:.6g}",
            f"{last_metrics['haspi']:.6g}",
            f"{last_metrics['stft']:.6g}",
            f"{last_metrics['ucl']:.6g}",
            f"{float(np.mean(fin_arr)):.3f}",
            f"{dt:.3f}",
          )
        )
      print(
        f"[Phase 5 step {step:4d}/{steps}] loss={last_metrics['loss']:.5g} "
        f"nap={last_metrics['nap_l1']:.5g} q={last_metrics['hasqi']:.3f} "
        f"p={last_metrics['haspi']:.3f} stft={last_metrics['stft']:.4g} "
        f"ucl={last_metrics['ucl']:.3g} ({dt:.2f}s)",
        flush=True,
      )
      if step % max(1, eval_every) == 0 or step == steps:
        sc = _evaluate_phase5(
          apply_fn=apply_jit,
          params=params,
          eval_clips=eval_clips,
          healthy_cochlea=healthy_dev,
          n3_profile=n3_prof,
          s2_profile=s2_prof,
          n3_cochlea=n3_dev,
          s2_cochlea=s2_dev,
          sample_rate_hz=sample_rate_hz,
          lookahead=lookahead,
          mask_samples=mask_samples,
          out_root=out_root,
          step_idx=step,
          mesh=mesh,
        )
        print(f"[Phase 5 eval {step}] {json.dumps(sc)}", flush=True)
        curr_params_host = tpu.to_host_tree(params)
        nap_inverter.save_direct_checkpoint(
          out_root / "checkpoint_latest.npz", curr_params_host, cfg
        )
        if (
          np.isfinite(sc["selection_loss"])
          and sc["selection_loss"] < best_sel_loss
        ):
          best_sel_loss = float(sc["selection_loss"])
          best_scores = sc
          best_params_host = curr_params_host
          nap_inverter.save_direct_checkpoint(ckpt_path, best_params_host, cfg)

  tpu.delete_tree(opt_state)
  gc.collect()
  return {
    "checkpoint_path": str(ckpt_path),
    "best_scores": best_scores,
    "last_metrics": last_metrics,
    "params": int(n_params),
  }


def main(argv: Sequence[str]) -> None:
  """CLI entrypoint for Phase 5 clinical training."""
  del argv
  if not _OUTPUT_DIR.value:
    raise app.UsageError("--output_dir is required.")
  run_phase5_training(
    output_dir=_OUTPUT_DIR.value,
    data_dir=_DATA_DIR.value,
    init_checkpoint=_INIT_CHECKPOINT.value,
    architecture=_ARCHITECTURE.value,
    width=_WIDTH.value,
    num_blocks=_NUM_BLOCKS.value,
    kernel_size=_KERNEL_SIZE.value,
    coarse_dim=_COARSE_DIM.value,
    coarse_hop=_COARSE_HOP.value,
    lookahead=_LOOKAHEAD.value,
    max_log_gain=_MAX_LOG_GAIN.value,
    sample_rate_hz=_SAMPLE_RATE_HZ.value,
    seconds=_SECONDS.value,
    mask_seconds=_MASK_SECONDS.value,
    output_highpass_hz=_OUTPUT_HIGHPASS_HZ.value,
    warmup_weight=_WARMUP_WEIGHT.value,
    lam=_LAM.value,
    stft_weight=_STFT_WEIGHT.value,
    hasqi_weight=_HASQI_WEIGHT.value,
    haspi_weight=_HASPI_WEIGHT.value,
    ucl_weight=_UCL_WEIGHT.value,
    clip_norm=_CLIP_NORM.value,
    lr=_LR.value,
    lr_schedule=_LR_SCHEDULE.value,
    steps=_STEPS.value,
    batch_size=_BATCH_SIZE.value,
    grad_accum_steps=_GRAD_ACCUM_STEPS.value,
    steps_per_loop=_STEPS_PER_LOOP.value,
    share_prescribe=_SHARE_PRESCRIBE.value,
    share_auralize=_SHARE_AURALIZE.value,
    eval_every=_EVAL_EVERY.value,
    eval_clips_count=_EVAL_CLIPS.value,
    score_seconds=_SCORE_SECONDS.value,
    seed=_SEED.value,
    compilation_cache_dir=_COMPILATION_CACHE_DIR.value,
    matmul_precision=_MATMUL_PRECISION.value,
  )


if __name__ == "__main__":
  define_flags()
  app.run(main)
