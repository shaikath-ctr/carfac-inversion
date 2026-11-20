"""Phase 5b TPU training script for direct generative SAI-to-audio inversion.

Streams 1D waveforms on host (``ProcessedClips`` ArrayRecord shards via Grain
or synthetic speech for tests), computes healthy and impaired CARFAC NAPs and
Stabilized Auditory Images (SAIs) on TPU via ``jax.vmap``, trains
:class:`carfac_inversion.sai_inverter.SaiDirectInverter` end-to-end with
phase-invariant multi-scale SAI cycle consistency + healthy MR-STFT, and
optionally fine-tunes with 1D MPD + MRD adversarial and MRD-only feature
matching losses.
"""

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
from hp_acoustic.filters import middle_ear
from hp_acoustic.io import audio_io
from hp_acoustic.models import carfac

from carfac_inversion import arcfac_metrics
from carfac_inversion import neural
from carfac_inversion import sai_inverter
from carfac_inversion import tpu
from carfac_inversion import vocoder_losses
from carfac_inversion.scripts import benchmark_suite
from carfac_inversion.scripts import train_phase4

FLAGS = flags.FLAGS

_OUTPUT_DIR: flags.FlagHolder[str | None]
_DATA_DIR: flags.FlagHolder[str | None]
_POOL: flags.FlagHolder[str | None]
_INIT_CHECKPOINT: flags.FlagHolder[str | None]
_MODEL_DIM: flags.FlagHolder[int]
_STEM_BLOCKS: flags.FlagHolder[int]
_DECODER_BLOCKS: flags.FlagHolder[int]
_SAMPLE_RATE_HZ: flags.FlagHolder[int]
_SECONDS: flags.FlagHolder[float]
_MASK_SECONDS: flags.FlagHolder[float]
_OUTPUT_HIGHPASS_HZ: flags.FlagHolder[float]
_STFT_WEIGHT: flags.FlagHolder[float]
_UCL_WEIGHT: flags.FlagHolder[float]
_CLIP_NORM: flags.FlagHolder[float]
_LR: flags.FlagHolder[float]
_LR_SCHEDULE: flags.FlagHolder[str]
_STEPS: flags.FlagHolder[int]
_HEALTHY_WARMUP_STEPS: flags.FlagHolder[int]
_GAN_STEPS: flags.FlagHolder[int]
_GAN_LR: flags.FlagHolder[float]
_GAN_WEIGHT: flags.FlagHolder[float]
_FM_WEIGHT: flags.FlagHolder[float]
_BATCH_SIZE: flags.FlagHolder[int]
_GRAD_ACCUM_STEPS: flags.FlagHolder[int]
_STEPS_PER_LOOP: flags.FlagHolder[int]
_EARS: flags.FlagHolder[str]
_SHARE_HEALTHY: flags.FlagHolder[float]
_SHARE_FLAT: flags.FlagHolder[float]
_MIDDLE_EAR: flags.FlagHolder[bool]
_EVAL_EVERY: flags.FlagHolder[int]
_EVAL_CLIPS: flags.FlagHolder[int]
_SCORE_SECONDS: flags.FlagHolder[float]
_SEED: flags.FlagHolder[int]
_COMPILATION_CACHE_DIR: flags.FlagHolder[str | None]
_MATMUL_PRECISION: flags.FlagHolder[str]


def define_flags(flag_values: flags.FlagValues = FLAGS) -> None:
  """Registers Phase 5b SAI-to-audio training flags."""
  global _OUTPUT_DIR
  global _DATA_DIR
  global _POOL
  global _INIT_CHECKPOINT
  global _MODEL_DIM
  global _STEM_BLOCKS
  global _DECODER_BLOCKS
  global _SAMPLE_RATE_HZ
  global _SECONDS
  global _MASK_SECONDS
  global _OUTPUT_HIGHPASS_HZ
  global _STFT_WEIGHT
  global _UCL_WEIGHT
  global _CLIP_NORM
  global _LR
  global _LR_SCHEDULE
  global _STEPS
  global _HEALTHY_WARMUP_STEPS
  global _GAN_STEPS
  global _GAN_LR
  global _GAN_WEIGHT
  global _FM_WEIGHT
  global _BATCH_SIZE
  global _GRAD_ACCUM_STEPS
  global _STEPS_PER_LOOP
  global _EARS
  global _SHARE_HEALTHY
  global _SHARE_FLAT
  global _MIDDLE_EAR
  global _EVAL_EVERY
  global _EVAL_CLIPS
  global _SCORE_SECONDS
  global _SEED
  global _COMPILATION_CACHE_DIR
  global _MATMUL_PRECISION

  _OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Directory for config, checkpoints, log.csv, and evaluation scores.",
    required=True,
    flag_values=flag_values,
  )
  _DATA_DIR = flags.DEFINE_string(
    "data_dir",
    None,
    "ProcessedClips ArrayRecord root with train/ and test/ splits.",
    flag_values=flag_values,
  )
  _POOL = flags.DEFINE_string(
    "pool",
    None,
    "Optional .npz listener fit pool holding ohc_health / v_half_shift.",
    flag_values=flag_values,
  )
  _INIT_CHECKPOINT = flags.DEFINE_string(
    "init_checkpoint",
    None,
    "Optional Phase 5b SAI .npz checkpoint for resuming or GAN fine-tuning.",
    flag_values=flag_values,
  )
  _MODEL_DIM = flags.DEFINE_integer(
    "model_dim",
    256,
    "Channel dimension of SaiDirectInverter stem and Vocos decoder.",
    flag_values=flag_values,
  )
  _STEM_BLOCKS = flags.DEFINE_integer(
    "stem_blocks",
    4,
    "Number of causal ConvNeXt blocks in the SAI frame stem.",
    flag_values=flag_values,
  )
  _DECODER_BLOCKS = flags.DEFINE_integer(
    "decoder_blocks",
    8,
    "Number of causal ConvNeXt blocks in the Vocos spectrogram decoder.",
    flag_values=flag_values,
  )
  _SAMPLE_RATE_HZ = flags.DEFINE_integer(
    "sample_rate_hz", 32000, "Sample rate in Hz.", flag_values=flag_values
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
    "Leading CARFAC start-from-rest mask duration in seconds.",
    flag_values=flag_values,
  )
  _OUTPUT_HIGHPASS_HZ = flags.DEFINE_float(
    "output_highpass_hz",
    20.0,
    "Causal 4th-order Butterworth high-pass cutoff in Hz.",
    flag_values=flag_values,
  )
  _STFT_WEIGHT = flags.DEFINE_float(
    "stft_weight",
    0.35,
    "Multi-resolution STFT loss weight on healthy targets.",
    flag_values=flag_values,
  )
  _UCL_WEIGHT = flags.DEFINE_float(
    "ucl_weight",
    1.0,
    "Quadratic full-scale safety penalty above 0.95 amplitude.",
    flag_values=flag_values,
  )
  _CLIP_NORM = flags.DEFINE_float(
    "clip_norm",
    1.0,
    "Global gradient norm clip threshold.",
    flag_values=flag_values,
  )
  _LR = flags.DEFINE_float(
    "lr", 1e-3, "Peak Adam learning rate for Stage 1.", flag_values=flag_values
  )
  _LR_SCHEDULE = flags.DEFINE_enum(
    "lr_schedule",
    "cosine",
    ("constant", "cosine"),
    "Stage 1 LR schedule.",
    flag_values=flag_values,
  )
  _STEPS = flags.DEFINE_integer(
    "steps", 800, "Stage 1 training steps.", flag_values=flag_values
  )
  _HEALTHY_WARMUP_STEPS = flags.DEFINE_integer(
    "healthy_warmup_steps",
    300,
    "Leading Stage 1 steps restricted to healthy target ears.",
    flag_values=flag_values,
  )
  _GAN_STEPS = flags.DEFINE_integer(
    "gan_steps",
    200,
    "Stage 2 1D MPD+MRD adversarial fine-tuning steps.",
    flag_values=flag_values,
  )
  _GAN_LR = flags.DEFINE_float(
    "gan_lr", 1.5e-4, "Stage 2 GAN learning rate.", flag_values=flag_values
  )
  _GAN_WEIGHT = flags.DEFINE_float(
    "gan_weight",
    0.05,
    "Adversarial loss weight in Stage 2.",
    flag_values=flag_values,
  )
  _FM_WEIGHT = flags.DEFINE_float(
    "fm_weight",
    2.0,
    "MRD-only feature matching multiplier in Stage 2 GAN.",
    flag_values=flag_values,
  )
  _BATCH_SIZE = flags.DEFINE_integer(
    "batch_size", 64, "Global training batch size.", flag_values=flag_values
  )
  _GRAD_ACCUM_STEPS = flags.DEFINE_integer(
    "grad_accum_steps",
    2,
    "Microbatch gradient accumulation steps.",
    flag_values=flag_values,
  )
  _STEPS_PER_LOOP = flags.DEFINE_integer(
    "steps_per_loop",
    25,
    "Optimizer steps fused per device jax.lax.scan dispatch.",
    flag_values=flag_values,
  )
  _EARS = flags.DEFINE_enum(
    "ears",
    "mixture",
    ("healthy", "mixture"),
    "Target listener distribution during training.",
    flag_values=flag_values,
  )
  _SHARE_HEALTHY = flags.DEFINE_float(
    "share_healthy",
    0.50,
    "Fraction of healthy target ears in mixture mode.",
    flag_values=flag_values,
  )
  _SHARE_FLAT = flags.DEFINE_float(
    "share_flat",
    0.25,
    "Fraction of flat OHC targets in mixture mode.",
    flag_values=flag_values,
  )
  _MIDDLE_EAR = flags.DEFINE_boolean(
    "middle_ear",
    False,
    "Enable middle-ear filter symmetrically on target and healthy ears.",
    flag_values=flag_values,
  )
  _EVAL_EVERY = flags.DEFINE_integer(
    "eval_every", 200, "Evaluation cadence in steps.", flag_values=flag_values
  )
  _EVAL_CLIPS = flags.DEFINE_integer(
    "eval_clips",
    2,
    "Number of held-out evaluation clips.",
    flag_values=flag_values,
  )
  _SCORE_SECONDS = flags.DEFINE_float(
    "score_seconds",
    2.0,
    "Clip length in seconds during held-out evaluation.",
    flag_values=flag_values,
  )
  _SEED = flags.DEFINE_integer("seed", 0, "PRNG seed.", flag_values=flag_values)
  _COMPILATION_CACHE_DIR = flags.DEFINE_string(
    "compilation_cache_dir",
    None,
    "Optional persistent XLA compilation cache path.",
    flag_values=flag_values,
  )
  _MATMUL_PRECISION = flags.DEFINE_enum(
    "matmul_precision",
    "high",
    ("default", "high", "highest", "float32"),
    "JAX default matmul precision policy.",
    flag_values=flag_values,
  )


def _iter_sai_batches(
  *,
  wave_iter: Iterator[np.ndarray],
  rng: np.random.Generator,
  num_channels: int,
  cfs: np.ndarray,
  ears_mode: str,
  share_healthy: float,
  share_flat: float,
  pool_ohc: np.ndarray | None,
  pool_shift: np.ndarray | None,
  stft_weight: float,
  healthy_warmup_steps: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
  """Pairs 1D audio waveforms with ear profiles and healthy STFT weights."""
  for batch_idx, wav in enumerate(wave_iter):
    bsz = wav.shape[0]
    eff_ears = "healthy" if batch_idx < healthy_warmup_steps else ears_mode
    ohc, shift, is_healthy = train_phase4._draw_listener_batch(
      rng,
      bsz,
      num_channels,
      cfs,
      ears_mode=eff_ears,
      share_healthy=share_healthy,
      share_flat=share_flat,
      pool_ohc=pool_ohc,
      pool_shift=pool_shift,
    )
    stft_b = np.where(is_healthy, stft_weight, 0.0).astype(np.float32)
    yield wav, ohc, shift, stft_b


def _evaluate_sai_inverter(
  *,
  apply_fn: Any,
  params: Mapping[str, Any],
  eval_clips: Sequence[tuple[str, np.ndarray]],
  healthy_cochlea: carfac.CarfacCochleaModel,
  impaired_cochlea: carfac.CarfacCochleaModel,
  sample_rate_hz: int,
  mask_samples: int,
  out_root: pathlib.Path,
  step_idx: int,
  mesh: jax.sharding.Mesh | None = None,
) -> dict[str, Any]:
  """Evaluates ``SaiDirectInverter`` on ``healthy`` and ``flat:0.5`` targets."""
  forward_jit = jax.jit(lambda w, cm: cm(w[:, None])[:, :, 0])
  sai_jit = jax.jit(
    lambda w, cm: sai_inverter.sai_from_naps_f32(cm(w[:, None])[:, :, 0][None])
  )
  hl_arr = benchmark_suite.HALF_OHC_HEARING_LOSS_DB
  targets = (
    ("healthy", healthy_cochlea, np.zeros(6, dtype=np.float64)),
    ("flat-0.5", impaired_cochlea, hl_arr),
  )

  def _place(arr: np.ndarray) -> jnp.ndarray:
    arr_j = jnp.asarray(arr, dtype=jnp.float32)
    return tpu.replicate_tree(arr_j, mesh) if mesh is not None else arr_j

  scores: dict[str, Any] = {"step": step_idx, "entries": {}}
  sai_rt_vals: list[float] = []
  sai_kept_vals: list[float] = []
  sai_dir_vals: list[float] = []
  stft_vals: list[float] = []
  hasqi_vals: list[float] = []

  render_dir = out_root / "renders" / f"step{step_idx}"
  render_dir.mkdir(parents=True, exist_ok=True)
  scores_dir = out_root / "scores"
  scores_dir.mkdir(parents=True, exist_ok=True)

  for clip_id, raw_x in eval_clips:
    x = benchmark_suite._scale_to_spl(raw_x, 65.0)
    x_dev = _place(x)
    n_hx = np.asarray(forward_jit(x_dev, healthy_cochlea), dtype=np.float32)
    s_hx = np.asarray(sai_jit(x_dev, healthy_cochlea), dtype=np.float32)
    for target_name, tgt_cm, aud_hl in targets:
      if target_name == "healthy":
        n_tgt = n_hx
        s_tgt = s_hx
      else:
        n_tgt = np.asarray(forward_jit(x_dev, tgt_cm), dtype=np.float32)
        s_tgt = np.asarray(sai_jit(x_dev, tgt_cm), dtype=np.float32)

      y = np.asarray(apply_fn(params, _place(s_tgt))[0], dtype=np.float32)
      n_hy = np.asarray(
        forward_jit(_place(y), healthy_cochlea), dtype=np.float32
      )
      sc = arcfac_metrics.score_arcfac(
        x,
        y,
        n_target=n_tgt,
        n_healthy_ref=n_hx,
        n_healthy_est=n_hy,
        sample_rate_hz=sample_rate_hz,
        delay=0,
        mask_samples=min(mask_samples, len(x) // 4),
      )
      metrics = benchmark_suite._score_pair(
        ref=x,
        est=y,
        target_nap=n_tgt,
        recon_nap=n_hy,
        healthy_ref_nap=n_hx,
        sample_rate_hz=sample_rate_hz,
        hearing_loss_db=aud_hl,
      )
      scores["entries"][f"{target_name}/{clip_id}"] = {
        **sc,
        "si_sdr": metrics["si_sdr"],
        "hasqi": metrics["hasqi"],
      }
      if target_name == "healthy":
        sai_rt_vals.append(float(sc["sai"]["round_trip_rel"]))
        stft_vals.append(float(sc["log_stft_to_x"]))
        hasqi_vals.append(float(metrics["hasqi"]))
      else:
        sai_kept_vals.append(float(sc["sai"].get("kept", float("nan"))))
        sai_dir_vals.append(float(sc["sai"].get("direction", float("nan"))))

      out_wav = render_dir / f"{clip_id}-{target_name}.wav"
      audio_io.save_audio_to_file(y, str(out_wav), sample_rate_hz)

  m_sai_rt = float(np.nanmean(sai_rt_vals)) if sai_rt_vals else float("nan")
  m_sai_kept = (
    float(np.nanmean(sai_kept_vals)) if sai_kept_vals else float("nan")
  )
  m_sai_dir = float(np.nanmean(sai_dir_vals)) if sai_dir_vals else float("nan")
  m_stft = float(np.nanmean(stft_vals)) if stft_vals else float("nan")
  m_hasqi = float(np.nanmean(hasqi_vals)) if hasqi_vals else float("nan")
  sel_loss = m_sai_rt + 0.5 * m_sai_kept + 0.3 * (1.0 - m_hasqi) + 0.1 * m_stft
  scores["summary"] = {
    "sai_round_trip_rel": m_sai_rt,
    "sai_kept": m_sai_kept,
    "sai_direction": m_sai_dir,
    "log_stft_to_x": m_stft,
    "hasqi": m_hasqi,
    "selection_loss": sel_loss,
  }
  with (scores_dir / f"step{step_idx}.json").open("w") as f:
    json.dump(scores, f, indent=1, default=str)
  return scores["summary"]


def run_sai_training(
  *,
  output_dir: str,
  data_dir: str | None = None,
  pool_path: str | None = None,
  init_checkpoint: str | None = None,
  model_dim: int = 256,
  stem_blocks: int = 4,
  decoder_blocks: int = 8,
  sample_rate_hz: int = 32000,
  seconds: float = 1.0,
  mask_seconds: float = 0.25,
  output_highpass_hz: float = 20.0,
  stft_weight: float = 0.35,
  ucl_weight: float = 1.0,
  clip_norm: float = 1.0,
  lr: float = 1e-3,
  lr_schedule: str = "cosine",
  steps: int = 800,
  healthy_warmup_steps: int = 300,
  gan_steps: int = 200,
  gan_lr: float = 1.5e-4,
  gan_weight: float = 0.05,
  fm_weight: float = 2.0,
  batch_size: int = 64,
  grad_accum_steps: int = 2,
  steps_per_loop: int = 25,
  ears: str = "mixture",
  share_healthy: float = 0.50,
  share_flat: float = 0.25,
  middle_ear_enabled: bool = False,
  eval_every: int = 200,
  eval_clips_count: int = 2,
  score_seconds: float = 2.0,
  seed: int = 0,
  compilation_cache_dir: str | None = None,
  matmul_precision: str = "high",
) -> dict[str, Any]:
  """Trains :class:`SaiDirectInverter` on TPU with SAI cycle + 1D MRD/MPD GAN."""
  if steps < 1 and gan_steps < 1:
    raise ValueError("At least one of steps or gan_steps must be >= 1.")
  if steps_per_loop < 1:
    raise ValueError(f"steps_per_loop must be >= 1, got {steps_per_loop}.")

  backend, n_devices = tpu.configure_runtime(
    compilation_cache_dir=compilation_cache_dir,
    matmul_precision=matmul_precision,
  )
  mesh = tpu.make_data_mesh()
  n_local = max(1, len(mesh.local_devices))
  if (
    grad_accum_steps < 1 or batch_size < 1 or batch_size % grad_accum_steps != 0
  ):
    raise ValueError(
      f"batch_size={batch_size} must be divisible by grad_accum_steps={grad_accum_steps}."
    )
  micro_batch_size = batch_size // grad_accum_steps
  if micro_batch_size % n_local != 0:
    raise ValueError(
      f"micro_batch_size={micro_batch_size} must be a positive multiple of {n_local}."
    )

  out_root = pathlib.Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)
  num_samples = max(2048, int(round(seconds * sample_rate_hz)))
  mask_samples = int(round(mask_seconds * sample_rate_hz))
  me_cfg = middle_ear.MiddleEarFilterConfig(enabled=middle_ear_enabled)

  with tpu.host_cpu_context():
    healthy_raw = carfac.build(fs=sample_rate_hz, middle_ear_filter=me_cfg)
    impaired_raw = carfac.build(
      fs=sample_rate_hz, ohc_health=0.5, middle_ear_filter=me_cfg
    )
    cfs = np.asarray(healthy_raw.channel_frequencies, dtype=np.float32)
    num_channels = int(healthy_raw.num_channels)
    healthy_host = neural.cochlea_to_host(healthy_raw)
    impaired_host = neural.cochlea_to_host(impaired_raw)
  healthy_dev = neural.replicate_cochlea(healthy_host, mesh)
  impaired_dev = neural.replicate_cochlea(impaired_host, mesh)

  pool_ohc, pool_shift = train_phase4._load_listener_pool(
    pool_path, num_channels
  )
  if data_dir:
    wave_iter = train_phase4._build_grain_waveform_iter(
      data_dir=data_dir,
      sample_rate_hz=sample_rate_hz,
      seconds=float(num_samples) / float(sample_rate_hz),
      batch_size=batch_size,
      seed=seed,
      level_db_spl=(20.0, 95.0),
    )
    eval_clips = benchmark_suite.load_speech_clips(
      data_dir,
      sample_rate_hz=sample_rate_hz,
      max_clips=eval_clips_count,
      clip_seconds=score_seconds,
      split="test",
    )
  else:
    wave_iter = train_phase4._build_synthetic_waveform_iter(
      sample_rate_hz=sample_rate_hz,
      num_samples=num_samples,
      batch_size=batch_size,
      seed=seed,
    )
    eval_clips = benchmark_suite.load_speech_clips(
      None,
      sample_rate_hz=sample_rate_hz,
      max_clips=min(eval_clips_count, 2),
      clip_seconds=score_seconds,
      split="test",
    )

  healthy_ear_fn = jax.jit(jax.vmap(lambda w: healthy_dev(w[:, None])[:, :, 0]))
  raw_calib = (
    np.asarray(eval_clips[0][1], dtype=np.float32)
    if eval_clips
    else np.asarray(next(wave_iter)[0], dtype=np.float32)
  )
  if float(np.sqrt(np.mean(raw_calib**2))) < 1e-6:
    raw_calib = np.asarray(next(wave_iter)[0], dtype=np.float32)
  if len(raw_calib) >= num_samples:
    calib_1d = raw_calib[:num_samples]
  else:
    reps = (num_samples + len(raw_calib) - 1) // max(1, len(raw_calib))
    calib_1d = np.tile(raw_calib, reps)[:num_samples]
  calib_65_1d = benchmark_suite._scale_to_spl(calib_1d, 65.0)
  ref_65 = np.broadcast_to(calib_65_1d[None, :], (n_local, num_samples)).copy()
  calib_nap = healthy_ear_fn(tpu.shard_batch(ref_65, mesh, stacked=False))
  calib_sai = sai_inverter.sai_from_naps_f32(calib_nap)
  amp_scale = max(float(jnp.sqrt(jnp.mean(jnp.maximum(calib_sai, 0.0)))), 1e-6)
  output_scale = max(float(np.sqrt(np.mean(ref_65**2))), 1e-6)

  if init_checkpoint:
    model, init_params_raw, cfg = sai_inverter.load_sai_checkpoint(
      init_checkpoint
    )
    init_params_host = tpu.to_host_tree(init_params_raw)
  else:
    cfg = sai_inverter.SaiInverterConfig(
      sample_rate_hz=float(sample_rate_hz),
      model_dim=model_dim,
      stem_blocks=stem_blocks,
      decoder_blocks=decoder_blocks,
      amp_scale=amp_scale,
      output_scale=output_scale,
      output_highpass_hz=output_highpass_hz,
    )
    model = sai_inverter.SaiDirectInverter(config=cfg)
    with tpu.host_cpu_context():
      dummy_sai = jnp.zeros(
        (
          1,
          max(1, int(calib_sai.shape[1])),
          int(calib_sai.shape[2]),
          num_channels,
        ),
        dtype=jnp.float32,
      )
      raw_p = model.init(
        jax.random.PRNGKey(seed),
        dummy_sai,
        target_samples=num_samples,
        noise_key=jax.random.PRNGKey(seed + 1),
      )
      init_params_host = tpu.to_host_tree(raw_p)

  params = cast(Mapping[str, Any], tpu.replicate_tree(init_params_host, mesh))
  n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

  target_naps_jit = jax.jit(
    jax.vmap(
      lambda w, o, s: train_phase4._apply_ear_with_profile(
        healthy_dev, w, o, s
      ),
      in_axes=(0, 0, 0),
    )
  )
  stft_res = vocoder_losses.stft_resolutions_for_rate(float(sample_rate_hz))
  first_frame = min(
    arcfac_metrics.first_unmasked_frame(mask_samples),
    max(0, int(calib_sai.shape[1]) - 1),
  )

  with (out_root / "config.json").open("w") as f:
    json.dump(
      {
        "cfg": dataclasses.asdict(cfg),
        "params": int(n_params),
        "sample_rate_hz": sample_rate_hz,
        "seconds": seconds,
        "mask_samples": mask_samples,
        "stft_weight": stft_weight,
        "ucl_weight": ucl_weight,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "steps": steps,
        "gan_steps": gan_steps,
        "gan_lr": gan_lr,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "micro_batch_size": micro_batch_size,
        "ears": ears,
        "share_healthy": share_healthy,
        "share_flat": share_flat,
        "backend": backend,
        "n_devices": n_devices,
      },
      f,
      indent=1,
      default=str,
    )

  print(
    f"[Phase 5b SAI] params={n_params:,} fs={sample_rate_hz}Hz T={num_samples} "
    f"F_sai={int(calib_sai.shape[1])} mask_frame={first_frame} B={batch_size} "
    f"(accum={grad_accum_steps}x{micro_batch_size}) K={steps_per_loop} "
    f"backend={backend} devices={n_devices}",
    flush=True,
  )

  def _sai_loss_core(
    p: Mapping[str, Any],
    x_b: jnp.ndarray,
    tgt_sai: jnp.ndarray,
    stft_b: jnp.ndarray,
    noise_key: jax.Array,
  ) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    y_b = cast(
      jnp.ndarray,
      model.apply(p, tgt_sai, target_samples=num_samples, noise_key=noise_key),
    )
    hy_nap = healthy_ear_fn(y_b)
    hy_sai = sai_inverter.sai_from_naps_f32(hy_nap)
    sqrt_hy = jnp.sqrt(jnp.maximum(hy_sai, 0.0) + 1e-10)
    sqrt_tg = jnp.sqrt(jnp.maximum(tgt_sai, 0.0) + 1e-10)
    sqrt_diff = jnp.abs(sqrt_hy - sqrt_tg)
    raw_diff = jnp.abs(hy_sai - tgt_sai)

    sqrt_steady = jnp.mean(sqrt_diff[:, first_frame:], axis=(1, 2, 3))
    raw_steady = jnp.mean(raw_diff[:, first_frame:], axis=(1, 2, 3))
    if first_frame > 0:
      sqrt_onset = jnp.mean(sqrt_diff[:, :first_frame], axis=(1, 2, 3))
    else:
      sqrt_onset = jnp.zeros_like(sqrt_steady)

    raw_denom = jnp.maximum(
      jnp.mean(jnp.abs(tgt_sai[:, first_frame:]), axis=(1, 2, 3)),
      max(0.01 * (amp_scale**2), 1e-6),
    )
    sqrt_denom = jnp.maximum(
      jnp.mean(sqrt_tg[:, first_frame:], axis=(1, 2, 3)),
      max(0.05 * amp_scale, 1e-4),
    )
    rel_raw = raw_steady / raw_denom
    rel_sqrt = (sqrt_steady + 0.25 * sqrt_onset) / sqrt_denom

    lag_center = tgt_sai.shape[2] // 2
    sq_hy = sqrt_hy[:, first_frame:]
    sq_tg = sqrt_tg[:, first_frame:]
    lag0_denom = jnp.maximum(
      jnp.mean(sq_tg[:, :, lag_center, :], axis=(1, 2)),
      max(0.05 * amp_scale, 1e-4),
    )
    lag0_loss = (
      jnp.mean(
        jnp.abs(sq_hy[:, :, lag_center, :] - sq_tg[:, :, lag_center, :]),
        axis=(1, 2),
      )
      / lag0_denom
    )
    l2_sqrt = (
      jnp.sqrt(jnp.mean((sq_hy - sq_tg) ** 2, axis=(1, 2, 3)) + 1e-12)
      / sqrt_denom
    )
    p_len = (sq_hy.shape[2] // 8) * 8
    p8_hy = (
      sq_hy[:, :, :p_len, :]
      .reshape((sq_hy.shape[0], sq_hy.shape[1], p_len // 8, 8, sq_hy.shape[3]))
      .mean(axis=3)
    )
    p8_tg = (
      sq_tg[:, :, :p_len, :]
      .reshape((sq_tg.shape[0], sq_tg.shape[1], p_len // 8, 8, sq_tg.shape[3]))
      .mean(axis=3)
    )
    pool8_sai = jnp.mean(jnp.abs(p8_hy - p8_tg), axis=(1, 2, 3)) / sqrt_denom
    sai_obj = 1.2 * lag0_loss + 0.8 * l2_sqrt + 0.4 * pool8_sai + 0.4 * rel_sqrt

    ucl_per_clip = jnp.mean(jax.nn.relu(jnp.abs(y_b) - 0.95) ** 2, axis=1)
    total_per_clip = sai_obj + float(ucl_weight) * ucl_per_clip
    if stft_weight > 0.0:

      def _norm_stft(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        c_rms = jnp.maximum(jnp.sqrt(jnp.mean(b**2)), 1e-4)
        return vocoder_losses.multi_resolution_stft_loss(
          (a / c_rms)[None, :], (b / c_rms)[None, :], resolutions=stft_res
        )[0]

      stft_per_clip = jax.vmap(_norm_stft)(y_b, x_b)
      total_per_clip = total_per_clip + stft_b * stft_per_clip
      active = stft_b > 0.0
      stft_mean = jnp.sum(jnp.where(active, stft_per_clip, 0.0)) / jnp.maximum(
        jnp.sum(active), 1.0
      )
    else:
      stft_mean = jnp.float32(0.0)

    total = jnp.mean(total_per_clip)
    return (
      cast(jnp.ndarray, total),
      y_b,
      {
        "loss": cast(jnp.ndarray, total),
        "sqrt_sai": cast(jnp.ndarray, jnp.mean(rel_sqrt)),
        "raw_sai": cast(jnp.ndarray, jnp.mean(rel_raw)),
        "stft": stft_mean,
        "ucl": cast(jnp.ndarray, jnp.mean(ucl_per_clip)),
      },
    )

  if lr_schedule == "cosine" and steps > 1:
    sched: optax.Schedule | float = optax.warmup_cosine_decay_schedule(
      init_value=lr * 0.1,
      peak_value=lr,
      warmup_steps=min(150, max(1, steps // 10)),
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
    rng_host = np.asarray(jax.random.PRNGKey(seed + 101))
  opt_carry: tuple[optax.OptState, jax.Array] = cast(
    tuple[optax.OptState, jax.Array],
    tpu.replicate_tree((opt_state_host, rng_host), mesh),
  )

  def _step_loss(
    p: Mapping[str, Any],
    x_b: jnp.ndarray,
    tgt_sai: jnp.ndarray,
    stft_b: jnp.ndarray,
    noise_key: jax.Array,
  ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    val, _, aux = _sai_loss_core(p, x_b, tgt_sai, stft_b, noise_key)
    return val, aux

  def _train_step_impl(
    p: Mapping[str, Any],
    carry_os_rng: tuple[optax.OptState, jax.Array],
    batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
  ) -> tuple[
    Mapping[str, Any],
    tuple[optax.OptState, jax.Array],
    dict[str, jnp.ndarray],
  ]:
    os_s, rng_key = carry_os_rng
    keys = jax.random.split(rng_key, grad_accum_steps + 1)
    next_rng = keys[0]
    micro_keys = keys[1:]
    micros = train_phase4._split_microbatches(batch, grad_accum_steps)
    zero_grads = jax.tree.map(jnp.zeros_like, p)
    zero_metrics = {
      name: jnp.zeros((), dtype=jnp.float32)
      for name in ("loss", "sqrt_sai", "raw_sai", "stft", "ucl")
    }

    def _accum_step(
      carry: tuple[Mapping[str, Any], dict[str, jnp.ndarray]],
      item: tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jax.Array
      ],
    ) -> tuple[tuple[Mapping[str, Any], dict[str, jnp.ndarray]], None]:
      g_acc, m_acc = carry
      (x_b, ohc_b, shift_b, stft_b), m_key = item
      tgt_nap = jax.lax.stop_gradient(target_naps_jit(x_b, ohc_b, shift_b))
      tgt_sai = jax.lax.stop_gradient(sai_inverter.sai_from_naps_f32(tgt_nap))
      (val, parts), grads = jax.value_and_grad(_step_loss, has_aux=True)(
        p, x_b, tgt_sai, stft_b, m_key
      )
      del val
      return (
        jax.tree.map(jnp.add, g_acc, grads),
        {k: m_acc[k] + parts[k] for k in m_acc},
      ), None

    (grads_sum, metrics_sum), _ = jax.lax.scan(
      _accum_step, (zero_grads, zero_metrics), (micros, micro_keys)
    )
    inv_accum = jnp.float32(1.0 / float(grad_accum_steps))
    grads = jax.tree.map(lambda g: g * inv_accum, grads_sum)
    avg_metrics = {k: v * inv_accum for k, v in metrics_sum.items()}
    updates, new_os = tx.update(grads, os_s, p)
    new_p = cast(Mapping[str, Any], optax.apply_updates(p, updates))
    finite = jnp.isfinite(avg_metrics["loss"]) & jnp.all(
      jnp.stack([jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads)])
    )
    kept_p = jax.tree.map(lambda a, b: jnp.where(finite, a, b), new_p, p)
    kept_os = jax.tree.map(lambda a, b: jnp.where(finite, a, b), new_os, os_s)
    return (
      kept_p,
      (kept_os, next_rng),
      {**avg_metrics, "finite": finite.astype(jnp.float32)},
    )

  single_step_fn, multi_step_fn = tpu.make_multi_step_fns(
    _train_step_impl, donate_argnums=(0, 1)
  )

  log_path = out_root / "log.csv"
  with log_path.open("w", newline="") as f:
    csv.writer(f).writerow(
      ("step", "loss", "sqrt_sai", "raw_sai", "stft", "ucl", "finite", "wall_s")
    )

  batch_iter = _iter_sai_batches(
    wave_iter=wave_iter,
    rng=np.random.default_rng(seed + 1),
    num_channels=num_channels,
    cfs=cfs,
    ears_mode=ears,
    share_healthy=share_healthy,
    share_flat=share_flat,
    pool_ohc=pool_ohc,
    pool_shift=pool_shift,
    stft_weight=stft_weight,
    healthy_warmup_steps=healthy_warmup_steps,
  )

  ckpt_path = out_root / "checkpoint.npz"
  best_params_host = init_params_host

  eval_target_len = len(eval_clips[0][1]) if eval_clips else num_samples
  eval_apply_jit = jax.jit(
    lambda p, s: model.apply(p, s, target_samples=eval_target_len)
  )

  step0_scores = _evaluate_sai_inverter(
    apply_fn=eval_apply_jit,
    params=params,
    eval_clips=eval_clips,
    healthy_cochlea=healthy_dev,
    impaired_cochlea=impaired_dev,
    sample_rate_hz=sample_rate_hz,
    mask_samples=mask_samples,
    out_root=out_root,
    step_idx=0,
    mesh=mesh,
  )
  print(f"[Phase 5b step 0] {json.dumps(step0_scores)}", flush=True)
  best_scores = step0_scores
  best_sel_loss = float(step0_scores["selection_loss"])
  sai_inverter.save_sai_checkpoint(ckpt_path, best_params_host, cfg)

  step = 0
  skipped_streak = 0
  last_metrics: dict[str, float] = {}
  eff_k = (
    min(steps_per_loop, steps, max(1, eval_every)) if steps_per_loop > 1 else 1
  )
  if steps > 0:
    with tpu.ThreadedDevicePrefetcher(
      batch_iter, mesh, steps_per_loop=eff_k, max_prefetch=2
    ) as prefetcher:
      while step < steps:
        rem = steps - step
        next_eval_dist = max(1, eval_every - (step % max(1, eval_every)))
        t0 = time.perf_counter()
        if eff_k > 1 and rem >= eff_k and next_eval_dist >= eff_k:
          chunk = cast(
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            next(prefetcher),
          )
          params, opt_carry, m_dev = multi_step_fn(params, opt_carry, chunk)
          c_len = eff_k
        else:
          sh = cast(
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            tpu.shard_batch(next(batch_iter), mesh, stacked=False),
          )
          params, opt_carry, m_dev = single_step_fn(params, opt_carry, sh)
          c_len = 1

        step += c_len
        fin_arr = np.asarray(m_dev["finite"])
        if float(np.min(fin_arr)) < 0.5:
          skipped_streak += int(np.sum(fin_arr < 0.5))
          if skipped_streak >= 20:
            raise FloatingPointError(
              f"{skipped_streak} consecutive non-finite steps at step {step}."
            )
        else:
          skipped_streak = 0
        dt = time.perf_counter() - t0
        last_metrics = {k: float(np.asarray(v)[-1]) for k, v in m_dev.items()}
        with log_path.open("a", newline="") as f:
          csv.writer(f).writerow(
            (
              step,
              f"{last_metrics['loss']:.6g}",
              f"{last_metrics['sqrt_sai']:.6g}",
              f"{last_metrics['raw_sai']:.6g}",
              f"{last_metrics['stft']:.6g}",
              f"{last_metrics['ucl']:.6g}",
              f"{float(np.mean(fin_arr)):.3f}",
              f"{dt:.3f}",
            )
          )
        print(
          f"[Phase 5b step {step:4d}/{steps}] loss={last_metrics['loss']:.5g} "
          f"sqrt_sai={last_metrics['sqrt_sai']:.5g} raw={last_metrics['raw_sai']:.5g} "
          f"stft={last_metrics['stft']:.4g} ucl={last_metrics['ucl']:.3g} ({dt:.2f}s)",
          flush=True,
        )
        if step % max(1, eval_every) == 0 or step == steps:
          sc = _evaluate_sai_inverter(
            apply_fn=eval_apply_jit,
            params=params,
            eval_clips=eval_clips,
            healthy_cochlea=healthy_dev,
            impaired_cochlea=impaired_dev,
            sample_rate_hz=sample_rate_hz,
            mask_samples=mask_samples,
            out_root=out_root,
            step_idx=step,
            mesh=mesh,
          )
          print(f"[Phase 5b eval {step}] {json.dumps(sc)}", flush=True)
          curr_host = tpu.to_host_tree(params)
          sai_inverter.save_sai_checkpoint(
            out_root / "checkpoint_latest.npz", curr_host, cfg
          )
          if (
            np.isfinite(sc["selection_loss"])
            and sc["selection_loss"] < best_sel_loss
          ):
            best_sel_loss = float(sc["selection_loss"])
            best_scores = sc
            best_params_host = curr_host
            sai_inverter.save_sai_checkpoint(ckpt_path, best_params_host, cfg)

    tpu.delete_tree(opt_carry)
    gc.collect()

  if gan_steps > 0:
    stage1_ckpt = out_root / "checkpoint_stage1.npz"
    sai_inverter.save_sai_checkpoint(stage1_ckpt, best_params_host, cfg)
    params = cast(Mapping[str, Any], tpu.replicate_tree(best_params_host, mesh))
    mpd = vocoder_losses.MultiPeriodDiscriminator()
    mrd = vocoder_losses.MultiResolutionDiscriminator(resolutions=stft_res)
    gan_start = mask_samples if num_samples - mask_samples >= 4096 else 0
    with tpu.host_cpu_context():
      dk1, dk2 = jax.random.split(jax.random.PRNGKey(seed + 199))
      dummy_wav = jnp.zeros(
        (1, max(2048, num_samples - gan_start)), dtype=jnp.float32
      )
      d_params_host = tpu.to_host_tree(
        {"mpd": mpd.init(dk1, dummy_wav), "mrd": mrd.init(dk2, dummy_wav)}
      )
    d_params = cast(Mapping[str, Any], tpu.replicate_tree(d_params_host, mesh))
    warmup_g = min(50, max(1, gan_steps // 4))
    g_sched: optax.Schedule | float = (
      optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=gan_lr,
        warmup_steps=warmup_g,
        decay_steps=max(warmup_g + 1, gan_steps),
        end_value=gan_lr * 0.1,
      )
      if gan_steps > 1
      else gan_lr
    )
    tx_g = optax.chain(
      optax.clip_by_global_norm(max(clip_norm, 1.0)),
      optax.adam(g_sched, b1=0.5, b2=0.9),
    )
    tx_d = optax.chain(
      optax.clip_by_global_norm(max(clip_norm, 1.0)),
      optax.adam(gan_lr, b1=0.5, b2=0.9),
    )
    with tpu.host_cpu_context():
      g_opt_h = tpu.to_host_tree(tx_g.init(best_params_host))
      d_opt_h = tpu.to_host_tree(tx_d.init(d_params_host))
      gan_rng_h = np.asarray(jax.random.PRNGKey(seed + 299))
    g_opt = tpu.replicate_tree(g_opt_h, mesh)
    d_opt = tpu.replicate_tree(d_opt_h, mesh)
    gan_rng = cast(jax.Array, tpu.replicate_tree(gan_rng_h, mesh))

    def _gan_step(
      carry_state: tuple[
        Mapping[str, Any],
        Mapping[str, Any],
        optax.OptState,
        optax.OptState,
        jax.Array,
      ],
      batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[
      tuple[
        Mapping[str, Any],
        Mapping[str, Any],
        optax.OptState,
        optax.OptState,
        jax.Array,
      ],
      dict[str, jnp.ndarray],
    ]:
      gp, dp, gos, dos, rkey = carry_state
      rkeys = jax.random.split(rkey, grad_accum_steps + 1)
      next_rkey = rkeys[0]
      micro_rkeys = rkeys[1:]
      micros = train_phase4._split_microbatches(batch, grad_accum_steps)
      inv_accum = jnp.float32(1.0 / float(grad_accum_steps))

      def _d_loss_micro(
        d_p: Mapping[str, Any],
        x_b: jnp.ndarray,
        y_fake: jnp.ndarray,
        clip_norm_rms: jnp.ndarray,
        healthy_mask: jnp.ndarray,
      ) -> jnp.ndarray:
        real_seg = x_b[:, gan_start:] / clip_norm_rms
        fake_seg = y_fake[:, gan_start:] / clip_norm_rms
        return vocoder_losses.discriminator_loss(
          cast(Any, mpd.apply(d_p["mpd"], real_seg)),
          cast(Any, mpd.apply(d_p["mpd"], fake_seg)),
          mask=healthy_mask,
        ) + vocoder_losses.discriminator_loss(
          cast(Any, mrd.apply(d_p["mrd"], real_seg)),
          cast(Any, mrd.apply(d_p["mrd"], fake_seg)),
          mask=healthy_mask,
        )

      def _d_accum_body(
        carry_d: tuple[jnp.ndarray, Mapping[str, Any]],
        item: tuple[
          tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jax.Array
        ],
      ) -> tuple[
        tuple[jnp.ndarray, Mapping[str, Any]],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
      ]:
        d_sum, dg_sum = carry_d
        (x_b, ohc_b, shift_b, _), m_key = item
        healthy_mask = (jnp.min(ohc_b, axis=1) > 0.999).astype(jnp.float32)
        tgt_nap = jax.lax.stop_gradient(target_naps_jit(x_b, ohc_b, shift_b))
        tgt_sai = jax.lax.stop_gradient(sai_inverter.sai_from_naps_f32(tgt_nap))
        clip_norm_rms = jnp.maximum(
          jnp.sqrt(jnp.mean(x_b[:, gan_start:] ** 2, axis=1, keepdims=True)),
          0.1 * output_scale,
        )
        y_fake = jax.lax.stop_gradient(
          cast(
            jnp.ndarray,
            model.apply(
              gp, tgt_sai, target_samples=num_samples, noise_key=m_key
            ),
          )
        )
        d_val_i, d_grad_i = jax.value_and_grad(_d_loss_micro)(
          dp, x_b, y_fake, clip_norm_rms, healthy_mask
        )
        return (
          d_sum + d_val_i,
          jax.tree.map(jnp.add, dg_sum, d_grad_i),
        ), (tgt_sai, clip_norm_rms, healthy_mask)

      zero_d_grads = jax.tree.map(jnp.zeros_like, dp)
      (d_val_sum, d_grads_sum), cached = jax.lax.scan(
        _d_accum_body,
        (jnp.zeros((), dtype=jnp.float32), zero_d_grads),
        (micros, micro_rkeys),
      )
      d_val = d_val_sum * inv_accum
      d_grads = jax.tree.map(lambda g: g * inv_accum, d_grads_sum)
      d_upd, new_dos = tx_d.update(d_grads, dos, dp)
      new_dp = cast(Mapping[str, Any], optax.apply_updates(dp, d_upd))

      def _g_loss_micro(
        g_p: Mapping[str, Any],
        x_b: jnp.ndarray,
        tgt_sai: jnp.ndarray,
        stft_b: jnp.ndarray,
        clip_norm_rms: jnp.ndarray,
        healthy_mask: jnp.ndarray,
        m_key: jax.Array,
      ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        base_loss, y_b, parts = _sai_loss_core(g_p, x_b, tgt_sai, stft_b, m_key)
        real_seg = x_b[:, gan_start:] / clip_norm_rms
        fake_seg = y_b[:, gan_start:] / clip_norm_rms
        fake_m = cast(Any, mpd.apply(new_dp["mpd"], fake_seg))
        real_r = cast(Any, mrd.apply(new_dp["mrd"], real_seg))
        fake_r = cast(Any, mrd.apply(new_dp["mrd"], fake_seg))
        adv = vocoder_losses.generator_loss(
          fake_m, mask=healthy_mask
        ) + vocoder_losses.generator_loss(fake_r, mask=healthy_mask)
        # MRD operates on phase-invariant STFT magnitudes; MPD is phase-sensitive.
        fm = vocoder_losses.feature_matching_loss(
          real_r, fake_r, mask=healthy_mask
        )
        total = base_loss + gan_weight * (adv + fm_weight * fm)
        return total, {**parts, "adv": adv, "fm": fm, "d_loss": d_val}

      zero_g_grads = jax.tree.map(jnp.zeros_like, gp)
      zero_g_metrics = {
        name: jnp.zeros((), dtype=jnp.float32)
        for name in (
          "loss",
          "sqrt_sai",
          "raw_sai",
          "stft",
          "ucl",
          "adv",
          "fm",
          "d_loss",
        )
      }

      def _g_accum_body(
        carry_g: tuple[Mapping[str, Any], dict[str, jnp.ndarray]],
        item: tuple[
          tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
          tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
          jax.Array,
        ],
      ) -> tuple[tuple[Mapping[str, Any], dict[str, jnp.ndarray]], None]:
        gg_acc, gm_acc = carry_g
        (x_b, _, _, stft_b), (tgt_sai, clip_norm_rms, healthy_mask), m_key = (
          item
        )
        (g_val_i, g_parts_i), g_grad_i = jax.value_and_grad(
          _g_loss_micro, has_aux=True
        )(gp, x_b, tgt_sai, stft_b, clip_norm_rms, healthy_mask, m_key)
        step_gm = {"loss": g_val_i, **g_parts_i}
        return (
          jax.tree.map(jnp.add, gg_acc, g_grad_i),
          {k: gm_acc[k] + step_gm[k] for k in gm_acc},
        ), None

      (g_grads_sum, g_metrics_sum), _ = jax.lax.scan(
        _g_accum_body,
        (zero_g_grads, zero_g_metrics),
        (micros, cached, micro_rkeys),
      )
      g_grads = jax.tree.map(lambda g: g * inv_accum, g_grads_sum)
      g_avg_m = {k: v * inv_accum for k, v in g_metrics_sum.items()}
      g_val = g_avg_m["loss"]
      g_upd, new_gos = tx_g.update(g_grads, gos, gp)
      new_gp = cast(Mapping[str, Any], optax.apply_updates(gp, g_upd))
      fin = (
        jnp.isfinite(g_val)
        & jnp.isfinite(d_val)
        & jnp.all(
          jnp.stack(
            [jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(g_grads)]
          )
        )
      )
      out_gp = jax.tree.map(lambda a, b: jnp.where(fin, a, b), new_gp, gp)
      out_dp = jax.tree.map(lambda a, b: jnp.where(fin, a, b), new_dp, dp)
      out_gos = jax.tree.map(lambda a, b: jnp.where(fin, a, b), new_gos, gos)
      out_dos = jax.tree.map(lambda a, b: jnp.where(fin, a, b), new_dos, dos)
      return (out_gp, out_dp, out_gos, out_dos, next_rkey), {
        **g_avg_m,
        "finite": fin.astype(jnp.float32),
      }

    gan_scan_k = min(steps_per_loop, gan_steps)
    gan_k_jit = jax.jit(
      lambda st, stk: jax.lax.scan(_gan_step, st, stk),
      donate_argnums=(0,),
    )
    gan_state = (params, d_params, g_opt, d_opt, gan_rng)
    g_step = 0
    with tpu.ThreadedDevicePrefetcher(
      batch_iter, mesh, steps_per_loop=gan_scan_k, max_prefetch=2
    ) as g_pre:
      while g_step < gan_steps:
        t0 = time.perf_counter()
        rem = gan_steps - g_step
        if rem >= gan_scan_k:
          chunk = next(g_pre)
          gan_state, gm_dev = gan_k_jit(gan_state, chunk)
          c_len = gan_scan_k
        else:
          sh = tuple(
            x[None, ...]
            for x in tpu.shard_batch(next(batch_iter), mesh, stacked=False)
          )
          gan_state, gm_dev = gan_k_jit(gan_state, sh)
          c_len = 1
        g_step += c_len
        dt = time.perf_counter() - t0
        last_metrics = {k: float(np.asarray(v)[-1]) for k, v in gm_dev.items()}
        print(
          f"[Phase 5b GAN {g_step:4d}/{gan_steps}] loss={last_metrics['loss']:.5g} "
          f"sqrt_sai={last_metrics['sqrt_sai']:.5g} adv={last_metrics['adv']:.4g} "
          f"fm={last_metrics['fm']:.4g} d={last_metrics['d_loss']:.4g} ({dt:.2f}s)",
          flush=True,
        )

    params = gan_state[0]
    gan_params_host = tpu.to_host_tree(params)
    tpu.delete_tree(gan_state[1:])
    gan_ckpt = out_root / "checkpoint_gan.npz"
    sai_inverter.save_sai_checkpoint(gan_ckpt, gan_params_host, cfg)
    sai_inverter.save_sai_checkpoint(
      out_root / "checkpoint_latest.npz", gan_params_host, cfg
    )
    gan_scores = _evaluate_sai_inverter(
      apply_fn=eval_apply_jit,
      params=params,
      eval_clips=eval_clips,
      healthy_cochlea=healthy_dev,
      impaired_cochlea=impaired_dev,
      sample_rate_hz=sample_rate_hz,
      mask_samples=mask_samples,
      out_root=out_root,
      step_idx=steps + gan_steps,
      mesh=mesh,
    )
    print(f"[Phase 5b GAN eval] {json.dumps(gan_scores)}", flush=True)
    if np.isfinite(gan_scores["selection_loss"]) and (
      steps == 0
      or not np.isfinite(best_sel_loss)
      or gan_scores["selection_loss"] <= best_sel_loss
    ):
      best_sel_loss = float(gan_scores["selection_loss"])
      best_scores = gan_scores
      best_params_host = gan_params_host
      sai_inverter.save_sai_checkpoint(ckpt_path, best_params_host, cfg)

  return {
    "checkpoint_path": str(ckpt_path),
    "best_scores": best_scores,
    "last_metrics": last_metrics,
    "params": int(n_params),
  }


def main(argv: Sequence[str]) -> None:
  """CLI entrypoint for Phase 5b SAI-to-audio training."""
  del argv
  if not _OUTPUT_DIR.value:
    raise app.UsageError("--output_dir is required.")
  run_sai_training(
    output_dir=_OUTPUT_DIR.value,
    data_dir=_DATA_DIR.value,
    pool_path=_POOL.value,
    init_checkpoint=_INIT_CHECKPOINT.value,
    model_dim=_MODEL_DIM.value,
    stem_blocks=_STEM_BLOCKS.value,
    decoder_blocks=_DECODER_BLOCKS.value,
    sample_rate_hz=_SAMPLE_RATE_HZ.value,
    seconds=_SECONDS.value,
    mask_seconds=_MASK_SECONDS.value,
    output_highpass_hz=_OUTPUT_HIGHPASS_HZ.value,
    stft_weight=_STFT_WEIGHT.value,
    ucl_weight=_UCL_WEIGHT.value,
    clip_norm=_CLIP_NORM.value,
    lr=_LR.value,
    lr_schedule=_LR_SCHEDULE.value,
    steps=_STEPS.value,
    healthy_warmup_steps=_HEALTHY_WARMUP_STEPS.value,
    gan_steps=_GAN_STEPS.value,
    gan_lr=_GAN_LR.value,
    gan_weight=_GAN_WEIGHT.value,
    fm_weight=_FM_WEIGHT.value,
    batch_size=_BATCH_SIZE.value,
    grad_accum_steps=_GRAD_ACCUM_STEPS.value,
    steps_per_loop=_STEPS_PER_LOOP.value,
    ears=_EARS.value,
    share_healthy=_SHARE_HEALTHY.value,
    share_flat=_SHARE_FLAT.value,
    middle_ear_enabled=_MIDDLE_EAR.value,
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
