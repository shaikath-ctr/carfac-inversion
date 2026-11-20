"""Phase 4 TPU training script combining AFHI/arcfac settings with CochleaHybrid.

Streams lightweight 1D waveforms on host (``ProcessedClips`` ArrayRecord shards
via Grain + ``hp_acoustic.data.pipeline``, with synthetic speech fallback for
tests), synthesizes healthy and impaired target CARFAC NAPs on TPU HBM via
``jax.vmap`` across randomized listener profiles, and optimizes direct causal
NAP-to-audio models inside multi-step ``jax.lax.scan`` loops.
"""

import csv
import dataclasses
import gc
import json
import math
import pathlib
import time
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import TypeVar
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app
from absl import flags
from hp_acoustic.data import pipeline
from hp_acoustic.data import transforms as transforms_lib
from hp_acoustic.data.sources import processed
from hp_acoustic.filters import middle_ear
from hp_acoustic.io import audio_io
from hp_acoustic.models import carfac

from carfac_inversion import arcfac_metrics
from carfac_inversion import nap_inverter
from carfac_inversion import neural
from carfac_inversion import tpu
from carfac_inversion import vocoder_losses
from carfac_inversion.scripts import benchmark_suite

FLAGS = flags.FLAGS

_OUTPUT_DIR: flags.FlagHolder[str | None]
_DATA_DIR: flags.FlagHolder[str | None]
_POOL: flags.FlagHolder[str | None]
_INIT_CHECKPOINT: flags.FlagHolder[str | None]
_ARCHITECTURE: flags.FlagHolder[str]
_WIDTH: flags.FlagHolder[int]
_NUM_BLOCKS: flags.FlagHolder[int]
_KERNEL_SIZE: flags.FlagHolder[int]
_COARSE_DIM: flags.FlagHolder[int]
_COARSE_HOP: flags.FlagHolder[int]
_LOOKAHEAD: flags.FlagHolder[int]
_SAMPLE_RATE_HZ: flags.FlagHolder[int]
_SECONDS: flags.FlagHolder[float]
_MASK_SECONDS: flags.FlagHolder[float]
_OUTPUT_HIGHPASS_HZ: flags.FlagHolder[float]
_WARMUP_WEIGHT: flags.FlagHolder[float]
_LAM: flags.FlagHolder[float]
_LAM_IMPAIRED: flags.FlagHolder[float]
_STFT_WEIGHT: flags.FlagHolder[float]
_SAI_WEIGHT: flags.FlagHolder[float]
_CLIP_NORM: flags.FlagHolder[float]
_LR: flags.FlagHolder[float]
_LR_SCHEDULE: flags.FlagHolder[str]
_STEPS: flags.FlagHolder[int]
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
  """Registers Phase 4 CLI flags cleanly without double-registration issues."""
  global _OUTPUT_DIR
  global _DATA_DIR
  global _POOL
  global _INIT_CHECKPOINT
  global _ARCHITECTURE
  global _WIDTH
  global _NUM_BLOCKS
  global _KERNEL_SIZE
  global _COARSE_DIM
  global _COARSE_HOP
  global _LOOKAHEAD
  global _SAMPLE_RATE_HZ
  global _SECONDS
  global _MASK_SECONDS
  global _OUTPUT_HIGHPASS_HZ
  global _WARMUP_WEIGHT
  global _LAM
  global _LAM_IMPAIRED
  global _STFT_WEIGHT
  global _SAI_WEIGHT
  global _CLIP_NORM
  global _LR
  global _LR_SCHEDULE
  global _STEPS
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
    "Directory for config, checkpoints, training log, and arcfac scores.",
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
    "Optional .npz file holding listener ohc_health / v_half_shift draws.",
    flag_values=flag_values,
  )
  _INIT_CHECKPOINT = flags.DEFINE_string(
    "init_checkpoint",
    None,
    "Optional Phase 4 .npz checkpoint for resuming or GAN fine-tuning.",
    flag_values=flag_values,
  )
  _ARCHITECTURE = flags.DEFINE_enum(
    "architecture",
    "cochlea_hybrid",
    ("tcn", "cochlea_hybrid"),
    "Causal inverter backbone: arcfac 'tcn' or Phase 4 'cochlea_hybrid'.",
    flag_values=flag_values,
  )
  _WIDTH = flags.DEFINE_integer(
    "width",
    64,
    "Sample-rate causal TCN channel width.",
    flag_values=flag_values,
  )
  _NUM_BLOCKS = flags.DEFINE_integer(
    "num_blocks",
    13,
    "Number of dilated causal blocks.",
    flag_values=flag_values,
  )
  _KERNEL_SIZE = flags.DEFINE_integer(
    "kernel_size", 3, "Causal conv kernel width.", flag_values=flag_values
  )
  _COARSE_DIM = flags.DEFINE_integer(
    "coarse_dim",
    96,
    "Coarse 2 ms frame-rate branch width for cochlea_hybrid.",
    flag_values=flag_values,
  )
  _COARSE_HOP = flags.DEFINE_integer(
    "coarse_hop",
    64,
    "Frame hop in samples for coarse AGC branch.",
    flag_values=flag_values,
  )
  _LOOKAHEAD = flags.DEFINE_integer(
    "lookahead",
    256,
    "Causal lookahead delay L in samples (8 ms at 32 kHz).",
    flag_values=flag_values,
  )
  _SAMPLE_RATE_HZ = flags.DEFINE_integer(
    "sample_rate_hz", 32000, "Sample rate in Hz.", flag_values=flag_values
  )
  _SECONDS = flags.DEFINE_float(
    "seconds",
    1.0,
    "Training chunk duration in seconds.",
    flag_values=flag_values,
  )
  _MASK_SECONDS = flags.DEFINE_float(
    "mask_seconds",
    0.25,
    "Leading CARFAC start-from-rest mask in seconds (8000 samples at 32 kHz).",
    flag_values=flag_values,
  )
  _OUTPUT_HIGHPASS_HZ = flags.DEFINE_float(
    "output_highpass_hz",
    20.0,
    "Causal 4th-order Butterworth high-pass cutoff in Hz (0 disables).",
    flag_values=flag_values,
  )
  _WARMUP_WEIGHT = flags.DEFINE_float(
    "warmup_weight",
    0.01,
    "Normalized prefix energy penalty over masked warmup window.",
    flag_values=flag_values,
  )
  _LAM = flags.DEFINE_float(
    "lam",
    0.01,
    "Waveform L1 tie-breaker weight on healthy targets.",
    flag_values=flag_values,
  )
  _LAM_IMPAIRED = flags.DEFINE_float(
    "lam_impaired",
    0.0,
    "Waveform L1 tie-breaker weight on impaired targets (0 prevents healing).",
    flag_values=flag_values,
  )
  _STFT_WEIGHT = flags.DEFINE_float(
    "stft_weight",
    0.0,
    "Multi-resolution STFT weight applied to healthy targets only.",
    flag_values=flag_values,
  )
  _SAI_WEIGHT = flags.DEFINE_float(
    "sai_weight",
    0.0,
    "Optional unmasked SAI L1 weight applied to all targets.",
    flag_values=flag_values,
  )
  _CLIP_NORM = flags.DEFINE_float(
    "clip", 1.0, "Global gradient-norm clip threshold.", flag_values=flag_values
  )
  _LR = flags.DEFINE_float(
    "lr",
    3e-3,
    "Adam learning rate (arcfac default 3e-3).",
    flag_values=flag_values,
  )
  _LR_SCHEDULE = flags.DEFINE_enum(
    "lr_schedule",
    "constant",
    ("constant", "cosine"),
    "Learning rate schedule for Stage 1 training.",
    flag_values=flag_values,
  )
  _STEPS = flags.DEFINE_integer(
    "steps", 3000, "Number of Stage 1 training steps.", flag_values=flag_values
  )
  _GAN_STEPS = flags.DEFINE_integer(
    "gan_steps",
    0,
    "Optional Stage 2 MPD+MRD adversarial fine-tuning steps.",
    flag_values=flag_values,
  )
  _GAN_LR = flags.DEFINE_float(
    "gan_lr",
    2e-4,
    "Generator/discriminator LR for Stage 2 GAN.",
    flag_values=flag_values,
  )
  _GAN_WEIGHT = flags.DEFINE_float(
    "gan_weight",
    0.05,
    "Adversarial loss weight in Stage 2 GAN.",
    flag_values=flag_values,
  )
  _FM_WEIGHT = flags.DEFINE_float(
    "fm_weight",
    2.0,
    "Feature matching multiplier inside Stage 2 GAN.",
    flag_values=flag_values,
  )
  _BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    32,
    "Global training mini-batch size.",
    flag_values=flag_values,
  )
  _GRAD_ACCUM_STEPS = flags.DEFINE_integer(
    "grad_accum_steps",
    1,
    "Gradient accumulation micro-steps splitting global batch_size evenly.",
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
    "Listener ear distribution during training.",
    flag_values=flag_values,
  )
  _SHARE_HEALTHY = flags.DEFINE_float(
    "share_healthy",
    0.25,
    "Fraction of healthy target ears in mixture mode.",
    flag_values=flag_values,
  )
  _SHARE_FLAT = flags.DEFINE_float(
    "share_flat",
    0.35,
    "Fraction of flat OHC loss ears [0.1, 1.0] in mixture mode.",
    flag_values=flag_values,
  )
  _MIDDLE_EAR = flags.DEFINE_boolean(
    "middle_ear",
    False,
    "Enable middle-ear filter symmetrically on target and healthy ears.",
    flag_values=flag_values,
  )
  _EVAL_EVERY = flags.DEFINE_integer(
    "eval_every",
    250,
    "Validation cadence in training steps.",
    flag_values=flag_values,
  )
  _EVAL_CLIPS = flags.DEFINE_integer(
    "eval_clips",
    3,
    "Number of held-out evaluation clips.",
    flag_values=flag_values,
  )
  _SCORE_SECONDS = flags.DEFINE_float(
    "score_seconds",
    2.0,
    "Duration in seconds evaluated per held-out clip.",
    flag_values=flag_values,
  )
  _SEED = flags.DEFINE_integer(
    "seed",
    0,
    "PRNG seed for weights, listeners, and data.",
    flag_values=flag_values,
  )
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
    "Default JAX matmul precision.",
    flag_values=flag_values,
  )


@dataclasses.dataclass(frozen=True)
class _ProbabilisticNoiseBlend:
  """Applies stochastic pure-NumPy white/brown noise blending without JAX RPCs."""

  probability: float = 0.35
  snr_db: tuple[float, float] = (5.0, 30.0)

  def __call__(
    self, example: transforms_lib.Example, rng: np.random.Generator
  ) -> transforms_lib.Example:
    """Blends noise into ``example.noisy`` with probability ``self.probability``."""
    if rng.random() >= self.probability:
      return example
    clean = np.asarray(example.clean, dtype=np.float32)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    if rng.random() < 0.5:
      # Brownian 1/f^2 tilt matching hp_acoustic ColorNoise("brown").
      walk = np.cumsum(noise, dtype=np.float64)
      walk -= np.mean(walk)
      peak = float(np.max(np.abs(walk)))
      noise = (walk / peak).astype(np.float32) if peak > 0.0 else noise
    snr = float(rng.uniform(*self.snr_db))
    noisy = transforms_lib.mix_at_snr(clean, noise, snr)
    return dataclasses.replace(example, noisy=noisy.astype(np.float32))


@dataclasses.dataclass(frozen=True)
class _LevelStepTransform:
  """Applies an occasional mid-clip gain step to exercise dynamic CARFAC AGC."""

  probability: float = 0.2
  step_db: tuple[float, float] = (-15.0, 15.0)

  def __call__(
    self, example: transforms_lib.Example, rng: np.random.Generator
  ) -> transforms_lib.Example:
    if rng.random() >= self.probability:
      return example
    clean = np.asarray(example.clean, dtype=np.float32)
    noisy = np.asarray(example.noisy, dtype=np.float32)
    n = len(clean)
    if n < 32:
      return example
    cut = int(rng.integers(max(1, n // 5), max(2, (4 * n) // 5)))
    gain = np.float32(10.0 ** (rng.uniform(*self.step_db) / 20.0))
    ramp_len = min(64, n - cut)
    env = np.ones(n, dtype=np.float32)
    env[cut:] = gain
    if ramp_len > 1:
      t = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
      env[cut : cut + ramp_len] = 1.0 + (gain - 1.0) * t
    return dataclasses.replace(example, clean=clean * env, noisy=noisy * env)


def _apply_ear_with_profile(
  base_cochlea: carfac.CarfacCochleaModel,
  wave_1d: jnp.ndarray,
  ohc_1d: jnp.ndarray,
  v_shift_2d: jnp.ndarray,
) -> jnp.ndarray:
  """Evaluates CARFAC on one 1D clip with per-example OHC and synaptic shifts."""
  new_ears = []
  for ear_w in base_cochlea.weights.ears:
    car_w = dataclasses.replace(
      ear_w.car, ohc_health=ohc_1d.astype(jnp.float32)
    )
    if ear_w.syn is not None:
      syn_w = dataclasses.replace(
        ear_w.syn, v_halfs=ear_w.syn.v_halfs + v_shift_2d.astype(jnp.float32)
      )
      new_ears.append(dataclasses.replace(ear_w, car=car_w, syn=syn_w))
    else:
      new_ears.append(dataclasses.replace(ear_w, car=car_w))
  new_weights = dataclasses.replace(base_cochlea.weights, ears=new_ears)
  model = dataclasses.replace(base_cochlea, weights=new_weights)
  return model(wave_1d[:, None])[:, :, 0]


def _draw_listener_batch(
  rng: np.random.Generator,
  batch_size: int,
  num_channels: int,
  cfs: np.ndarray,
  *,
  ears_mode: str,
  share_healthy: float,
  share_flat: float,
  pool_ohc: np.ndarray | None,
  pool_shift: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Samples ``(ohc, v_half_shift, is_healthy)`` matching ``arcfac`` mixture."""
  ohc = np.ones((batch_size, num_channels), dtype=np.float32)
  shift = np.zeros((batch_size, num_channels, 3), dtype=np.float32)
  healthy = np.ones((batch_size,), dtype=bool)
  if ears_mode == "healthy":
    return ohc, shift, healthy

  norm_f = np.linspace(0.0, 1.0, num_channels, dtype=np.float32)
  if len(cfs) == num_channels and float(cfs[0]) > float(cfs[-1]):
    # Pole freqs ordered high-to-low: flip so 0 is basal high-CF or use CF ratio.
    hf_weight = np.clip((np.log10(np.maximum(cfs, 50.0)) - 2.3) / 1.5, 0.0, 1.0)
  else:
    hf_weight = 1.0 - norm_f

  draws = rng.uniform(size=batch_size)
  for i in range(batch_size):
    u = float(draws[i])
    if u < share_healthy:
      continue
    healthy[i] = False
    if u < share_healthy + share_flat:
      ohc[i] = np.float32(rng.uniform(0.1, 1.0))
    elif pool_ohc is not None:
      idx = int(rng.integers(0, len(pool_ohc)))
      ohc[i] = pool_ohc[idx]
      if pool_shift is not None:
        shift[i] = pool_shift[idx]
    else:
      # Random clinical sloping/notched OHC curve + mild synaptic shift.
      low_h = rng.uniform(0.4, 1.0)
      high_h = rng.uniform(0.05, low_h)
      curve = low_h * (1.0 - hf_weight) + high_h * hf_weight
      ohc[i] = np.clip(curve, 0.05, 1.0).astype(np.float32)
      if rng.random() < 0.5:
        v_s = rng.uniform(0.0, 0.015)
        shift[i] = np.float32(v_s)
  return ohc, shift, healthy


def _load_listener_pool(
  pool_path: str | None,
  num_channels: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
  """Loads optional listener fit pool ``.npz`` holding OHC and synaptic shifts."""
  if not pool_path:
    return None, None
  with np.load(pool_path) as d:
    ohc = np.asarray(d["ohc_health"], dtype=np.float32)
    if ohc.ndim != 2 or ohc.shape[1] != num_channels:
      raise ValueError(
        f"Listener pool ohc_health has shape {ohc.shape}, expected (*, {num_channels})."
      )
    shift = (
      np.asarray(d["v_half_shift"], dtype=np.float32)
      if "v_half_shift" in d
      else np.zeros((len(ohc), num_channels, 3), dtype=np.float32)
    )
    return ohc, shift


def _build_synthetic_waveform_iter(
  *,
  sample_rate_hz: int,
  num_samples: int,
  batch_size: int,
  seed: int,
) -> Iterator[np.ndarray]:
  """Endless deterministic synthetic speech-like batches for offline tests."""
  rng = np.random.default_rng(seed)
  t = np.arange(num_samples, dtype=np.float32) / float(sample_rate_hz)
  target_ref = 1.0 / math.sqrt(2.0)
  while True:
    batch = np.zeros((batch_size, num_samples), dtype=np.float32)
    for b in range(batch_size):
      if rng.random() < 0.05:
        continue
      f0 = float(rng.uniform(95.0, 260.0))
      f_mod = float(rng.uniform(2.5, 6.0))
      env = 0.5 + 0.5 * np.sin(
        2.0 * math.pi * f_mod * t + float(rng.uniform(0, 6.28))
      )
      sig = np.zeros_like(t)
      for k in range(1, 16):
        fk = k * f0
        if fk < 0.42 * sample_rate_hz:
          sig += (1.0 / (k**1.1)) * np.sin(
            2.0 * math.pi * fk * t + float(rng.uniform(0, 6.28))
          )
      sig *= env
      sig += 0.08 * rng.standard_normal(num_samples).astype(np.float32)
      rms = float(np.sqrt(np.mean(sig**2))) + 1e-8
      spl = float(rng.uniform(30.0, 95.0))
      tgt_rms = target_ref * (10.0 ** ((spl - 104.0) / 20.0))
      batch[b] = (sig * (tgt_rms / rms)).astype(np.float32)
    yield batch


@dataclasses.dataclass(frozen=True)
class _SilenceExampleTransform:
  """Zeroes a small fraction of training examples symmetrically."""

  probability: float = 0.05

  def __call__(
    self, example: transforms_lib.Example, rng: np.random.Generator
  ) -> transforms_lib.Example:
    """Silences ``clean`` and ``noisy`` with probability ``self.probability``."""
    if rng.random() >= self.probability:
      return example
    zeros_c = np.zeros_like(example.clean, dtype=np.float32)
    zeros_n = np.zeros_like(example.noisy, dtype=np.float32)
    return dataclasses.replace(example, clean=zeros_c, noisy=zeros_n)


def _build_grain_waveform_iter(
  *,
  data_dir: str,
  sample_rate_hz: int,
  seconds: float,
  batch_size: int,
  seed: int,
  level_db_spl: tuple[float, float] = (30.0, 95.0),
) -> Iterator[np.ndarray]:
  """Streams 1D float32 audio batches from ProcessedClips via Grain."""
  speech_src = processed.ProcessedClips(data_dir)
  transforms = (
    _ProbabilisticNoiseBlend(probability=0.35, snr_db=(5.0, 30.0)),
    transforms_lib.NormalizeLevel(level_db_spl=level_db_spl),
    _LevelStepTransform(probability=0.20, step_db=(-15.0, 15.0)),
    transforms_lib.CosineWindow(ramp_seconds=0.005),
    _SilenceExampleTransform(probability=0.05),
  )
  cfg = pipeline.DataConfig(
    sources=(speech_src,),
    chunk_seconds=seconds,
    chunks_per_clip=2,
    sample_rate=sample_rate_hz,
    transforms=transforms,
    global_batch_size=batch_size,
    max_epochs=10000,
    seed=seed,
  )
  with tpu.host_cpu_context():
    ds = pipeline.build(cfg, split="train")
    it = iter(ds)
  while True:
    try:
      with tpu.host_cpu_context():
        noisy_b, _ = next(it)
      yield np.asarray(noisy_b[..., 0], dtype=np.float32)
    except StopIteration:
      with tpu.host_cpu_context():
        it = iter(ds)


def _iter_training_batches(
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
  lam_healthy: float,
  lam_impaired: float,
  stft_weight: float,
) -> Iterator[
  tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
  """Pairs raw 1D waveforms with lightweight per-clip ear and loss masks."""
  for wav in wave_iter:
    bsz = wav.shape[0]
    ohc, shift, is_healthy = _draw_listener_batch(
      rng,
      bsz,
      num_channels,
      cfs,
      ears_mode=ears_mode,
      share_healthy=share_healthy,
      share_flat=share_flat,
      pool_ohc=pool_ohc,
      pool_shift=pool_shift,
    )
    lam_b = np.where(is_healthy, lam_healthy, lam_impaired).astype(np.float32)
    stft_b = np.where(is_healthy, stft_weight, 0.0).astype(np.float32)
    yield wav, ohc, shift, lam_b, stft_b


def _evaluate_phase4(
  *,
  apply_fn: Any,
  params: Mapping[str, Any],
  eval_clips: Sequence[tuple[str, np.ndarray]],
  healthy_cochlea: carfac.CarfacCochleaModel,
  impaired_cochlea: carfac.CarfacCochleaModel,
  sample_rate_hz: int,
  lookahead: int,
  mask_samples: int,
  out_root: pathlib.Path,
  step_idx: int,
  mesh: jax.sharding.Mesh | None = None,
) -> dict[str, Any]:
  """Evaluates ``params`` on ``healthy`` & ``flat:0.5`` using ``arcfac.score``."""
  forward_jit = jax.jit(lambda w, cm: cm(w[:, None])[:, :, 0])
  hl_arr = benchmark_suite.HALF_OHC_HEARING_LOSS_DB
  targets = (
    ("healthy", healthy_cochlea, np.zeros(6, dtype=np.float64)),
    ("flat-0.5", impaired_cochlea, hl_arr),
  )

  def _place(arr: np.ndarray) -> jnp.ndarray:
    arr_j = jnp.asarray(arr, dtype=jnp.float32)
    return tpu.replicate_tree(arr_j, mesh) if mesh is not None else arr_j

  # Compile per unique padded length cleanly inside loop.
  scores: dict[str, Any] = {"step": step_idx, "entries": {}}
  nap_rt_vals: list[float] = []
  nap_kept_vals: list[float] = []
  sai_rt_vals: list[float] = []
  sai_dir_vals: list[float] = []
  sdr_vals: list[float] = []

  render_dir = out_root / "renders" / f"step{step_idx}"
  render_dir.mkdir(parents=True, exist_ok=True)
  scores_dir = out_root / "scores"
  scores_dir.mkdir(parents=True, exist_ok=True)

  for clip_id, raw_x in eval_clips:
    x = benchmark_suite._scale_to_spl(raw_x, 65.0)
    x_pad = np.concatenate([x, np.zeros(lookahead, dtype=np.float32)])
    healthy_nap_pad = np.asarray(
      forward_jit(_place(x_pad), healthy_cochlea),
      dtype=np.float32,
    )
    n_hx = healthy_nap_pad[: len(x)]
    for target_name, tgt_cm, aud_hl in targets:
      nap_pad = (
        healthy_nap_pad
        if target_name == "healthy"
        else np.asarray(forward_jit(_place(x_pad), tgt_cm), dtype=np.float32)
      )
      y_pad = np.asarray(
        apply_fn(params, _place(nap_pad[None]))[0], dtype=np.float32
      )
      y = y_pad[lookahead : lookahead + len(x)]
      n_tgt = nap_pad[: len(x)]
      y_eval_pad = np.concatenate([y, np.zeros(lookahead, dtype=np.float32)])
      n_hy = np.asarray(
        forward_jit(_place(y_eval_pad), healthy_cochlea), dtype=np.float32
      )[: len(x)]

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
        nap_rt_vals.append(float(sc["nap"]["round_trip_rel"]))
        sai_rt_vals.append(float(sc["sai"]["round_trip_rel"]))
        sdr_vals.append(float(metrics["si_sdr"]))
      else:
        nap_kept_vals.append(float(sc["nap"].get("kept", float("nan"))))
        sai_dir_vals.append(float(sc["sai"].get("direction", float("nan"))))

      out_wav = render_dir / f"{clip_id}-{target_name}.wav"
      audio_io.save_audio_to_file(y, str(out_wav), sample_rate_hz)

  mean_rt = float(np.nanmean(nap_rt_vals)) if nap_rt_vals else float("nan")
  mean_kept = (
    float(np.nanmean(nap_kept_vals)) if nap_kept_vals else float("nan")
  )
  mean_sai_rt = float(np.nanmean(sai_rt_vals)) if sai_rt_vals else float("nan")
  mean_sai_dir = (
    float(np.nanmean(sai_dir_vals)) if sai_dir_vals else float("nan")
  )
  mean_sdr = float(np.nanmean(sdr_vals)) if sdr_vals else float("nan")
  score_proxy = (
    mean_rt + mean_kept
    if np.isfinite(mean_rt) and np.isfinite(mean_kept)
    else mean_rt
  )
  scores["summary"] = {
    "nap_round_trip_rel": mean_rt,
    "sai_round_trip_rel": mean_sai_rt,
    "nap_kept": mean_kept,
    "sai_direction": mean_sai_dir,
    "si_sdr_db": mean_sdr,
    "selection_loss": score_proxy,
  }
  with (scores_dir / f"step{step_idx}.json").open("w") as f:
    json.dump(scores, f, indent=1, default=str)
  return scores["summary"]


_BatchT = TypeVar("_BatchT", bound=tuple[jnp.ndarray, ...])


def _split_microbatches(
  batch: _BatchT,
  grad_accum_steps: int,
) -> _BatchT:
  """Splits a ``(B, ...)`` sharded batch into ``(A, B // A, ...)`` microbatches."""
  if grad_accum_steps == 1:
    return cast(_BatchT, jax.tree.map(lambda x: x[None, ...], batch))

  def _split(x: jnp.ndarray) -> jnp.ndarray:
    micro_b = x.shape[0] // grad_accum_steps
    # Keep leading axis divisible by mesh size before swapping axes so GSPMD
    # avoids cross-chip resharding.
    return jnp.swapaxes(
      x.reshape((micro_b, grad_accum_steps, *x.shape[1:])), 0, 1
    )

  return cast(_BatchT, jax.tree.map(_split, batch))


def run_phase4_training(
  *,
  output_dir: str,
  data_dir: str | None = None,
  pool_path: str | None = None,
  init_checkpoint: str | None = None,
  architecture: str = "cochlea_hybrid",
  width: int = 64,
  num_blocks: int = 13,
  kernel_size: int = 3,
  coarse_dim: int = 96,
  coarse_hop: int = 64,
  lookahead: int = 256,
  sample_rate_hz: int = 32000,
  seconds: float = 1.0,
  mask_seconds: float = 0.25,
  output_highpass_hz: float = 20.0,
  warmup_weight: float = 0.01,
  lam: float = 0.01,
  lam_impaired: float = 0.0,
  stft_weight: float = 0.0,
  sai_weight: float = 0.0,
  clip_norm: float = 1.0,
  lr: float = 3e-3,
  lr_schedule: str = "constant",
  steps: int = 3000,
  gan_steps: int = 0,
  gan_lr: float = 2e-4,
  gan_weight: float = 0.05,
  fm_weight: float = 2.0,
  batch_size: int = 32,
  grad_accum_steps: int = 1,
  steps_per_loop: int = 25,
  ears: str = "mixture",
  share_healthy: float = 0.25,
  share_flat: float = 0.35,
  middle_ear_enabled: bool = False,
  eval_every: int = 250,
  eval_clips_count: int = 3,
  score_seconds: float = 2.0,
  seed: int = 0,
  compilation_cache_dir: str | None = None,
  matmul_precision: str = "high",
) -> dict[str, Any]:
  """Trains a Phase 4 direct causal NAP-to-audio inverter using ``arcfac`` settings."""
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
  if grad_accum_steps < 1:
    raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}.")
  if batch_size < 1 or batch_size % grad_accum_steps != 0:
    raise ValueError(
      f"batch_size={batch_size} must be divisible by grad_accum_steps={grad_accum_steps}."
    )
  micro_batch_size = batch_size // grad_accum_steps
  if micro_batch_size % n_local != 0:
    raise ValueError(
      f"micro_batch_size={micro_batch_size} (batch_size // grad_accum_steps) "
      f"must be a positive multiple of {n_local}."
    )

  out_root = pathlib.Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)
  num_samples = max(64, int(round(seconds * sample_rate_hz)))
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

  pool_ohc, pool_shift = _load_listener_pool(pool_path, num_channels)
  if data_dir:
    wave_iter = _build_grain_waveform_iter(
      data_dir=data_dir,
      sample_rate_hz=sample_rate_hz,
      seconds=seconds,
      batch_size=batch_size,
      seed=seed,
    )
    eval_clips = benchmark_suite.load_speech_clips(
      data_dir,
      sample_rate_hz=sample_rate_hz,
      max_clips=eval_clips_count,
      clip_seconds=score_seconds,
      split="test",
    )
  else:
    wave_iter = _build_synthetic_waveform_iter(
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

  # Calibrate input/output scales on a healthy batch just like arcfac.nap.train.
  calib_x_np = next(wave_iter)
  # Ensure non-silent calibration energy if SilenceExample drew a silent batch.
  if float(np.sqrt(np.mean(calib_x_np**2))) < 1e-6:
    calib_x_np = next(wave_iter)
  healthy_ear_fn = jax.jit(jax.vmap(lambda w: healthy_dev(w[:, None])[:, :, 0]))
  calib_nap = healthy_ear_fn(tpu.shard_batch(calib_x_np, mesh, stacked=False))
  input_scale = max(float(jnp.sqrt(jnp.mean(calib_nap**2))), 1e-6)
  output_scale = max(float(np.sqrt(np.mean(calib_x_np**2))), 1e-6)

  if init_checkpoint:
    model, init_params_raw, cfg = nap_inverter.load_direct_checkpoint(
      init_checkpoint
    )
    init_params_host = tpu.to_host_tree(init_params_raw)
  else:
    cfg = nap_inverter.DirectInverterConfig(
      architecture=architecture,
      width=width,
      kernel_size=kernel_size,
      num_blocks=num_blocks,
      coarse_dim=coarse_dim,
      coarse_hop=coarse_hop,
      input_scale=input_scale,
      output_scale=output_scale,
      lookahead=lookahead,
      output_highpass_hz=output_highpass_hz,
      fs=float(sample_rate_hz),
    )
    model = nap_inverter.CochleaDirectInverter(config=cfg)
    with tpu.host_cpu_context():
      rng_key = jax.random.PRNGKey(seed)
      raw_p = model.init(
        rng_key,
        jnp.zeros((1, min(1024, num_samples), num_channels), dtype=jnp.float32),
      )
      init_params_host = tpu.to_host_tree(raw_p)

  params = cast(Mapping[str, Any], tpu.replicate_tree(init_params_host, mesh))
  n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

  target_naps_jit = jax.jit(
    jax.vmap(
      lambda w, o, s: _apply_ear_with_profile(healthy_dev, w, o, s),
      in_axes=(0, 0, 0),
    )
  )
  stft_res = vocoder_losses.stft_resolutions_for_rate(float(sample_rate_hz))
  apply_stft = stft_weight > 0.0

  config_payload = {
    "architecture": architecture,
    "cfg": dataclasses.asdict(cfg),
    "params": int(n_params),
    "sample_rate_hz": sample_rate_hz,
    "seconds": seconds,
    "mask_samples": mask_samples,
    "lam": lam,
    "lam_impaired": lam_impaired,
    "stft_weight": stft_weight,
    "sai_weight": sai_weight,
    "warmup_weight": warmup_weight,
    "lr": lr,
    "lr_schedule": lr_schedule,
    "clip_norm": clip_norm,
    "steps": steps,
    "gan_steps": gan_steps,
    "batch_size": batch_size,
    "grad_accum_steps": grad_accum_steps,
    "micro_batch_size": micro_batch_size,
    "steps_per_loop": steps_per_loop,
    "ears": ears,
    "share_healthy": share_healthy,
    "share_flat": share_flat,
    "middle_ear": middle_ear_enabled,
    "backend": backend,
    "n_devices": n_devices,
  }
  with (out_root / "config.json").open("w") as f:
    json.dump(config_payload, f, indent=1, default=str)

  print(
    f"[Phase 4] arch={architecture} params={n_params:,} fs={sample_rate_hz}Hz "
    f"T={num_samples} mask={mask_samples} L={lookahead} B={batch_size} "
    f"(accum={grad_accum_steps}x{micro_batch_size}) K={steps_per_loop} "
    f"backend={backend} devices={n_devices} lam=({lam}/{lam_impaired}) stft={stft_weight}",
    flush=True,
  )

  # Build optimizer schedule & guarded K-step scan update.
  if lr_schedule == "cosine" and steps > 1:
    sched: optax.Schedule | float = optax.warmup_cosine_decay_schedule(
      init_value=lr * 0.1,
      peak_value=lr,
      warmup_steps=min(200, max(1, steps // 10)),
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
    p: Mapping[str, Any],
    x_b: jnp.ndarray,
    tgt_nap: jnp.ndarray,
    lam_b: jnp.ndarray,
    stft_b: jnp.ndarray,
  ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    y_b = cast(jnp.ndarray, model.apply(p, tgt_nap))
    return vocoder_losses.compute_arcfac_inverter_loss(
      y_b,
      x_b,
      tgt_nap,
      healthy_ear_fn,
      lookahead=lookahead,
      mask_samples=mask_samples,
      output_scale=output_scale,
      lam=lam_b,
      warmup_weight=warmup_weight,
      stft_weight=stft_b,
      apply_stft=apply_stft,
      sai_weight=sai_weight,
      stft_resolutions=stft_res,
    )

  def _train_step_impl(
    p: Mapping[str, Any],
    os_s: optax.OptState,
    batch: tuple[
      jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],
  ) -> tuple[Mapping[str, Any], optax.OptState, dict[str, jnp.ndarray]]:
    micros = _split_microbatches(batch, grad_accum_steps)
    zero_grads = jax.tree.map(jnp.zeros_like, p)
    zero_metrics = {
      name: jnp.zeros((), dtype=jnp.float32)
      for name in (
        "loss",
        "nap_l1",
        "sai_l1",
        "l1_waveform",
        "warmup_energy",
        "stft",
      )
    }

    def _accum_step(
      carry: tuple[Mapping[str, Any], dict[str, jnp.ndarray]],
      micro: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
      ],
    ) -> tuple[tuple[Mapping[str, Any], dict[str, jnp.ndarray]], None]:
      g_acc, m_acc = carry
      x_b, ohc_b, shift_b, lam_b, stft_b = micro
      tgt_nap = jax.lax.stop_gradient(target_naps_jit(x_b, ohc_b, shift_b))
      (val, parts), grads = jax.value_and_grad(_step_loss, has_aux=True)(
        p, x_b, tgt_nap, lam_b, stft_b
      )
      next_g = jax.tree.map(jnp.add, g_acc, grads)
      step_m = {"loss": val, **parts}
      next_m = {k: m_acc[k] + step_m[k] for k in m_acc}
      return (next_g, next_m), None

    (grads_sum, metrics_sum), _ = jax.lax.scan(
      _accum_step, (zero_grads, zero_metrics), micros
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
      kept_os,
      {**avg_metrics, "finite": finite.astype(jnp.float32)},
    )

  single_step_fn, multi_step_fn = tpu.make_multi_step_fns(
    _train_step_impl, donate_argnums=(0, 1)
  )
  apply_jit = jax.jit(model.apply)

  log_path = out_root / "log.csv"
  with log_path.open("w", newline="") as f:
    csv.writer(f).writerow(
      (
        "step",
        "loss",
        "nap_l1",
        "sai_l1",
        "l1_waveform",
        "warmup_energy",
        "stft",
        "finite_frac",
        "seconds",
      )
    )

  np_rng = np.random.default_rng(seed + 1)
  batch_iter = _iter_training_batches(
    wave_iter=wave_iter,
    rng=np_rng,
    num_channels=num_channels,
    cfs=cfs,
    ears_mode=ears,
    share_healthy=share_healthy,
    share_flat=share_flat,
    pool_ohc=pool_ohc,
    pool_shift=pool_shift,
    lam_healthy=lam,
    lam_impaired=lam_impaired,
    stft_weight=stft_weight,
  )

  ckpt_path = out_root / "checkpoint.npz"
  best_sel_loss = float("inf")
  best_params_host = init_params_host
  best_scores: dict[str, Any] = {}
  last_metrics: dict[str, float] = {}

  # Step 0 baseline score before updates.
  step0_scores = _evaluate_phase4(
    apply_fn=apply_jit,
    params=params,
    eval_clips=eval_clips,
    healthy_cochlea=healthy_dev,
    impaired_cochlea=impaired_dev,
    sample_rate_hz=sample_rate_hz,
    lookahead=lookahead,
    mask_samples=mask_samples,
    out_root=out_root,
    step_idx=0,
    mesh=mesh,
  )
  print(f"[Phase 4 step 0] {json.dumps(step0_scores)}", flush=True)
  best_scores = step0_scores
  best_sel_loss = float(step0_scores["selection_loss"])
  nap_inverter.save_direct_checkpoint(ckpt_path, best_params_host, cfg)

  skipped_streak = 0
  step = 0
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
            tuple[
              jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
            ],
            next(prefetcher),
          )
          params, opt_state, m_dev = multi_step_fn(params, opt_state, chunk)
          chunk_len = eff_k
        else:
          host_one = next(batch_iter)
          sh_one = cast(
            tuple[
              jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
            ],
            tpu.shard_batch(host_one, mesh, stacked=False),
          )
          params, opt_state, m_dev = single_step_fn(params, opt_state, sh_one)
          chunk_len = 1

        step += chunk_len
        finite_arr = np.asarray(m_dev["finite"])
        if float(np.min(finite_arr)) < 0.5:
          skipped_streak += int(np.sum(finite_arr < 0.5))
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
              f"{last_metrics['nap_l1']:.6g}",
              f"{last_metrics['sai_l1']:.6g}",
              f"{last_metrics['l1_waveform']:.6g}",
              f"{last_metrics['warmup_energy']:.6g}",
              f"{last_metrics['stft']:.6g}",
              f"{float(np.mean(finite_arr)):.3f}",
              f"{dt:.3f}",
            )
          )
        print(
          f"[Phase 4 step {step:4d}/{steps}] loss={last_metrics['loss']:.5g} "
          f"nap={last_metrics['nap_l1']:.5g} sai={last_metrics['sai_l1']:.4g} "
          f"wave={last_metrics['l1_waveform']:.4g} warm={last_metrics['warmup_energy']:.4g} "
          f"stft={last_metrics['stft']:.4g} ({dt:.2f}s)",
          flush=True,
        )
        if step % max(1, eval_every) == 0 or step == steps:
          sc_sum = _evaluate_phase4(
            apply_fn=apply_jit,
            params=params,
            eval_clips=eval_clips,
            healthy_cochlea=healthy_dev,
            impaired_cochlea=impaired_dev,
            sample_rate_hz=sample_rate_hz,
            lookahead=lookahead,
            mask_samples=mask_samples,
            out_root=out_root,
            step_idx=step,
            mesh=mesh,
          )
          print(f"[Phase 4 eval {step}] {json.dumps(sc_sum)}", flush=True)
          if (
            np.isfinite(sc_sum["selection_loss"])
            and sc_sum["selection_loss"] < best_sel_loss
          ):
            best_sel_loss = float(sc_sum["selection_loss"])
            best_scores = sc_sum
            best_params_host = tpu.to_host_tree(params)
            nap_inverter.save_direct_checkpoint(
              ckpt_path, best_params_host, cfg
            )

    tpu.delete_tree(opt_state)
    gc.collect()

  if gan_steps > 0:
    stage1_ckpt = out_root / "checkpoint_stage1.npz"
    nap_inverter.save_direct_checkpoint(stage1_ckpt, best_params_host, cfg)
    params = cast(Mapping[str, Any], tpu.replicate_tree(best_params_host, mesh))
    mpd = vocoder_losses.MultiPeriodDiscriminator()
    mrd = vocoder_losses.MultiResolutionDiscriminator(resolutions=stft_res)
    start_idx = max(mask_samples, lookahead)
    gan_start = lookahead if num_samples - lookahead >= 2048 else 0
    with tpu.host_cpu_context():
      dk1, dk2 = jax.random.split(jax.random.PRNGKey(seed + 99))
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
    g_opt = tpu.replicate_tree(g_opt_h, mesh)
    d_opt = tpu.replicate_tree(d_opt_h, mesh)

    def _gan_step(
      carry_state: tuple[
        Mapping[str, Any], Mapping[str, Any], optax.OptState, optax.OptState
      ],
      batch: tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
      ],
    ) -> tuple[
      tuple[
        Mapping[str, Any], Mapping[str, Any], optax.OptState, optax.OptState
      ],
      dict[str, jnp.ndarray],
    ]:
      gp, dp, gos, dos = carry_state
      micros = _split_microbatches(batch, grad_accum_steps)
      inv_accum = jnp.float32(1.0 / float(grad_accum_steps))

      def _d_loss_micro(
        d_p: Mapping[str, Any],
        x_delayed: jnp.ndarray,
        y_fake: jnp.ndarray,
        clip_norm_rms: jnp.ndarray,
        healthy_mask: jnp.ndarray,
      ) -> jnp.ndarray:
        real_seg = x_delayed[:, gan_start:] / clip_norm_rms
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
        micro: tuple[
          jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
        ],
      ) -> tuple[
        tuple[jnp.ndarray, Mapping[str, Any]],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
      ]:
        d_sum, dg_sum = carry_d
        x_b, ohc_b, shift_b, _, _ = micro
        healthy_mask = (jnp.min(ohc_b, axis=1) > 0.999).astype(jnp.float32)
        tgt_nap = jax.lax.stop_gradient(target_naps_jit(x_b, ohc_b, shift_b))
        x_delayed = vocoder_losses.delay_samples(x_b, lookahead, axis=1)
        # Normalize amplitude entering D using steady-state RMS across 30-95 dB.
        clip_norm_rms = jnp.maximum(
          jnp.sqrt(
            jnp.mean(x_delayed[:, start_idx:] ** 2, axis=1, keepdims=True)
          ),
          0.1 * output_scale,
        )
        y_fake = jax.lax.stop_gradient(
          cast(jnp.ndarray, model.apply(gp, tgt_nap))
        )
        d_val_i, d_grad_i = jax.value_and_grad(_d_loss_micro)(
          dp, x_delayed, y_fake, clip_norm_rms, healthy_mask
        )
        return (
          d_sum + d_val_i,
          jax.tree.map(jnp.add, dg_sum, d_grad_i),
        ), (tgt_nap, x_delayed, clip_norm_rms, healthy_mask)

      zero_d_grads = jax.tree.map(jnp.zeros_like, dp)
      (d_val_sum, d_grads_sum), cached = jax.lax.scan(
        _d_accum_body, (jnp.zeros((), dtype=jnp.float32), zero_d_grads), micros
      )
      d_val = d_val_sum * inv_accum
      d_grads = jax.tree.map(lambda g: g * inv_accum, d_grads_sum)
      d_upd, new_dos = tx_d.update(d_grads, dos, dp)
      new_dp = cast(Mapping[str, Any], optax.apply_updates(dp, d_upd))

      def _g_loss_micro(
        g_p: Mapping[str, Any],
        x_b: jnp.ndarray,
        tgt_nap: jnp.ndarray,
        lam_b: jnp.ndarray,
        stft_b: jnp.ndarray,
        x_delayed: jnp.ndarray,
        clip_norm_rms: jnp.ndarray,
        healthy_mask: jnp.ndarray,
      ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        y_b = cast(jnp.ndarray, model.apply(g_p, tgt_nap))
        base_loss, parts = vocoder_losses.compute_arcfac_inverter_loss(
          y_b,
          x_b,
          tgt_nap,
          healthy_ear_fn,
          lookahead=lookahead,
          mask_samples=mask_samples,
          output_scale=output_scale,
          lam=lam_b,
          warmup_weight=warmup_weight,
          stft_weight=stft_b,
          apply_stft=apply_stft,
          sai_weight=sai_weight,
          stft_resolutions=stft_res,
        )
        real_seg = x_delayed[:, gan_start:] / clip_norm_rms
        fake_seg = y_b[:, gan_start:] / clip_norm_rms
        real_m = cast(Any, mpd.apply(new_dp["mpd"], real_seg))
        fake_m = cast(Any, mpd.apply(new_dp["mpd"], fake_seg))
        real_r = cast(Any, mrd.apply(new_dp["mrd"], real_seg))
        fake_r = cast(Any, mrd.apply(new_dp["mrd"], fake_seg))
        adv = vocoder_losses.generator_loss(
          fake_m, mask=healthy_mask
        ) + vocoder_losses.generator_loss(fake_r, mask=healthy_mask)
        fm = vocoder_losses.feature_matching_loss(
          real_m, fake_m, mask=healthy_mask
        ) + vocoder_losses.feature_matching_loss(
          real_r, fake_r, mask=healthy_mask
        )
        total = base_loss + gan_weight * (adv + fm_weight * fm)
        return total, {**parts, "adv": adv, "fm": fm, "d_loss": d_val}

      zero_g_grads = jax.tree.map(jnp.zeros_like, gp)
      zero_g_metrics = {
        name: jnp.zeros((), dtype=jnp.float32)
        for name in (
          "loss",
          "nap_l1",
          "sai_l1",
          "l1_waveform",
          "warmup_energy",
          "stft",
          "adv",
          "fm",
          "d_loss",
        )
      }

      def _g_accum_body(
        carry_g: tuple[Mapping[str, Any], dict[str, jnp.ndarray]],
        item: tuple[
          tuple[
            jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
          ],
          tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        ],
      ) -> tuple[tuple[Mapping[str, Any], dict[str, jnp.ndarray]], None]:
        gg_acc, gm_acc = carry_g
        (
          (x_b, _, _, lam_b, stft_b),
          (
            tgt_nap,
            x_delayed,
            clip_norm_rms,
            healthy_mask,
          ),
        ) = item
        (g_val_i, g_parts_i), g_grad_i = jax.value_and_grad(
          _g_loss_micro, has_aux=True
        )(
          gp,
          x_b,
          tgt_nap,
          lam_b,
          stft_b,
          x_delayed,
          clip_norm_rms,
          healthy_mask,
        )
        step_gm = {"loss": g_val_i, **g_parts_i}
        return (
          jax.tree.map(jnp.add, gg_acc, g_grad_i),
          {k: gm_acc[k] + step_gm[k] for k in gm_acc},
        ), None

      (g_grads_sum, g_metrics_sum), _ = jax.lax.scan(
        _g_accum_body, (zero_g_grads, zero_g_metrics), (micros, cached)
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
      return (out_gp, out_dp, out_gos, out_dos), {
        **g_avg_m,
        "finite": fin.astype(jnp.float32),
      }

    gan_scan_k = min(steps_per_loop, gan_steps)
    gan_k_jit = jax.jit(
      lambda st, stk: jax.lax.scan(_gan_step, st, stk),
      donate_argnums=(0,),
    )
    gan_state = (params, d_params, g_opt, d_opt)
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
          f"[Phase 4 GAN {g_step:4d}/{gan_steps}] loss={last_metrics['loss']:.5g} "
          f"nap={last_metrics['nap_l1']:.5g} adv={last_metrics['adv']:.4g} "
          f"fm={last_metrics['fm']:.4g} d={last_metrics['d_loss']:.4g} ({dt:.2f}s)",
          flush=True,
        )
    params = gan_state[0]
    gan_params_host = tpu.to_host_tree(params)
    tpu.delete_tree(gan_state[1:])
    gan_ckpt = out_root / "checkpoint_gan.npz"
    nap_inverter.save_direct_checkpoint(gan_ckpt, gan_params_host, cfg)
    gan_scores = _evaluate_phase4(
      apply_fn=apply_jit,
      params=params,
      eval_clips=eval_clips,
      healthy_cochlea=healthy_dev,
      impaired_cochlea=impaired_dev,
      sample_rate_hz=sample_rate_hz,
      lookahead=lookahead,
      mask_samples=mask_samples,
      out_root=out_root,
      step_idx=steps + gan_steps,
      mesh=mesh,
    )
    print(f"[Phase 4 GAN eval] {json.dumps(gan_scores)}", flush=True)
    if np.isfinite(gan_scores["selection_loss"]) and (
      steps == 0
      or not np.isfinite(best_sel_loss)
      or gan_scores["selection_loss"] <= best_sel_loss
    ):
      best_sel_loss = float(gan_scores["selection_loss"])
      best_scores = gan_scores
      best_params_host = gan_params_host
      nap_inverter.save_direct_checkpoint(ckpt_path, best_params_host, cfg)

  return {
    "checkpoint_path": str(ckpt_path),
    "best_scores": best_scores,
    "last_metrics": last_metrics,
    "params": int(n_params),
  }


def main(argv: Sequence[str]) -> None:
  """CLI entrypoint for Phase 4 training."""
  del argv
  if not _OUTPUT_DIR.value:
    raise app.UsageError("--output_dir is required.")
  run_phase4_training(
    output_dir=_OUTPUT_DIR.value,
    data_dir=_DATA_DIR.value,
    pool_path=_POOL.value,
    init_checkpoint=_INIT_CHECKPOINT.value,
    architecture=_ARCHITECTURE.value,
    width=_WIDTH.value,
    num_blocks=_NUM_BLOCKS.value,
    kernel_size=_KERNEL_SIZE.value,
    coarse_dim=_COARSE_DIM.value,
    coarse_hop=_COARSE_HOP.value,
    lookahead=_LOOKAHEAD.value,
    sample_rate_hz=_SAMPLE_RATE_HZ.value,
    seconds=_SECONDS.value,
    mask_seconds=_MASK_SECONDS.value,
    output_highpass_hz=_OUTPUT_HIGHPASS_HZ.value,
    warmup_weight=_WARMUP_WEIGHT.value,
    lam=_LAM.value,
    lam_impaired=_LAM_IMPAIRED.value,
    stft_weight=_STFT_WEIGHT.value,
    sai_weight=_SAI_WEIGHT.value,
    clip_norm=_CLIP_NORM.value,
    lr=_LR.value,
    lr_schedule=_LR_SCHEDULE.value,
    steps=_STEPS.value,
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
