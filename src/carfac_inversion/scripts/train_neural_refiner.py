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
import math
import pathlib
import time
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app
from absl import flags
from hp_acoustic import audio_constants
from hp_acoustic.losses import waveform_losses
from hp_acoustic.models import carfac

from carfac_inversion import neural

FLAGS = flags.FLAGS

_TRAIN_CACHE: flags.FlagHolder[str | None]
_VAL_CACHE: flags.FlagHolder[str | None]
_OUTPUT_CHECKPOINT: flags.FlagHolder[str | None]
_STAGE1_STEPS: flags.FlagHolder[int]
_STAGE2_STEPS: flags.FlagHolder[int]
_BATCH_SIZE: flags.FlagHolder[int]
_STAGE2_BATCH_SIZE: flags.FlagHolder[int]
_STAGE1_LR: flags.FlagHolder[float]
_STAGE2_LR: flags.FlagHolder[float]
_EVAL_EVERY: flags.FlagHolder[int]
_CARFAC_WEIGHT: flags.FlagHolder[float]
_SEED: flags.FlagHolder[int]

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
  eval_batch_fn,
  ws_wavs: np.ndarray,
  tgt_wavs: np.ndarray,
  canon_tgt_naps: np.ndarray,
  canon_ws_naps: np.ndarray,
  raw_tgt_naps: np.ndarray,
  params: dict,
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
    ws_b = ws_wavs[start:end]
    tgt_b = tgt_wavs[start:end]
    tc_b = canon_tgt_naps[start:end]
    wc_b = canon_ws_naps[start:end]
    tr_b = raw_tgt_naps[start:end]
    metrics = eval_batch_fn(
      params,
      jnp.asarray(ws_b),
      jnp.asarray(tc_b),
      jnp.asarray(wc_b),
      jnp.asarray(tgt_b),
      jnp.asarray(tr_b),
    )
    for key, val in metrics.items():
      sums[key] = sums.get(key, 0.0) + float(val) * k
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
) -> TrainingResult:
  """Trains a Phase 3 :class:`NapResidualRefiner` in two stages."""
  t0 = time.perf_counter()
  cfg = neural.NeuralRefinerConfig() if model_config is None else model_config
  model = neural.NapResidualRefiner(config=cfg)
  fs = int(round(cfg.sample_rate_hz))
  cochlea_model = carfac.build(fs=fs)

  with np.load(train_cache) as d_tr:
    tr_tgt_w = np.asarray(d_tr["target_wavs"], dtype=np.float32)
    tr_ws_w = np.asarray(d_tr["ws_wavs"], dtype=np.float32)
  with np.load(val_cache) as d_va:
    va_tgt_w = np.asarray(d_va["target_wavs"], dtype=np.float32)
    va_ws_w = np.asarray(d_va["ws_wavs"], dtype=np.float32)

  n_train, t_len = tr_tgt_w.shape
  rest_nap = np.asarray(
    cochlea_model(jnp.zeros((min(512, t_len), 1), dtype=jnp.float32))[-1, :, 0],
    dtype=np.float32,
  )
  proj_w = neural.cochlea_cf_projection(cochlea_model, cfg)
  cfs = np.asarray(cochlea_model.channel_frequencies, dtype=np.float32)
  hf_chan_mask = jnp.asarray((cfs >= 3500.0).astype(np.float32))
  rest_nap_j = jnp.asarray(rest_nap, dtype=jnp.float32)
  fixed_cochlea = jax.tree_util.tree_map(jax.lax.stop_gradient, cochlea_model)

  print(
    f"Precomputing cochlear features ({n_train} train, {len(va_tgt_w)} val)..."
  )
  tr_raw_nap, tr_can_tgt, tr_can_ws = _precompute_cochlear_features(
    tr_tgt_w, tr_ws_w, fixed_cochlea, proj_w, rest_nap
  )
  va_raw_nap, va_can_tgt, va_can_ws = _precompute_cochlear_features(
    va_tgt_w, va_ws_w, fixed_cochlea, proj_w, rest_nap
  )
  print(f"Feature precompute done in {time.perf_counter() - t0:.1f}s")

  rng = jax.random.PRNGKey(seed)
  np_rng = np.random.default_rng(seed)
  params = model.init(
    rng,
    jnp.asarray(tr_ws_w[:batch_size]),
    jnp.asarray(tr_can_tgt[:batch_size]),
    jnp.asarray(tr_can_ws[:batch_size]),
  )
  n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
  print(f"Initialized NapResidualRefiner with {n_params:,} parameters")

  @functools.partial(jax.jit, static_argnames=("use_carfac",))
  def eval_step(
    p: dict,
    ws_b: jnp.ndarray,
    tc_b: jnp.ndarray,
    wc_b: jnp.ndarray,
    tgt_b: jnp.ndarray,
    tr_b: jnp.ndarray,
    *,
    use_carfac: bool,
  ) -> dict[str, jnp.ndarray]:
    pred = model.apply(p, ws_b, tc_b, wc_b)
    _, metrics = compute_refiner_losses(
      pred,
      tgt_b,
      sample_rate_hz=fs,
      cochlea_model=fixed_cochlea if use_carfac else None,
      target_nap_raw=tr_b if use_carfac else None,
      rest_nap_raw=rest_nap_j if use_carfac else None,
      hf_chan_mask=hf_chan_mask if use_carfac else None,
      carfac_weight=carfac_weight if use_carfac else 0.0,
    )
    return metrics

  def make_train_step(tx: optax.GradientTransformation, *, use_carfac: bool):
    @jax.jit
    def step_fn(
      p: dict,
      opt_state: optax.OptState,
      ws_b: jnp.ndarray,
      tc_b: jnp.ndarray,
      wc_b: jnp.ndarray,
      tgt_b: jnp.ndarray,
      tr_b: jnp.ndarray,
    ) -> tuple[dict, optax.OptState, dict[str, jnp.ndarray]]:
      def loss_fn(curr_p: dict) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        pred = model.apply(curr_p, ws_b, tc_b, wc_b)
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

      (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
      updates, next_opt = tx.update(grads, opt_state, p)
      return optax.apply_updates(p, updates), next_opt, metrics

    return step_fn

  def sample_idx(n_items: int, bsz: int) -> np.ndarray:
    replace = n_items < bsz
    return np_rng.choice(n_items, size=bsz, replace=replace)

  out_path = pathlib.Path(output_checkpoint)
  stage1_path = out_path.with_name(f"{out_path.stem}_stage1.npz")
  history: list[dict[str, float]] = []

  # Baseline val evaluation (step 0: should match warm start cleanly).
  val_b = min(batch_size, len(va_tgt_w))
  eval_s1_fn = functools.partial(eval_step, use_carfac=False)
  init_val = _evaluate_split(
    eval_s1_fn,
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
    f"hf={init_val['hf_spec']:.4f}, log_rms={init_val['log_rms']:.4f}"
  )
  best_s1_loss = float(init_val["total"])
  best_s1_sisdr = float(init_val["sisdr_db"])
  best_s1_params = params

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
    opt_state1 = tx1.init(params)
    step1_fn = make_train_step(tx1, use_carfac=False)
    t_s1 = time.perf_counter()

    for step in range(1, stage1_steps + 1):
      idx = sample_idx(n_train, batch_size)
      params, opt_state1, tr_m = step1_fn(
        params,
        opt_state1,
        jnp.asarray(tr_ws_w[idx]),
        jnp.asarray(tr_can_tgt[idx]),
        jnp.asarray(tr_can_ws[idx]),
        jnp.asarray(tr_tgt_w[idx]),
        jnp.asarray(tr_raw_nap[idx]),
      )
      if step % eval_every == 0 or step == stage1_steps:
        val_m = _evaluate_split(
          eval_s1_fn,
          va_ws_w,
          va_tgt_w,
          va_can_tgt,
          va_can_ws,
          va_raw_nap,
          params,
          batch_size=val_b,
        )
        elapsed = time.perf_counter() - t_s1
        print(
          f"[S1 {step:4d}/{stage1_steps} | {elapsed:5.1f}s] "
          f"tr={float(tr_m['total']):.4f} (sdr={float(tr_m['sisdr_db']):.2f}) | "
          f"va={val_m['total']:.4f} sdr={val_m['sisdr_db']:.2f}dB "
          f"stft={val_m['stft']:.4f} hf={val_m['hf_spec']:.4f} "
          f"lrms={val_m['log_rms']:.4f}"
        )
        history.append({"stage": 1.0, "step": float(step), **val_m})
        if math.isfinite(val_m["total"]) and val_m["total"] < best_s1_loss:
          best_s1_loss = float(val_m["total"])
          best_s1_sisdr = float(val_m["sisdr_db"])
          best_s1_params = params

    neural.save_refiner_checkpoint(stage1_path, best_s1_params, cfg)
    print(
      f"Saved Stage 1 best checkpoint -> {stage1_path} (val={best_s1_loss:.4f})"
    )
    params = best_s1_params

  best_s2_loss = best_s1_loss
  best_s2_sisdr = best_s1_sisdr
  best_s2_params = params

  if stage2_steps > 0:
    eval_s2_fn = functools.partial(eval_step, use_carfac=True)
    s2_b = min(stage2_batch_size, len(va_tgt_w))
    s2_init_val = _evaluate_split(
      eval_s2_fn,
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
      f"sdr={s2_init_val['sisdr_db']:.2f}dB nap={s2_init_val['carfac_nap']:.4f}"
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
    opt_state2 = tx2.init(params)
    step2_fn = make_train_step(tx2, use_carfac=True)
    t_s2 = time.perf_counter()
    s2_eval_cadence = max(50, min(eval_every, max(50, stage2_steps // 4)))

    for step in range(1, stage2_steps + 1):
      idx = sample_idx(n_train, stage2_batch_size)
      params, opt_state2, tr_m = step2_fn(
        params,
        opt_state2,
        jnp.asarray(tr_ws_w[idx]),
        jnp.asarray(tr_can_tgt[idx]),
        jnp.asarray(tr_can_ws[idx]),
        jnp.asarray(tr_tgt_w[idx]),
        jnp.asarray(tr_raw_nap[idx]),
      )
      if step % s2_eval_cadence == 0 or step == stage2_steps:
        val_m = _evaluate_split(
          eval_s2_fn,
          va_ws_w,
          va_tgt_w,
          va_can_tgt,
          va_can_ws,
          va_raw_nap,
          params,
          batch_size=s2_b,
        )
        elapsed = time.perf_counter() - t_s2
        print(
          f"[S2 {step:4d}/{stage2_steps} | {elapsed:5.1f}s] "
          f"tr={float(tr_m['total']):.4f} nap={float(tr_m['carfac_nap']):.4f} | "
          f"va={val_m['total']:.4f} sdr={val_m['sisdr_db']:.2f}dB "
          f"nap={val_m['carfac_nap']:.4f} hf={val_m['hf_spec']:.4f} "
          f"lrms={val_m['log_rms']:.4f}"
        )
        history.append({"stage": 2.0, "step": float(step), **val_m})
        if math.isfinite(val_m["total"]) and val_m["total"] < best_s2_loss:
          best_s2_loss = float(val_m["total"])
          best_s2_sisdr = float(val_m["sisdr_db"])
          best_s2_params = params

  neural.save_refiner_checkpoint(out_path, best_s2_params, cfg)
  print(
    f"Saved final checkpoint -> {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)"
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
  run_training(
    train_cache=_TRAIN_CACHE.value,
    val_cache=_VAL_CACHE.value,
    output_checkpoint=_OUTPUT_CHECKPOINT.value,
    stage1_steps=_STAGE1_STEPS.value,
    stage2_steps=_STAGE2_STEPS.value,
    batch_size=_BATCH_SIZE.value,
    stage2_batch_size=_STAGE2_BATCH_SIZE.value,
    stage1_lr=_STAGE1_LR.value,
    stage2_lr=_STAGE2_LR.value,
    eval_every=_EVAL_EVERY.value,
    carfac_weight=_CARFAC_WEIGHT.value,
    seed=_SEED.value,
  )


if __name__ == "__main__":
  define_flags()
  app.run(main)
