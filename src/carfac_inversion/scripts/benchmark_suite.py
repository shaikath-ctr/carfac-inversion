"""Real-speech diagnostic benchmark and listening-artifact generator.

Supports raw WAV/FLAC directories, prepared ``hp-acoustic`` ArrayRecord
corpora (``ProcessedClips``), optional LibriSpeech ``dev-clean`` download via
``hp_acoustic.scripts.data.prepare_librispeech``, or deterministic synthetic
speech when ``--input_dir`` is omitted.
"""

import csv
import importlib
import math
import pathlib
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from absl import app
from absl import flags
from hp_acoustic.data import records
from hp_acoustic.data import transforms
from hp_acoustic.io import audio_io
from hp_acoustic.io import resampling
from hp_acoustic.metrics import context as metric_context
from hp_acoustic.metrics import perceptual
from hp_acoustic.metrics import sii
from hp_acoustic.metrics import waveform as waveform_metrics
from hp_acoustic.models import carfac
from hp_acoustic.models import sai as sai_lib
from jax_haspi_hasqi import hasqi as hasqi_lib

from carfac_inversion import inversion


def _load_optional_module(name: str) -> Any | None:
  """Imports an optional baseline dependency, returning None if absent."""
  try:
    return importlib.import_module(name)
  except ImportError:
    return None


FLAGS = flags.FLAGS

_INPUT_DIR: flags.FlagHolder[str | None]
_OUTPUT_DIR: flags.FlagHolder[str]
_SPLIT: flags.FlagHolder[str]
_MAX_CLIPS: flags.FlagHolder[int]
_CLIP_SECONDS: flags.FlagHolder[float]
_SAMPLE_RATES: flags.FlagHolder[list[str]]
_LEVELS_DB_SPL: flags.FlagHolder[list[str]]
_SNRS_DB: flags.FlagHolder[list[str]]
_STEP_BUDGETS: flags.FlagHolder[list[str]]
_ENHANCERS: flags.FlagHolder[list[str]]
_CHUNK_SECONDS: flags.FlagHolder[float]
_SEED: flags.FlagHolder[int]
_PREPARE_LIBRISPEECH_ARCHIVE: flags.FlagHolder[str | None]
_NEURAL_CHECKPOINT: flags.FlagHolder[str | None]

_BANDS_HZ: tuple[tuple[int, int], ...] = (
  (60, 300),
  (300, 1000),
  (1000, 2200),
  (2200, 4500),
  (4500, 7500),
)


def define_flags(flag_values: flags.FlagValues = FLAGS) -> None:
  """Registers benchmark CLI flags on ``flag_values``."""
  global _INPUT_DIR
  global _OUTPUT_DIR
  global _SPLIT
  global _MAX_CLIPS
  global _CLIP_SECONDS
  global _SAMPLE_RATES
  global _LEVELS_DB_SPL
  global _SNRS_DB
  global _STEP_BUDGETS
  global _ENHANCERS
  global _CHUNK_SECONDS
  global _SEED
  global _PREPARE_LIBRISPEECH_ARCHIVE
  global _NEURAL_CHECKPOINT

  _INPUT_DIR = flags.DEFINE_string(
    "input_dir",
    None,
    "Directory of raw WAV/FLAC files or prepared ProcessedClips shards.",
    flag_values=flag_values,
  )
  _OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "tmp/carfac_inversion_benchmark",
    "Output directory for metrics.csv, WAV pairs, and diagnostic charts.",
    flag_values=flag_values,
  )
  _SPLIT = flags.DEFINE_string(
    "split",
    "test",
    "Dataset split when reading a ProcessedClips directory.",
    flag_values=flag_values,
  )
  _MAX_CLIPS = flags.DEFINE_integer(
    "max_clips",
    3,
    "Maximum number of speech clips to evaluate.",
    flag_values=flag_values,
  )
  _CLIP_SECONDS = flags.DEFINE_float(
    "clip_seconds",
    2.0,
    "Target clip duration in seconds (cropped or zero-padded if shorter).",
    flag_values=flag_values,
  )
  _SAMPLE_RATES = flags.DEFINE_list(
    "sample_rates",
    ["16000"],
    "Comma-separated sample rates in Hz (e.g. 16000,32000).",
    flag_values=flag_values,
  )
  _LEVELS_DB_SPL = flags.DEFINE_list(
    "levels_db_spl",
    ["50", "65", "80"],
    "Comma-separated presentation levels in dB SPL.",
    flag_values=flag_values,
  )
  _SNRS_DB = flags.DEFINE_list(
    "snrs_db",
    ["clean", "10", "0"],
    "Comma-separated noise SNRs in dB ('clean' or float).",
    flag_values=flag_values,
  )
  _STEP_BUDGETS = flags.DEFINE_list(
    "step_budgets",
    ["0+0", "20+60", "45+135"],
    "Comma-separated adam+lbfgs step budgets for the Pareto curve.",
    flag_values=flag_values,
  )
  _ENHANCERS = flags.DEFINE_list(
    "enhancers",
    [],
    "Optional enhancers to include in auralization (e.g. shaca,viking,seanet).",
    flag_values=flag_values,
  )
  _CHUNK_SECONDS = flags.DEFINE_float(
    "chunk_seconds",
    1.5,
    "Fixed OLA optimization window duration in seconds (<=0 disables OLA).",
    flag_values=flag_values,
  )
  _SEED = flags.DEFINE_integer(
    "seed",
    0,
    "Random seed for noise synthesis and warm-start HF carriers.",
    flag_values=flag_values,
  )
  _PREPARE_LIBRISPEECH_ARCHIVE = flags.DEFINE_string(
    "prepare_librispeech_archive",
    None,
    "Optional OpenSLR archive name (e.g. 'dev-clean') to prepare into "
    "--input_dir before running the benchmark.",
    flag_values=flag_values,
  )
  _NEURAL_CHECKPOINT = flags.DEFINE_string(
    "neural_checkpoint",
    None,
    "Optional path to a trained Phase 3 neural residual refiner .npz.",
    flag_values=flag_values,
  )


def _synthetic_utterance(sample_rate_hz: int, duration_s: float) -> np.ndarray:
  """Generates a fallback voiced+fricative utterance for offline smoke runs."""
  n = int(round(sample_rate_hz * duration_s))
  t = np.arange(n, dtype=np.float64) / sample_rate_hz
  f0 = 120.0 + 30.0 * np.sin(2.0 * np.pi * 1.8 * t)
  phase = np.cumsum(2.0 * np.pi * f0 / sample_rate_hz)
  voiced_env = np.exp(
    -0.5 * ((t - 0.35 * duration_s) / (0.18 * duration_s)) ** 2
  )
  f1 = 700.0 - 340.0 * np.clip(t / (0.6 * duration_s), 0.0, 1.0)
  f2 = 1150.0 + 1000.0 * np.clip(t / (0.6 * duration_s), 0.0, 1.0)
  voiced = np.zeros_like(t)
  for k in range(1, 34):
    fk = k * f0
    g = 1.0 / (1.0 + ((fk - f1) / 65.0) ** 2) + 0.7 / (
      1.0 + ((fk - f2) / 90.0) ** 2
    )
    voiced += g * (fk < 0.45 * sample_rate_hz) * np.sin(k * phase)
  voiced *= voiced_env
  rng = np.random.default_rng(7)
  freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
  band = (freqs >= 3500.0) & (freqs <= min(6800.0, 0.45 * sample_rate_hz))
  fric = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * band, n=n)
  fric = fric / (np.std(fric) + 1e-12)
  fric *= 0.45 * np.exp(
    -0.5 * ((t - 0.78 * duration_s) / (0.06 * duration_s)) ** 2
  )
  sig = voiced + fric
  ramp = max(1, int(0.005 * sample_rate_hz))
  taper = np.sin(0.5 * np.pi * np.arange(ramp) / ramp) ** 2
  sig[:ramp] *= taper
  sig[-ramp:] *= taper[::-1]
  return sig.astype(np.float32)


def load_speech_clips(
  input_dir: str | None,
  *,
  sample_rate_hz: int,
  max_clips: int,
  clip_seconds: float,
  split: str = "test",
) -> list[tuple[str, np.ndarray]]:
  """Loads mono float32 clips from ProcessedClips shards, WAV/FLAC dir, or synth."""
  target_len = int(round(sample_rate_hz * clip_seconds))
  loaded: list[tuple[str, np.ndarray]] = []

  if input_dir is None:
    for idx in range(max(1, max_clips)):
      audio = _synthetic_utterance(sample_rate_hz, clip_seconds)
      loaded.append((f"synth_{idx:02d}", audio[:target_len]))
    return loaded

  root = pathlib.Path(input_dir)
  if not root.exists():
    raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

  split_dir = root / split
  if split_dir.is_dir() and any(split_dir.glob(f"{records.SHARD_STEM}-*")):
    clips = records.open_clips(str(split_dir))
    for idx in range(min(max_clips, len(clips))):
      clip = clips[idx]
      audio = np.asarray(clip.audio, dtype=np.float32).squeeze()
      if clip.sample_rate != sample_rate_hz:
        audio = np.asarray(
          resampling.resample(audio, clip.sample_rate, sample_rate_hz),
          dtype=np.float32,
        )
      cid = (clip.clip_id or f"shard_{idx:03d}").replace("/", "_")
      loaded.append((cid, _fit_length(audio, target_len)))
    return loaded

  audio_paths = sorted(
    p
    for p in root.rglob("*")
    if p.is_file() and p.suffix.lower() in (".wav", ".flac")
  )
  if not audio_paths:
    raise ValueError(
      f"No ProcessedClips shards or .wav/.flac files found under {input_dir}"
    )
  for p in audio_paths[:max_clips]:
    audio, _ = audio_io.read_wav(str(p), target_sr=sample_rate_hz)
    loaded.append((p.stem, _fit_length(audio, target_len)))
  return loaded


def _fit_length(audio: np.ndarray, target_len: int) -> np.ndarray:
  """Crops or zero-pads mono audio to ``target_len`` samples."""
  if len(audio) >= target_len:
    return np.ascontiguousarray(audio[:target_len], dtype=np.float32)
  out = np.zeros(target_len, dtype=np.float32)
  out[: len(audio)] = audio
  return out


def _scale_to_spl(audio: np.ndarray, level_db_spl: float) -> np.ndarray:
  """Scales audio RMS to ``level_db_spl`` under hp-acoustic calibration."""
  rms = float(np.sqrt(np.mean(audio**2)))
  if rms <= 1e-12:
    return audio.astype(np.float32)
  target_rms = 10.0 ** ((level_db_spl - sii.CARFAC_RMS_ONE_DB_SPL) / 20.0)
  return (audio * (target_rms / rms)).astype(np.float32)


def _sloping_ohc_health(cochlea_model: carfac.CarfacCochleaModel) -> np.ndarray:
  """Constructs a sloping high-frequency OHC loss profile (0.95 -> 0.20)."""
  cfs = np.asarray(cochlea_model.channel_frequencies, dtype=np.float32)
  log_lo = math.log10(250.0)
  log_hi = math.log10(6000.0)
  frac = np.clip(
    (np.log10(np.maximum(cfs, 20.0)) - log_lo) / (log_hi - log_lo), 0.0, 1.0
  )
  return (0.95 - 0.75 * frac).astype(np.float32)


def _run_optional_enhancer(
  enhancer_name: str, noisy: np.ndarray, sample_rate_hz: int
) -> np.ndarray | None:
  """Runs an optional baseline enhancer (SHACA, Viking, SEANet) if installed."""
  n_orig = len(noisy)
  key = jax.random.key(0)
  if enhancer_name == "shaca":
    shaca_mod = _load_optional_module("shaca_jax")
    shaca_cfg_mod = _load_optional_module("shaca_jax.config")
    if shaca_mod is None or shaca_cfg_mod is None:
      return None
    frame_len = int(shaca_cfg_mod.DEFAULT_FRAME_LEN)
    padded = np.pad(noisy, (0, (-n_orig) % frame_len)).astype(np.float32)
    x_1d = jnp.asarray(padded, dtype=jnp.float32)
    params, initial_state = shaca_mod.shaca_setup_jax(
      sample_rate_hz=float(sample_rate_hz), forward_mode="simd"
    )
    sc_powers = shaca_mod.precompute_sidechain_powers_jax(
      x_1d, fs=float(sample_rate_hz), frame_len=frame_len
    )
    pwr_smooth = shaca_mod.smooth_sidechain_powers_jax(
      sc_powers, params["alpha_pwr"]
    )
    _, aided = shaca_mod.shaca_sample_rate_loop_jax(
      params, initial_state, x_1d, pwr_smooth
    )
    return np.asarray(aided[:n_orig], dtype=np.float32)
  if enhancer_name == "viking":
    viking_mod = _load_optional_module("viking_jax.viking_jax")
    if viking_mod is None:
      return None
    padded = np.pad(noisy, (0, (-n_orig) % 32)).astype(np.float32)
    x = jnp.asarray(padded[np.newaxis, :, np.newaxis], dtype=jnp.float32)
    model = viking_mod.VikingModel(
      module=viking_mod.VikingGenerator(
        use_low_latency_filterbanks=True,
        convert_to_min_phase=True,
        shift_factor=32,
        num_subbands=256,
        prototype_filter_length=256,
        num_taps=128,
        init_mask_gain=1.8,
      )
    )
    params = model.init(key)
    y = model.apply(params, x)
    return np.asarray(y[0, :n_orig, 0], dtype=np.float32)
  if enhancer_name == "seanet":
    seanet_mod = _load_optional_module("seanet.models.seanet")
    if seanet_mod is None:
      return None
    padded = np.pad(noisy, (0, (-n_orig) % 256)).astype(np.float32)
    x = jnp.asarray(padded[np.newaxis, :, np.newaxis], dtype=jnp.float32)
    model = seanet_mod.SEANetModel()
    params = model.init(key)
    y = model.apply(params, x)
    return np.asarray(y[0, :n_orig, 0], dtype=np.float32)
  raise ValueError(f"Unsupported enhancer {enhancer_name!r}")


def _band_snrs_db(
  ref: np.ndarray, est: np.ndarray, sample_rate_hz: int
) -> dict[str, float]:
  """Computes per-frequency-band SNRs in dB."""
  freqs = np.fft.rfftfreq(len(ref), d=1.0 / sample_rate_hz)
  spec_r = np.fft.rfft(ref)
  spec_e = np.fft.rfft(est)
  out: dict[str, float] = {}
  for lo, hi in _BANDS_HZ:
    if hi >= 0.5 * sample_rate_hz:
      continue
    mask = (freqs >= lo) & (freqs < hi)
    p_r = float(np.sum(np.abs(spec_r[mask]) ** 2)) + 1e-18
    p_err = float(np.sum(np.abs(spec_r[mask] - spec_e[mask]) ** 2)) + 1e-18
    out[f"snr_{lo}_{hi}_hz"] = float(10.0 * math.log10(p_r / p_err))
  return out


def _score_pair(
  *,
  ref: np.ndarray,
  est: np.ndarray,
  target_nap: np.ndarray,
  recon_nap: np.ndarray,
  sample_rate_hz: int,
  hearing_loss_db: np.ndarray,
) -> dict[str, float]:
  """Scores waveform, perceptual, and auditory-domain fidelity."""
  common = min(len(ref), len(est))
  ref_c = ref[:common]
  est_c = est[:common]
  ctx = metric_context.Context(
    reference=ref_c,
    estimation=est_c,
    sample_rate=sample_rate_hz,
    hearing_loss_db=hearing_loss_db,
  )
  haspi_val = float("nan")
  hasqi_q = float("nan")
  hasqi_nl = float("nan")
  hasqi_lin = float("nan")
  try:
    haspi_val = perceptual.haspi(ctx)
  except ValueError:
    pass
  if np.any(ctx.reference):
    try:
      q, nl, lin, _ = hasqi_lib.hasqi_v2(
        ref_c,
        sample_rate_hz,
        est_c,
        sample_rate_hz,
        hearing_loss_db,
        level1=sii.CARFAC_RMS_ONE_DB_SPL,
      )
      hasqi_nl = float(nl)
      hasqi_lin = float(lin)
      if hasqi_nl != 0.0:
        hasqi_q = float(q)
    except ValueError:
      pass

  freqs = np.fft.rfftfreq(len(target_nap), d=1.0 / sample_rate_hz)
  lpf = (1.0 / (1.0 + (freqs / 200.0) ** 4))[:, None]
  env_t = np.fft.irfft(
    np.fft.rfft(np.maximum(target_nap, 0.0), axis=0) * lpf,
    n=len(target_nap),
    axis=0,
  )
  env_r = np.fft.irfft(
    np.fft.rfft(np.maximum(recon_nap, 0.0), axis=0) * lpf,
    n=len(recon_nap),
    axis=0,
  )
  sai_len = min(2048, 2 ** int(math.floor(math.log2(len(target_nap)))))
  sai_corr = float("nan")
  if sai_len >= 64:
    sai_h = sai_lib.SaiHypers(fft_len=sai_len)
    s_t = sai_lib.from_naps(jnp.asarray(target_nap[None, :, :]), sai_h)
    s_r = sai_lib.from_naps(jnp.asarray(recon_nap[None, :, :]), sai_h)
    sai_corr = float(
      np.corrcoef(np.asarray(s_t).ravel(), np.asarray(s_r).ravel())[0, 1]
    )

  row: dict[str, float] = {
    "si_sdr": waveform_metrics.si_sdr(ctx),
    "snr": waveform_metrics.snr(ctx),
    "stoi": perceptual.stoi(ctx),
    "haspi": haspi_val,
    "hasqi": hasqi_q,
    "hasqi_nl": hasqi_nl,
    "hasqi_lin": hasqi_lin,
    "nap_corr": float(np.corrcoef(target_nap.ravel(), recon_nap.ravel())[0, 1]),
    "env_corr": float(np.corrcoef(env_t.ravel(), env_r.ravel())[0, 1]),
    "sai_corr": sai_corr,
  }
  row.update(_band_snrs_db(ref_c, est_c, sample_rate_hz))
  return row


def _parse_step_budget(spec: str) -> tuple[int, int]:
  """Parses '<adam>+<lbfgs>' step budget strings."""
  parts = spec.strip().split("+")
  if len(parts) != 2:
    raise ValueError(
      f"Step budget must have form '<adam>+<lbfgs>', got {spec!r}"
    )
  return int(parts[0]), int(parts[1])


def _nanmean(values: Sequence[float]) -> float:
  """Computes the mean of finite values without empty-slice warnings."""
  finite = [float(v) for v in values if math.isfinite(v)]
  return float(np.mean(finite)) if finite else float("nan")


def _plot_summary_charts(
  rows: list[dict[str, Any]], charts_dir: pathlib.Path
) -> None:
  """Writes diagnostic PNG plots summarizing level, band, and Pareto sweeps."""
  charts_dir.mkdir(parents=True, exist_ok=True)
  rt_rows = [r for r in rows if r["scenario"] == "roundtrip"]
  if rt_rows:
    levels = sorted({float(r["level_db_spl"]) for r in rt_rows})
    si_sdrs = [
      _nanmean([r["si_sdr"] for r in rt_rows if float(r["level_db_spl"]) == lv])
      for lv in levels
    ]
    stois = [
      _nanmean([r["stoi"] for r in rt_rows if float(r["level_db_spl"]) == lv])
      for lv in levels
    ]
    hasqis = [
      _nanmean([r["hasqi"] for r in rt_rows if float(r["level_db_spl"]) == lv])
      for lv in levels
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(levels, si_sdrs, marker="o", label="SI-SDR (dB)")
    axes[0].set_title("Round-Trip SI-SDR vs. Presentation Level")
    axes[0].set_xlabel("Level (dB SPL)")
    axes[0].set_ylabel("SI-SDR (dB)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(levels, stois, marker="o", label="STOI")
    axes[1].plot(levels, hasqis, marker="s", label="HASQI")
    axes[1].set_title("Perceptual Metrics vs. Presentation Level")
    axes[1].set_xlabel("Level (dB SPL)")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "roundtrip_by_level.png", dpi=200)
    plt.close(fig)

  pareto_rows = [r for r in rows if r["scenario"] == "pareto"]
  if pareto_rows:
    budgets = sorted(
      {str(r["label"]) for r in pareto_rows},
      key=lambda s: _nanmean(
        [r["runtime_s"] for r in pareto_rows if str(r["label"]) == s]
      ),
    )
    times = [
      _nanmean([r["runtime_s"] for r in pareto_rows if str(r["label"]) == b])
      for b in budgets
    ]
    sdrs = [
      _nanmean([r["si_sdr"] for r in pareto_rows if str(r["label"]) == b])
      for b in budgets
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, sdrs, marker="o")
    for x_t, y_s, b in zip(times, sdrs, budgets):
      ax.annotate(b, (x_t, y_s), textcoords="offset points", xytext=(6, 4))
    ax.set_title("Step-Budget Pareto Curve (SI-SDR vs. Runtime)")
    ax.set_xlabel("Runtime per clip (s)")
    ax.set_ylabel("SI-SDR (dB)")
    ax.margins(x=0.14, y=0.14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "pareto_quality_vs_runtime.png", dpi=200)
    plt.close(fig)

  aur_rows = [r for r in rows if r["scenario"] == "auralization"]
  if aur_rows:
    labels = sorted({str(r["label"]) for r in aur_rows})
    env_means = [
      _nanmean([r["env_corr"] for r in aur_rows if str(r["label"]) == lb])
      for lb in labels
    ]
    sai_means = [
      _nanmean([r["sai_corr"] for r in aur_rows if str(r["label"]) == lb])
      for lb in labels
    ]
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x_pos - 0.18, env_means, width=0.35, label="Envelope corr")
    ax.bar(x_pos + 0.18, sai_means, width=0.35, label="SAI corr")
    ax.set_xticks(x_pos, labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.15)
    ax.set_title("Cross-Cochlea Auralization Alignment")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "auralization_comparison.png", dpi=200)
    plt.close(fig)


def run_benchmark(
  *,
  input_dir: str | None,
  output_dir: str,
  split: str = "test",
  max_clips: int = 3,
  clip_seconds: float = 2.0,
  sample_rates: Sequence[int] = (16000,),
  levels_db_spl: Sequence[float] = (50.0, 65.0, 80.0),
  snrs_db: Sequence[float | None] = (None, 10.0, 0.0),
  step_budgets: Sequence[tuple[int, int]] = ((0, 0), (20, 60), (45, 135)),
  enhancers: Sequence[str] = (),
  chunk_seconds: float | None = 1.5,
  neural_checkpoint: str | None = None,
  seed: int = 0,
) -> list[dict[str, Any]]:
  """Executes the full real-speech round-trip, Pareto, and auralization suite."""
  out_root = pathlib.Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(seed)
  rows: list[dict[str, Any]] = []

  for fs in sample_rates:
    clips = load_speech_clips(
      input_dir,
      sample_rate_hz=fs,
      max_clips=max_clips,
      clip_seconds=clip_seconds,
      split=split,
    )
    healthy = carfac.build(fs=fs)
    impaired_half = carfac.build(fs=fs, ohc_health=0.5)
    impaired_sloping = carfac.build(
      fs=fs, ohc_health=_sloping_ohc_health(healthy)
    )

    for clip_id, raw_clean in clips:
      clip_audio_dir = out_root / "audio" / f"{fs}hz_{clip_id}"
      noise = rng.standard_normal(len(raw_clean)).astype(np.float32)

      # 1. Round-trip sweep across levels and SNRs using default budget.
      default_adam, default_lbfgs = step_budgets[-1]
      for snr_db in snrs_db:
        if snr_db is None:
          sig_base = raw_clean
          snr_tag = "clean"
        else:
          sig_base = transforms.mix_at_snr(raw_clean, noise, snr_db)
          snr_tag = f"snr{int(snr_db)}"
        for lvl in levels_db_spl:
          if snr_db is not None and lvl != 65.0:
            continue
          sig = _scale_to_spl(sig_base, lvl)
          audio_io.save_audio_to_file(
            sig, str(clip_audio_dir / f"input_{snr_tag}_{int(lvl)}dB.wav"), fs
          )
          nap = np.asarray(healthy(jnp.asarray(sig[:, None]))[:, :, 0])
          cfg = inversion.InversionConfig(
            mode="exact",
            adam_steps=default_adam,
            lbfgs_steps=default_lbfgs,
            chunk_seconds=chunk_seconds,
            neural_checkpoint=neural_checkpoint,
            seed=seed,
          )
          t0 = time.perf_counter()
          res = inversion.invert_nap(nap, healthy, cfg)
          dt = time.perf_counter() - t0
          audio_io.save_audio_to_file(
            res.waveform,
            str(clip_audio_dir / f"roundtrip_{snr_tag}_{int(lvl)}dB.wav"),
            fs,
          )
          metrics = _score_pair(
            ref=sig,
            est=res.waveform,
            target_nap=nap,
            recon_nap=res.reconstructed_nap,
            sample_rate_hz=fs,
            hearing_loss_db=np.zeros(6),
          )
          rows.append(
            {
              "scenario": "roundtrip",
              "label": f"{snr_tag}_{int(lvl)}dB",
              "clip_id": clip_id,
              "sample_rate_hz": fs,
              "level_db_spl": lvl,
              "snr_db": snr_tag,
              "mode": "exact",
              "runtime_s": dt,
              **metrics,
            }
          )

      # 2. Step-budget Pareto curve at 65 dB SPL clean speech.
      sig65 = _scale_to_spl(raw_clean, 65.0)
      nap65 = np.asarray(healthy(jnp.asarray(sig65[:, None]))[:, :, 0])
      for adam_s, lbfgs_s in step_budgets:
        cfg = inversion.InversionConfig(
          mode="exact",
          adam_steps=adam_s,
          lbfgs_steps=lbfgs_s,
          chunk_seconds=chunk_seconds,
          neural_checkpoint=neural_checkpoint,
          seed=seed,
        )
        t0 = time.perf_counter()
        res = inversion.invert_nap(nap65, healthy, cfg)
        dt = time.perf_counter() - t0
        metrics = _score_pair(
          ref=sig65,
          est=res.waveform,
          target_nap=nap65,
          recon_nap=res.reconstructed_nap,
          sample_rate_hz=fs,
          hearing_loss_db=np.zeros(6),
        )
        rows.append(
          {
            "scenario": "pareto",
            "label": f"{adam_s}+{lbfgs_s}",
            "clip_id": clip_id,
            "sample_rate_hz": fs,
            "level_db_spl": 65.0,
            "snr_db": "clean",
            "mode": "exact",
            "runtime_s": dt,
            **metrics,
          }
        )

      # 3. Cross-cochlea auralization + perceptual-loss ablation (10 dB noisy).
      noisy10 = _scale_to_spl(
        transforms.mix_at_snr(raw_clean, noise, 10.0), 65.0
      )
      signals_to_auralize: list[tuple[str, np.ndarray]] = [("noisy", noisy10)]
      for enh_name in enhancers:
        enh = _run_optional_enhancer(enh_name, noisy10, fs)
        if enh is not None:
          signals_to_auralize.append((enh_name, enh))

      for sig_tag, sig_in in signals_to_auralize:
        audio_io.save_audio_to_file(
          sig_in, str(clip_audio_dir / f"source_{sig_tag}.wav"), fs
        )
        for coch_tag, src_cochlea, hl_arr in (
          (
            "ohc05",
            impaired_half,
            np.asarray(perceptual.HALF_OHC_HEARING_LOSS_DB, dtype=np.float64),
          ),
          ("sloping", impaired_sloping, np.zeros(6)),
        ):
          nap_src = np.asarray(
            src_cochlea(jnp.asarray(sig_in[:, None]))[:, :, 0]
          )
          ablation_cfgs: list[tuple[str, inversion.InversionMode, float]] = [
            (
              f"{coch_tag}_{sig_tag}_perc030",
              "perceptual",
              0.30,
            ),
          ]
          if sig_tag == "noisy" and coch_tag == "ohc05":
            ablation_cfgs.extend(
              [
                (f"{coch_tag}_{sig_tag}_perc000", "perceptual", 0.0),
                (f"{coch_tag}_{sig_tag}_exact", "exact", 0.30),
              ]
            )
          for lbl, mode_val, nap_w in ablation_cfgs:
            cfg = inversion.InversionConfig(
              mode=mode_val,
              perceptual_nap_weight=nap_w,
              adam_steps=default_adam,
              lbfgs_steps=default_lbfgs,
              chunk_seconds=chunk_seconds,
              neural_checkpoint=neural_checkpoint,
              seed=seed,
            )
            t0 = time.perf_counter()
            res = inversion.invert_nap(nap_src, healthy, cfg)
            dt = time.perf_counter() - t0
            audio_io.save_audio_to_file(
              res.waveform, str(clip_audio_dir / f"auralized_{lbl}.wav"), fs
            )
            metrics = _score_pair(
              ref=sig_in,
              est=res.waveform,
              target_nap=nap_src,
              recon_nap=res.reconstructed_nap,
              sample_rate_hz=fs,
              hearing_loss_db=hl_arr,
            )
            rows.append(
              {
                "scenario": "auralization",
                "label": lbl,
                "clip_id": clip_id,
                "sample_rate_hz": fs,
                "level_db_spl": 65.0,
                "snr_db": "10",
                "mode": mode_val,
                "runtime_s": dt,
                **metrics,
              }
            )

  if rows:
    fieldnames = list(rows[0].keys())
    with (out_root / "metrics.csv").open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(rows)
    _plot_summary_charts(rows, out_root / "charts")

  return rows


def main(argv: Sequence[str]) -> None:
  """CLI entrypoint for the real-speech diagnostic benchmark suite."""
  del argv
  if _PREPARE_LIBRISPEECH_ARCHIVE.value is not None:
    if _INPUT_DIR.value is None:
      raise ValueError(
        "--input_dir must be set when --prepare_librispeech_archive is given."
      )
    subprocess.run(
      [
        sys.executable,
        "-m",
        "hp_acoustic.scripts.data.prepare_librispeech",
        f"--archive={_PREPARE_LIBRISPEECH_ARCHIVE.value}",
        f"--output_dir={_INPUT_DIR.value}",
      ],
      check=True,
    )

  sample_rates = [int(x) for x in _SAMPLE_RATES.value]
  levels = [float(x) for x in _LEVELS_DB_SPL.value]
  snrs: list[float | None] = [
    None if x.strip().lower() == "clean" else float(x) for x in _SNRS_DB.value
  ]
  budgets = [_parse_step_budget(x) for x in _STEP_BUDGETS.value]
  chunk_s = _CHUNK_SECONDS.value if _CHUNK_SECONDS.value > 0.0 else None

  rows = run_benchmark(
    input_dir=_INPUT_DIR.value,
    output_dir=_OUTPUT_DIR.value,
    split=_SPLIT.value,
    max_clips=_MAX_CLIPS.value,
    clip_seconds=_CLIP_SECONDS.value,
    sample_rates=sample_rates,
    levels_db_spl=levels,
    snrs_db=snrs,
    step_budgets=budgets,
    enhancers=_ENHANCERS.value,
    chunk_seconds=chunk_s,
    neural_checkpoint=_NEURAL_CHECKPOINT.value,
    seed=_SEED.value,
  )
  print(f"Wrote {len(rows)} benchmark rows to {_OUTPUT_DIR.value}/metrics.csv")


if __name__ == "__main__":
  define_flags()
  app.run(main)
