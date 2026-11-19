"""Direct head-to-head benchmark comparing ``carfac_inversion`` vs. ``AFHI/arcfac``.

Evaluates both solvers and baselines on ``arcfac``'s native representation and
spectral metrics (``nap.round_trip_rel``, ``sai.round_trip_rel``, ``kept``,
``healed``, ``direction``, ``off_axis``, ``log_stft_to_x``, ``corr_with_x``)
alongside perceptual speech metrics (``SI-SDR``, ``STOI``, ``HASPI``, ``HASQI``).
"""

import csv
import importlib
import pathlib
import sys
import time
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl import app
from absl import flags
from hp_acoustic.io import audio_io
from hp_acoustic.models import carfac

from carfac_inversion import arcfac_metrics
from carfac_inversion import clinical
from carfac_inversion import inversion
from carfac_inversion import nap_inverter
from carfac_inversion import neural
from carfac_inversion import tpu
from carfac_inversion.scripts import benchmark_suite

FLAGS = flags.FLAGS

_INPUT_DIR: flags.FlagHolder[str | None]
_OUTPUT_DIR: flags.FlagHolder[str]
_SPLIT: flags.FlagHolder[str]
_MAX_CLIPS: flags.FlagHolder[int]
_CLIP_SECONDS: flags.FlagHolder[float]
_SAMPLE_RATE_HZ: flags.FlagHolder[int]
_LEVEL_DB_SPL: flags.FlagHolder[float]
_TARGETS: flags.FlagHolder[list[str]]
_METHODS: flags.FlagHolder[list[str]]
_ARCFAC_SRC: flags.FlagHolder[str | None]
_CARFAC_ADAM_STEPS: flags.FlagHolder[int]
_CARFAC_LBFGS_STEPS: flags.FlagHolder[int]
_CHUNK_SECONDS: flags.FlagHolder[float]
_ARCFAC_STEPS: flags.FlagHolder[int]
_ARCFAC_START: flags.FlagHolder[str]
_ARCFAC_LAM: flags.FlagHolder[float]
_ARCFAC_LR_REL: flags.FlagHolder[float]
_ARCFAC_EPS_REL: flags.FlagHolder[float]
_SLANEY_ITERS: flags.FlagHolder[int]
_NEURAL_CHECKPOINT: flags.FlagHolder[str | None]
_PERCEPTUAL_NEURAL_CHECKPOINT: flags.FlagHolder[str | None]
_PHASE4_CHECKPOINT: flags.FlagHolder[str | None]
_SEED: flags.FlagHolder[int]


def define_flags(flag_values: flags.FlagValues = FLAGS) -> None:
  """Registers head-to-head ``arcfac`` benchmark CLI flags."""
  global _INPUT_DIR
  global _OUTPUT_DIR
  global _SPLIT
  global _MAX_CLIPS
  global _CLIP_SECONDS
  global _SAMPLE_RATE_HZ
  global _LEVEL_DB_SPL
  global _TARGETS
  global _METHODS
  global _ARCFAC_SRC
  global _CARFAC_ADAM_STEPS
  global _CARFAC_LBFGS_STEPS
  global _CHUNK_SECONDS
  global _ARCFAC_STEPS
  global _ARCFAC_START
  global _ARCFAC_LAM
  global _ARCFAC_LR_REL
  global _ARCFAC_EPS_REL
  global _SLANEY_ITERS
  global _NEURAL_CHECKPOINT
  global _PERCEPTUAL_NEURAL_CHECKPOINT
  global _PHASE4_CHECKPOINT
  global _SEED

  _INPUT_DIR = flags.DEFINE_string(
    "input_dir",
    None,
    "Optional WAV/FLAC dir or ProcessedClips root (synthetic speech if None).",
    flag_values=flag_values,
  )
  _OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "tmp/arcfac_comparison",
    "Directory for comparison CSV, Markdown summary, and rendered WAVs.",
    flag_values=flag_values,
  )
  _SPLIT = flags.DEFINE_string(
    "split",
    "test",
    "Split name when reading ProcessedClips shards.",
    flag_values=flag_values,
  )
  _MAX_CLIPS = flags.DEFINE_integer(
    "max_clips", 2, "Maximum clips to evaluate.", flag_values=flag_values
  )
  _CLIP_SECONDS = flags.DEFINE_float(
    "clip_seconds", 2.0, "Clip length in seconds.", flag_values=flag_values
  )
  _SAMPLE_RATE_HZ = flags.DEFINE_integer(
    "sample_rate_hz",
    32000,
    "Sample rate in Hz (32000 matches arcfac default).",
    flag_values=flag_values,
  )
  _LEVEL_DB_SPL = flags.DEFINE_float(
    "level_db_spl",
    65.0,
    "Presentation level in dB SPL.",
    flag_values=flag_values,
  )
  _TARGETS = flags.DEFINE_list(
    "targets",
    ["healthy", "flat:0.5"],
    "Comma-separated ear targets: healthy, flat:<ohc>, sloping.",
    flag_values=flag_values,
  )
  _METHODS = flags.DEFINE_list(
    "methods",
    [
      "carfac_exact",
      "carfac_perceptual",
      "arcfac_match_nap",
      "arcfac_match_sai",
      "arcfac_slaney",
    ],
    "Comma-separated benchmark methods.",
    flag_values=flag_values,
  )
  _ARCFAC_SRC = flags.DEFINE_string(
    "arcfac_src",
    None,
    "Optional path to a local arcfac/src checkout when arcfac is not installed.",
    flag_values=flag_values,
  )
  _CARFAC_ADAM_STEPS = flags.DEFINE_integer(
    "carfac_adam_steps",
    45,
    "Adam steps for carfac_inversion methods.",
    flag_values=flag_values,
  )
  _CARFAC_LBFGS_STEPS = flags.DEFINE_integer(
    "carfac_lbfgs_steps",
    135,
    "L-BFGS steps for carfac_inversion methods.",
    flag_values=flag_values,
  )
  _CHUNK_SECONDS = flags.DEFINE_float(
    "chunk_seconds",
    1.5,
    "OLA window duration in seconds for carfac_inversion (<=0 disables OLA).",
    flag_values=flag_values,
  )
  _ARCFAC_STEPS = flags.DEFINE_integer(
    "arcfac_steps",
    200,
    "Optimization steps for arcfac.match and arcfac.slaney Stage B.",
    flag_values=flag_values,
  )
  _ARCFAC_START = flags.DEFINE_string(
    "arcfac_start",
    "noise",
    "Initial waveform start passed to arcfac.match (noise, randphase, flip).",
    flag_values=flag_values,
  )
  _ARCFAC_LAM = flags.DEFINE_float(
    "arcfac_lam",
    0.0,
    "Waveform tie-breaker weight passed to arcfac.match.",
    flag_values=flag_values,
  )
  _ARCFAC_LR_REL = flags.DEFINE_float(
    "arcfac_lr_rel",
    0.05,
    "Relative learning rate lr = lr_rel * rms(x) for arcfac methods.",
    flag_values=flag_values,
  )
  _ARCFAC_EPS_REL = flags.DEFINE_float(
    "arcfac_eps_rel",
    1.0,
    "Adam eps multiplier eps = eps_rel * rms(g0) for arcfac.match.",
    flag_values=flag_values,
  )
  _SLANEY_ITERS = flags.DEFINE_integer(
    "slaney_iters",
    30,
    "Griffin-Lim Stage A iterations for arcfac.slaney.",
    flag_values=flag_values,
  )
  _NEURAL_CHECKPOINT = flags.DEFINE_string(
    "neural_checkpoint",
    None,
    "Optional exact neural refiner checkpoint path.",
    flag_values=flag_values,
  )
  _PERCEPTUAL_NEURAL_CHECKPOINT = flags.DEFINE_string(
    "perceptual_neural_checkpoint",
    None,
    "Optional perceptual neural refiner checkpoint path.",
    flag_values=flag_values,
  )
  _PHASE4_CHECKPOINT = flags.DEFINE_string(
    "phase4_checkpoint",
    None,
    "Optional Phase 4 direct causal inverter checkpoint (.npz) path.",
    flag_values=flag_values,
  )
  _SEED = flags.DEFINE_integer("seed", 0, "PRNG seed.", flag_values=flag_values)


def _load_arcfac_modules(arcfac_src: str | None) -> tuple[Any, Any, Any]:
  """Imports ``arcfac.ear``, ``arcfac.match``, and ``arcfac.slaney``."""
  if arcfac_src:
    resolved = str(pathlib.Path(arcfac_src).expanduser().resolve())
    if resolved not in sys.path:
      sys.path.insert(0, resolved)
  try:
    ear_mod = importlib.import_module("arcfac.ear")
    match_mod = importlib.import_module("arcfac.match")
    slaney_mod = importlib.import_module("arcfac.slaney")
  except ImportError as exc:
    raise ImportError(
      "Running arcfac baseline methods requires the 'arcfac' package or "
      "--arcfac_src=/path/to/arcfac/src."
    ) from exc
  return ear_mod, match_mod, slaney_mod


_cochlea_to_host = neural.cochlea_to_host
_replicate_cochlea = neural.replicate_cochlea


def _build_target_cochlea(
  spec: str, healthy: carfac.CarfacCochleaModel, sample_rate_hz: int
) -> tuple[carfac.CarfacCochleaModel, np.ndarray]:
  """Builds target ``CarfacCochleaModel`` and HASPI/HASQI audiogram array."""
  if spec == "healthy":
    return healthy, np.zeros(6, dtype=np.float64)
  low = spec.strip().lower()
  if (
    low.startswith("bisgaard:")
    or low.startswith("iso7029:")
    or spec.strip().upper() in clinical.BISGAARD_AUDIOGRAMS_DB_HL
  ):
    prof = clinical.build_clinical_profile(
      spec,
      sample_rate_hz=float(sample_rate_hz),
      channel_freqs_hz=np.asarray(
        healthy.channel_frequencies, dtype=np.float32
      ),
    )
    return (
      clinical.build_cochlea_from_profile(prof),
      np.asarray(prof.audiogram_6_db_hl, dtype=np.float64),
    )
  if spec == "sloping":
    return (
      carfac.build(
        fs=sample_rate_hz,
        ohc_health=benchmark_suite._sloping_ohc_health(healthy),
      ),
      np.zeros(6, dtype=np.float64),
    )
  kind, _, arg = spec.partition(":")
  if kind == "flat":
    ohc = float(arg)
    hl = (
      benchmark_suite.HALF_OHC_HEARING_LOSS_DB
      if abs(ohc - 0.5) < 1e-6
      else np.zeros(6, dtype=np.float64)
    )
    return carfac.build(fs=sample_rate_hz, ohc_health=ohc), hl
  raise ValueError(
    f"Unsupported target spec {spec!r}; expected 'healthy', 'sloping', or 'flat:<ohc>'."
  )


def _run_method(
  method: str,
  *,
  sig: np.ndarray,
  target_nap: np.ndarray,
  target_sai: np.ndarray,
  healthy: carfac.CarfacCochleaModel,
  sample_rate_hz: int,
  mask_samples: int,
  carfac_adam_steps: int,
  carfac_lbfgs_steps: int,
  chunk_seconds: float | None,
  neural_checkpoint: str | None,
  perceptual_neural_checkpoint: str | None,
  phase4_checkpoint: str | None,
  arcfac_mods: tuple[Any, Any, Any] | None,
  arcfac_steps: int,
  arcfac_start: str,
  arcfac_lam: float,
  arcfac_lr_rel: float,
  arcfac_eps_rel: float,
  slaney_iters: int,
  seed: int,
) -> tuple[np.ndarray, float]:
  """Executes one inversion method and returns ``(est_waveform, runtime_s)``."""
  if method in ("carfac_exact", "carfac_perceptual"):
    mode: inversion.InversionMode = (
      "exact" if method == "carfac_exact" else "perceptual"
    )
    ckpt = (
      neural_checkpoint if mode == "exact" else perceptual_neural_checkpoint
    )
    cfg = inversion.InversionConfig(
      mode=mode,
      adam_steps=carfac_adam_steps,
      lbfgs_steps=carfac_lbfgs_steps,
      chunk_seconds=chunk_seconds,
      neural_checkpoint=ckpt,
      seed=seed,
    )
    t0 = time.perf_counter()
    res = inversion.invert_nap(target_nap, healthy, cfg)
    return res.waveform, time.perf_counter() - t0

  if method == "phase4_direct":
    if not phase4_checkpoint:
      raise ValueError(
        "Method 'phase4_direct' requires --phase4_checkpoint=<path>."
      )
    t0 = time.perf_counter()
    wav = nap_inverter.invert_nap_direct(
      target_nap,
      phase4_checkpoint,
      align_lookahead=True,
      cochlea_model=healthy,
    )
    return wav, time.perf_counter() - t0

  if arcfac_mods is None:
    raise RuntimeError(f"arcfac modules were not loaded for method {method!r}.")
  _, match_mod, slaney_mod = arcfac_mods
  healthy_ear_fn = jax.jit(jax.vmap(lambda w: healthy(w[:, None])[:, :, 0]))
  rms = float(np.sqrt(np.mean(np.square(sig))))

  if method in ("arcfac_match_nap", "arcfac_match_sai"):
    obj = "nap" if method == "arcfac_match_nap" else "sai"
    fn = match_mod.make_objective(obj, healthy_ear_fn, mask_samples)
    tgt_arr = jnp.asarray(
      target_nap[None] if obj == "nap" else target_sai, dtype=jnp.float32
    )
    y0 = match_mod.make_start(arcfac_start, sig, seed=seed)
    t0 = time.perf_counter()
    y, _, _ = match_mod.run_one(
      y0,
      sig,
      tgt_arr,
      fn,
      arcfac_lam,
      arcfac_steps,
      arcfac_lr_rel * rms,
      arcfac_eps_rel,
      max(1, arcfac_steps),
      f"{method}@{sample_rate_hz}",
    )
    return np.asarray(y, dtype=np.float32), time.perf_counter() - t0

  if method == "arcfac_slaney":
    t0 = time.perf_counter()
    nap_est, _ = slaney_mod.invert_sai(
      np.asarray(target_sai[0]),
      len(sig),
      iterations=slaney_iters,
    )
    y, _ = slaney_mod.nap_to_audio(
      jnp.asarray(nap_est, dtype=jnp.float32),
      healthy_ear_fn,
      level=rms,
      steps=arcfac_steps,
      lr_rel=arcfac_lr_rel,
      seed=seed,
    )
    return np.asarray(y, dtype=np.float32), time.perf_counter() - t0

  raise ValueError(f"Unsupported benchmark method {method!r}.")


def format_summary_table(rows: list[dict[str, Any]]) -> str:
  """Builds a concise Markdown summary grouped by ``(target, method)``."""
  if not rows:
    return "No benchmark rows recorded."
  lines = [
    "| target | method | nap_rt_rel | sai_rt_rel | nap_kept | nap_dir | "
    "nap_off | sai_kept | sai_dir | sai_off | log_stft | si_sdr | hasqi | "
    "time_s |",
    "| :--- | :--- | ---: | ---: | ---: | ---: | "
    "---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  groups: list[tuple[str, str]] = []
  seen: set[tuple[str, str]] = set()
  for r in rows:
    key = (str(r["target"]), str(r["method"]))
    if key not in seen:
      seen.add(key)
      groups.append(key)

  def _fmt(v: float, prec: int = 3) -> str:
    return "-" if not np.isfinite(v) else f"{v:.{prec}f}"

  for target, method in groups:
    sub = [
      r
      for r in rows
      if str(r["target"]) == target and str(r["method"]) == method
    ]
    mean = lambda k: benchmark_suite._nanmean([float(r[k]) for r in sub])  # noqa: E731
    lines.append(
      f"| {target} | {method} | "
      f"{_fmt(mean('arcfac_nap_round_trip_rel'))} | "
      f"{_fmt(mean('arcfac_sai_round_trip_rel'))} | "
      f"{_fmt(mean('arcfac_nap_kept'))} | "
      f"{_fmt(mean('arcfac_nap_direction'))} | "
      f"{_fmt(mean('arcfac_nap_off_axis'))} | "
      f"{_fmt(mean('arcfac_sai_kept'))} | "
      f"{_fmt(mean('arcfac_sai_direction'))} | "
      f"{_fmt(mean('arcfac_sai_off_axis'))} | "
      f"{_fmt(mean('arcfac_log_stft'))} | "
      f"{_fmt(mean('si_sdr'), 2)} | "
      f"{_fmt(mean('hasqi'))} | "
      f"{_fmt(mean('runtime_s'), 2)} |"
    )
  return "\n".join(lines)


def run_arcfac_benchmark(
  *,
  input_dir: str | None,
  output_dir: str,
  split: str = "test",
  max_clips: int = 2,
  clip_seconds: float = 2.0,
  sample_rate_hz: int = 32000,
  level_db_spl: float = 65.0,
  targets: Sequence[str] = ("healthy", "flat:0.5"),
  methods: Sequence[str] = (
    "carfac_exact",
    "carfac_perceptual",
    "arcfac_match_nap",
    "arcfac_match_sai",
    "arcfac_slaney",
  ),
  arcfac_src: str | None = None,
  carfac_adam_steps: int = 45,
  carfac_lbfgs_steps: int = 135,
  chunk_seconds: float | None = 1.5,
  arcfac_steps: int = 200,
  arcfac_start: str = "noise",
  arcfac_lam: float = 0.0,
  arcfac_lr_rel: float = 0.05,
  arcfac_eps_rel: float = 1.0,
  slaney_iters: int = 30,
  neural_checkpoint: str | None = None,
  perceptual_neural_checkpoint: str | None = None,
  phase4_checkpoint: str | None = None,
  seed: int = 0,
) -> list[dict[str, Any]]:
  """Runs head-to-head comparison between ``carfac_inversion`` and ``arcfac``."""
  needs_arcfac = any(m.startswith("arcfac_") for m in methods)
  arcfac_mods = _load_arcfac_modules(arcfac_src) if needs_arcfac else None

  out_root = pathlib.Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)
  clips = benchmark_suite.load_speech_clips(
    input_dir,
    sample_rate_hz=sample_rate_hz,
    max_clips=max_clips,
    clip_seconds=clip_seconds,
    split=split,
  )
  mesh = tpu.make_data_mesh()
  with tpu.host_cpu_context():
    healthy_host = _cochlea_to_host(carfac.build(fs=sample_rate_hz))
  healthy = _replicate_cochlea(healthy_host, mesh)
  forward_jit = jax.jit(lambda w, cm: cm(w[:, None])[:, :, 0])

  rows: list[dict[str, Any]] = []
  for clip_id, raw_clean in clips:
    sig = benchmark_suite._scale_to_spl(raw_clean, level_db_spl)
    healthy_ref_nap = np.asarray(
      forward_jit(jnp.asarray(sig), healthy), dtype=np.float32
    )
    mask_samples = min(int(round(0.25 * sample_rate_hz)), len(sig) // 4)

    for target_spec in targets:
      with tpu.host_cpu_context():
        tgt_raw, hl_arr = _build_target_cochlea(
          target_spec, healthy_host, sample_rate_hz
        )
        tgt_host = _cochlea_to_host(tgt_raw)
      target_cochlea = (
        healthy
        if target_spec == "healthy"
        else _replicate_cochlea(tgt_host, mesh)
      )
      target_nap = (
        healthy_ref_nap
        if target_spec == "healthy"
        else np.asarray(
          forward_jit(jnp.asarray(sig), target_cochlea),
          dtype=np.float32,
        )
      )
      target_sai = arcfac_metrics.sai_of_naps(target_nap[None])

      for method in methods:
        print(
          f"[bench_arcfac] running {clip_id} | target={target_spec} | method={method}...",
          flush=True,
        )
        est, runtime_s = _run_method(
          method,
          sig=sig,
          target_nap=target_nap,
          target_sai=target_sai,
          healthy=healthy,
          sample_rate_hz=sample_rate_hz,
          mask_samples=mask_samples,
          carfac_adam_steps=carfac_adam_steps,
          carfac_lbfgs_steps=carfac_lbfgs_steps,
          chunk_seconds=chunk_seconds,
          neural_checkpoint=neural_checkpoint,
          perceptual_neural_checkpoint=perceptual_neural_checkpoint,
          phase4_checkpoint=phase4_checkpoint,
          arcfac_mods=arcfac_mods,
          arcfac_steps=arcfac_steps,
          arcfac_start=arcfac_start,
          arcfac_lam=arcfac_lam,
          arcfac_lr_rel=arcfac_lr_rel,
          arcfac_eps_rel=arcfac_eps_rel,
          slaney_iters=slaney_iters,
          seed=seed,
        )
        recon_nap = np.asarray(
          forward_jit(jnp.asarray(est), healthy), dtype=np.float32
        )
        wav_out = (
          out_root
          / "audio"
          / f"{sample_rate_hz}hz_{clip_id}"
          / f"{target_spec.replace(':', '_')}_{method}.wav"
        )
        audio_io.save_audio_to_file(est, str(wav_out), sample_rate_hz)

        metrics = benchmark_suite._score_pair(
          ref=sig,
          est=est,
          target_nap=target_nap,
          recon_nap=recon_nap,
          healthy_ref_nap=healthy_ref_nap,
          sample_rate_hz=sample_rate_hz,
          hearing_loss_db=hl_arr,
        )
        row = {
          "target": target_spec,
          "method": method,
          "clip_id": clip_id,
          "sample_rate_hz": sample_rate_hz,
          "level_db_spl": level_db_spl,
          "runtime_s": runtime_s,
          **metrics,
        }
        rows.append(row)
        print(
          f"[bench_arcfac] done {target_spec}/{method} in {runtime_s:.1f}s | "
          f"nap_rt={row['arcfac_nap_round_trip_rel']:.3f} "
          f"sai_rt={row['arcfac_sai_round_trip_rel']:.3f} "
          f"nap_kept={row['arcfac_nap_kept']:.3f} "
          f"sai_dir={row['arcfac_sai_direction']:.3f} "
          f"stft={row['arcfac_log_stft']:.3f} sdr={row['si_sdr']:.2f}dB",
          flush=True,
        )
        fieldnames = list(rows[0].keys())
        with (out_root / "benchmark_arcfac.csv").open("w", newline="") as f:
          writer = csv.DictWriter(f, fieldnames=fieldnames)
          writer.writeheader()
          writer.writerows(rows)

  if rows:
    fieldnames = list(rows[0].keys())
    with (out_root / "benchmark_arcfac.csv").open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(rows)
    table = format_summary_table(rows)
    (out_root / "summary.md").write_text(table + "\n")
    print(table, flush=True)

  return rows


def main(argv: Sequence[str]) -> None:
  """CLI entrypoint for the ``carfac_inversion`` vs. ``arcfac`` benchmark."""
  del argv
  chunk_s = _CHUNK_SECONDS.value if _CHUNK_SECONDS.value > 0.0 else None
  run_arcfac_benchmark(
    input_dir=_INPUT_DIR.value,
    output_dir=_OUTPUT_DIR.value,
    split=_SPLIT.value,
    max_clips=_MAX_CLIPS.value,
    clip_seconds=_CLIP_SECONDS.value,
    sample_rate_hz=_SAMPLE_RATE_HZ.value,
    level_db_spl=_LEVEL_DB_SPL.value,
    targets=_TARGETS.value,
    methods=_METHODS.value,
    arcfac_src=_ARCFAC_SRC.value,
    carfac_adam_steps=_CARFAC_ADAM_STEPS.value,
    carfac_lbfgs_steps=_CARFAC_LBFGS_STEPS.value,
    chunk_seconds=chunk_s,
    arcfac_steps=_ARCFAC_STEPS.value,
    arcfac_start=_ARCFAC_START.value,
    arcfac_lam=_ARCFAC_LAM.value,
    arcfac_lr_rel=_ARCFAC_LR_REL.value,
    arcfac_eps_rel=_ARCFAC_EPS_REL.value,
    slaney_iters=_SLANEY_ITERS.value,
    neural_checkpoint=_NEURAL_CHECKPOINT.value,
    perceptual_neural_checkpoint=_PERCEPTUAL_NEURAL_CHECKPOINT.value,
    phase4_checkpoint=_PHASE4_CHECKPOINT.value,
    seed=_SEED.value,
  )


if __name__ == "__main__":
  define_flags()
  app.run(main)
