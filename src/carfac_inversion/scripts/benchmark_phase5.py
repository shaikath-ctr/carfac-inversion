"""Head-to-head Phase 5 prescription benchmark: Unaided vs. NAL-R vs. Phase 5."""

import csv
import math
import pathlib
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
from jax_haspi_hasqi import haspi as haspi_lib
from jax_haspi_hasqi import hasqi as hasqi_lib

from carfac_inversion import arcfac_metrics
from carfac_inversion import clinical
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
_LEVELS_DB_SPL: flags.FlagHolder[list[str]]
_PROFILES: flags.FlagHolder[list[str]]
_METHODS: flags.FlagHolder[list[str]]
_PHASE5_CHECKPOINT: flags.FlagHolder[str | None]
_FIT_STEPS: flags.FlagHolder[int]
_MPO_CLIP: flags.FlagHolder[float]

_DEFAULT_PROFILES = [
  "bisgaard:N1",
  "bisgaard:N2",
  "bisgaard:N3",
  "bisgaard:N4",
  "bisgaard:S1",
  "bisgaard:S2",
  "bisgaard:S3",
  "iso7029:70:male:0.5",
]


def define_flags(flag_values: flags.FlagValues = FLAGS) -> None:
  """Registers Phase 5 clinical prescription benchmark CLI flags."""
  global _INPUT_DIR
  global _OUTPUT_DIR
  global _SPLIT
  global _MAX_CLIPS
  global _CLIP_SECONDS
  global _SAMPLE_RATE_HZ
  global _LEVELS_DB_SPL
  global _PROFILES
  global _METHODS
  global _PHASE5_CHECKPOINT
  global _FIT_STEPS
  global _MPO_CLIP

  _INPUT_DIR = flags.DEFINE_string(
    "input_dir",
    None,
    "Optional WAV/FLAC dir or ProcessedClips root (synthetic speech if None).",
    flag_values=flag_values,
  )
  _OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "tmp/phase5_comparison",
    "Directory for benchmark CSV, Markdown summary, and rendered WAVs.",
    flag_values=flag_values,
  )
  _SPLIT = flags.DEFINE_string(
    "split", "test", "ProcessedClips split name.", flag_values=flag_values
  )
  _MAX_CLIPS = flags.DEFINE_integer(
    "max_clips", 4, "Maximum clips to evaluate.", flag_values=flag_values
  )
  _CLIP_SECONDS = flags.DEFINE_float(
    "clip_seconds", 2.0, "Clip duration in seconds.", flag_values=flag_values
  )
  _SAMPLE_RATE_HZ = flags.DEFINE_integer(
    "sample_rate_hz", 32000, "Sample rate in Hz.", flag_values=flag_values
  )
  _LEVELS_DB_SPL = flags.DEFINE_list(
    "levels_db_spl",
    ["65.0"],
    "Comma-separated presentation levels in dB SPL.",
    flag_values=flag_values,
  )
  _PROFILES = flags.DEFINE_list(
    "profiles",
    _DEFAULT_PROFILES,
    "Comma-separated clinical listener profile specs.",
    flag_values=flag_values,
  )
  _METHODS = flags.DEFINE_list(
    "methods",
    ["unaided", "nalr", "phase5_direct"],
    "Comma-separated prescription methods.",
    flag_values=flag_values,
  )
  _PHASE5_CHECKPOINT = flags.DEFINE_string(
    "phase5_checkpoint",
    None,
    "Conditioned Phase 5 checkpoint (.npz) path.",
    flag_values=flag_values,
  )
  _FIT_STEPS = flags.DEFINE_integer(
    "fit_steps",
    0,
    "Optional CARFAC audiogram OHC fit refinement steps (0 = fast analytic mapping).",
    flag_values=flag_values,
  )
  _MPO_CLIP = flags.DEFINE_float(
    "mpo_clip",
    0.99,
    "Output peak comfort clamp (0 disables).",
    flag_values=flag_values,
  )


def format_phase5_summary(rows: Sequence[dict[str, Any]]) -> str:
  """Formats grouped Markdown table by ``(level_db_spl, profile, method)``."""
  if not rows:
    return "No Phase 5 benchmark rows recorded."
  lines = [
    "| level_db | profile | method | cochlear_nap_rel | cochlear_sai_rel | "
    "nap_healed_pct | haspi | hasqi | hasqi_nl | hasqi_lin | peak_abs | time_s |",
    "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | "
    "---: | ---: | ---: |",
  ]
  groups: list[tuple[float, str, str]] = []
  seen: set[tuple[float, str, str]] = set()
  for r in rows:
    key = (float(r["level_db_spl"]), str(r["profile"]), str(r["method"]))
    if key not in seen:
      seen.add(key)
      groups.append(key)

  def _fmt(val: float, prec: int = 3) -> str:
    return "-" if not math.isfinite(val) else f"{val:.{prec}f}"

  for lv, prof, meth in groups:
    sub = [
      r
      for r in rows
      if float(r["level_db_spl"]) == lv
      and str(r["profile"]) == prof
      and str(r["method"]) == meth
    ]
    mean = lambda k: benchmark_suite._nanmean([float(r[k]) for r in sub])  # noqa: E731
    lines.append(
      f"| {lv:g} | {prof} | {meth} | "
      f"{_fmt(mean('cochlear_nap_rel'))} | "
      f"{_fmt(mean('cochlear_sai_rel'))} | "
      f"{_fmt(mean('nap_healed_pct'), 1)}% | "
      f"{_fmt(mean('haspi'))} | "
      f"{_fmt(mean('hasqi'))} | "
      f"{_fmt(mean('hasqi_nl'))} | "
      f"{_fmt(mean('hasqi_lin'))} | "
      f"{_fmt(mean('peak_abs'))} | "
      f"{_fmt(mean('runtime_s'), 2)} |"
    )
  return "\n".join(lines)


def _score_prescription(
  *,
  ref: np.ndarray,
  est: np.ndarray,
  healthy_ref_nap: np.ndarray,
  healthy_ref_sai: np.ndarray,
  impaired_cochlea: carfac.CarfacCochleaModel,
  forward_jit: Any,
  audiogram_6_db_hl: np.ndarray,
  sample_rate_hz: int,
  mask_samples: int,
  unaided_nap_gap: float | None = None,
) -> dict[str, float]:
  """Scores how closely aided audio ``est`` restores healthy NAP/SAI & HASPI/HASQI."""
  aided_nap = np.asarray(
    forward_jit(jnp.asarray(est, jnp.float32), impaired_cochlea),
    dtype=np.float32,
  )
  aided_sai = np.asarray(
    arcfac_metrics.sai_of_naps(aided_nap[None]), dtype=np.float32
  )
  m = min(mask_samples, max(0, len(ref) - 1))
  hx_norm = max(float(np.mean(np.abs(healthy_ref_nap[m:]))), 1e-8)
  nap_err = float(np.mean(np.abs(aided_nap[m:] - healthy_ref_nap[m:])))
  healed_pct = (
    100.0 * (1.0 - nap_err / max(unaided_nap_gap, 1e-8))
    if unaided_nap_gap is not None
    else 0.0
  )

  first_frame = min(
    arcfac_metrics.first_unmasked_frame(m),
    max(0, healthy_ref_sai.shape[1] - 1),
  )
  sai_norm = max(float(np.mean(np.abs(healthy_ref_sai[:, first_frame:]))), 1e-8)
  sai_err = float(
    np.mean(
      np.abs(aided_sai[:, first_frame:] - healthy_ref_sai[:, first_frame:])
    )
  )

  aud64 = np.asarray(audiogram_6_db_hl, dtype=np.float64)
  level1 = benchmark_suite.CARFAC_RMS_ONE_DB_SPL
  with tpu.host_cpu_context():
    haspi_val = float(
      haspi_lib.haspi_v2(
        ref, sample_rate_hz, est, sample_rate_hz, aud64, level1=level1
      )[0]
    )
    q, nl, lin, _ = hasqi_lib.hasqi_v2(
      ref, sample_rate_hz, est, sample_rate_hz, aud64, level1=level1
    )
  return {
    "cochlear_nap_rel": nap_err / hx_norm,
    "cochlear_sai_rel": sai_err / sai_norm,
    "nap_healed_pct": healed_pct,
    "nap_raw_err": nap_err,
    "haspi": haspi_val,
    "hasqi": float(q),
    "hasqi_nl": float(nl),
    "hasqi_lin": float(lin),
    "peak_abs": float(np.max(np.abs(est))),
  }


def run_phase5_benchmark(
  *,
  input_dir: str | None,
  output_dir: str,
  split: str = "test",
  max_clips: int = 4,
  clip_seconds: float = 2.0,
  sample_rate_hz: int = 32000,
  levels_db_spl: Sequence[float] = (65.0,),
  profiles: Sequence[str] = _DEFAULT_PROFILES,
  methods: Sequence[str] = ("unaided", "nalr", "phase5_direct"),
  phase5_checkpoint: str | None = None,
  fit_steps: int = 0,
  mpo_clip: float = 0.99,
) -> list[dict[str, Any]]:
  """Runs head-to-head hearing-aid prescription comparisons across profiles & SPLs."""
  if "phase5_direct" in methods and not phase5_checkpoint:
    raise ValueError(
      "Method 'phase5_direct' requires --phase5_checkpoint=<path>."
    )

  out_root = pathlib.Path(output_dir)
  out_root.mkdir(parents=True, exist_ok=True)
  clips = benchmark_suite.load_speech_clips(
    input_dir,
    sample_rate_hz=sample_rate_hz,
    max_clips=max_clips,
    clip_seconds=clip_seconds,
    split=split,
  )
  with tpu.host_cpu_context():
    healthy = neural.cochlea_to_host(carfac.build(fs=sample_rate_hz))
  forward_jit = jax.jit(lambda w, cm: cm(w[:, None])[:, :, 0])
  cfs = np.asarray(healthy.channel_frequencies, dtype=np.float32)

  built_profiles: list[
    tuple[clinical.ClinicalListenerProfile, carfac.CarfacCochleaModel]
  ] = []
  for spec in profiles:
    prof = clinical.build_clinical_profile(
      spec,
      sample_rate_hz=float(sample_rate_hz),
      channel_freqs_hz=cfs,
      fit_steps=fit_steps,
    )
    with tpu.host_cpu_context():
      cm = clinical.build_cochlea_from_profile(prof, base_cochlea=healthy)
    built_profiles.append((prof, cm))

  rows: list[dict[str, Any]] = []
  for lv in levels_db_spl:
    lv_f = float(lv)
    for clip_id, raw_clean in clips:
      sig = benchmark_suite._scale_to_spl(raw_clean, lv_f)
      mask_samples = min(int(round(0.25 * sample_rate_hz)), len(sig) // 4)
      healthy_nap = np.asarray(
        forward_jit(jnp.asarray(sig, jnp.float32), healthy), dtype=np.float32
      )
      healthy_sai = np.asarray(
        arcfac_metrics.sai_of_naps(healthy_nap[None]), dtype=np.float32
      )

      for prof, imp_cm in built_profiles:
        unaided_scores = _score_prescription(
          ref=sig,
          est=sig,
          healthy_ref_nap=healthy_nap,
          healthy_ref_sai=healthy_sai,
          impaired_cochlea=imp_cm,
          forward_jit=forward_jit,
          audiogram_6_db_hl=prof.audiogram_6_db_hl,
          sample_rate_hz=sample_rate_hz,
          mask_samples=mask_samples,
        )
        unaided_gap = float(unaided_scores["nap_raw_err"])

        for method in methods:
          if method == "unaided":
            est = sig
            runtime_s = 0.0
            sc = {**unaided_scores, "nap_healed_pct": 0.0}
          else:
            t0 = time.perf_counter()
            if method == "nalr":
              est = clinical.apply_nalr(
                sig,
                prof.audiogram_6_db_hl,
                float(sample_rate_hz),
                mpo_clip=mpo_clip,
              )
            elif method == "phase5_direct":
              assert phase5_checkpoint is not None
              est = nap_inverter.prescribe_audio_direct(
                sig,
                prof,
                phase5_checkpoint,
                healthy_cochlea=healthy,
                align_lookahead=True,
                mpo_clip=mpo_clip,
              )
            else:
              raise ValueError(f"Unsupported Phase 5 method {method!r}.")
            runtime_s = time.perf_counter() - t0
            sc = _score_prescription(
              ref=sig,
              est=est,
              healthy_ref_nap=healthy_nap,
              healthy_ref_sai=healthy_sai,
              impaired_cochlea=imp_cm,
              forward_jit=forward_jit,
              audiogram_6_db_hl=prof.audiogram_6_db_hl,
              sample_rate_hz=sample_rate_hz,
              mask_samples=mask_samples,
              unaided_nap_gap=unaided_gap,
            )

          wav_path = (
            out_root
            / "audio"
            / f"{lv_f:g}dB_{clip_id}"
            / f"{prof.name.replace(':', '_')}_{method}.wav"
          )
          audio_io.save_audio_to_file(est, str(wav_path), sample_rate_hz)
          rows.append(
            {
              "level_db_spl": lv_f,
              "profile": prof.name,
              "method": method,
              "clip_id": clip_id,
              "sample_rate_hz": sample_rate_hz,
              "runtime_s": runtime_s,
              **sc,
            }
          )

  if rows:
    csv_path = out_root / "benchmark_phase5.csv"
    with csv_path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
      writer.writeheader()
      writer.writerows(rows)
    table = format_phase5_summary(rows)
    (out_root / "summary.md").write_text(table + "\n")
    print(table, flush=True)
  return rows


def main(argv: Sequence[str]) -> None:
  """CLI entrypoint for Phase 5 prescription benchmarking."""
  del argv
  run_phase5_benchmark(
    input_dir=_INPUT_DIR.value,
    output_dir=_OUTPUT_DIR.value,
    split=_SPLIT.value,
    max_clips=_MAX_CLIPS.value,
    clip_seconds=_CLIP_SECONDS.value,
    sample_rate_hz=_SAMPLE_RATE_HZ.value,
    levels_db_spl=[float(x) for x in _LEVELS_DB_SPL.value],
    profiles=_PROFILES.value,
    methods=_METHODS.value,
    phase5_checkpoint=_PHASE5_CHECKPOINT.value,
    fit_steps=_FIT_STEPS.value,
    mpo_clip=_MPO_CLIP.value,
  )


if __name__ == "__main__":
  define_flags()
  app.run(main)
