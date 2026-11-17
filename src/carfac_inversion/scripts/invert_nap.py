"""CLI to invert a WAV file's CARFAC NAP back to audio (round-trip or auralized)."""

import dataclasses
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from absl import app
from absl import flags
from hp_acoustic import audio_constants
from hp_acoustic.audiology import impaired_cochlea
from hp_acoustic.io import audio_io
from hp_acoustic.metrics import context as metric_context
from hp_acoustic.metrics import perceptual
from hp_acoustic.metrics import waveform as waveform_metrics
from hp_acoustic.models import carfac

from carfac_inversion import inversion

FLAGS = flags.FLAGS

_INPUT_WAV: flags.FlagHolder[str | None]
_OUTPUT_WAV: flags.FlagHolder[str | None]
_SAMPLE_RATE_HZ: flags.FlagHolder[int]
_LEVEL_DB_SPL: flags.FlagHolder[float | None]
_SOURCE_OHC_HEALTH: flags.FlagHolder[float | None]
_FITTED_PROFILE_PATH: flags.FlagHolder[str | None]
_MODE: flags.FlagHolder[str | None]
_ADAM_STEPS: flags.FlagHolder[int]
_LBFGS_STEPS: flags.FlagHolder[int]
_CHUNK_SECONDS: flags.FlagHolder[float]
_NEURAL_CHECKPOINT: flags.FlagHolder[str | None]


def define_flags(flag_values: flags.FlagValues = FLAGS) -> None:
  """Registers CLI flags on ``flag_values``."""
  global _INPUT_WAV
  global _OUTPUT_WAV
  global _SAMPLE_RATE_HZ
  global _LEVEL_DB_SPL
  global _SOURCE_OHC_HEALTH
  global _FITTED_PROFILE_PATH
  global _MODE
  global _ADAM_STEPS
  global _LBFGS_STEPS
  global _CHUNK_SECONDS
  global _NEURAL_CHECKPOINT

  _INPUT_WAV = flags.DEFINE_string(
    "input_wav",
    None,
    "Path to the input WAV audio file.",
    required=True,
    flag_values=flag_values,
  )
  _OUTPUT_WAV = flags.DEFINE_string(
    "output_wav",
    None,
    "Path to save the reconstructed WAV file.",
    required=True,
    flag_values=flag_values,
  )
  _SAMPLE_RATE_HZ = flags.DEFINE_integer(
    "sample_rate_hz",
    audio_constants.SOURCE_SAMPLE_RATE,
    "Target sample rate in Hz for CARFAC processing.",
    flag_values=flag_values,
  )
  _LEVEL_DB_SPL = flags.DEFINE_float(
    "level_db_spl",
    None,
    "Optional RMS calibration level in dB SPL (0 dBFS = 104 dB SPL).",
    flag_values=flag_values,
  )
  _SOURCE_OHC_HEALTH = flags.DEFINE_float(
    "source_ohc_health",
    None,
    "Optional OHC health in [0, 1] for the source cochlea before healthy "
    "auralization. When None, performs same-cochlea round-trip inversion.",
    flag_values=flag_values,
  )
  _FITTED_PROFILE_PATH = flags.DEFINE_string(
    "fitted_profile_path",
    None,
    "Optional .npz fitted cochlear profile for cross-cochlea auralization.",
    flag_values=flag_values,
  )
  _MODE = flags.DEFINE_enum(
    "mode",
    None,
    ["exact", "perceptual"],
    "Inversion mode. Defaults to 'perceptual' when a source impairment is "
    "set, else 'exact'.",
    flag_values=flag_values,
  )
  _ADAM_STEPS = flags.DEFINE_integer(
    "adam_steps",
    45,
    "Number of initial Adam optimization steps.",
    flag_values=flag_values,
  )
  _LBFGS_STEPS = flags.DEFINE_integer(
    "lbfgs_steps",
    135,
    "Number of RMS-normalized L-BFGS refinement steps.",
    flag_values=flag_values,
  )
  _CHUNK_SECONDS = flags.DEFINE_float(
    "chunk_seconds",
    1.5,
    "Fixed OLA window duration in seconds (<=0 disables OLA chunking).",
    flag_values=flag_values,
  )
  _NEURAL_CHECKPOINT = flags.DEFINE_string(
    "neural_checkpoint",
    None,
    "Optional path to a trained Phase 3 neural residual refiner .npz.",
    flag_values=flag_values,
  )


def run_invert_wav(
  *,
  input_wav: str,
  output_wav: str,
  sample_rate_hz: int = audio_constants.SOURCE_SAMPLE_RATE,
  level_db_spl: float | None = None,
  source_ohc_health: float | None = None,
  fitted_profile_path: str | None = None,
  mode: inversion.InversionMode | None = None,
  adam_steps: int = 45,
  lbfgs_steps: int = 135,
  chunk_seconds: float | None = 1.5,
  neural_checkpoint: str | None = None,
) -> inversion.NapInversionResult:
  """Loads a WAV file, passes it through source CARFAC, and inverts to WAV."""
  if fitted_profile_path is not None and source_ohc_health is not None:
    raise ValueError(
      "Specify at most one of fitted_profile_path and source_ohc_health."
    )

  audio, _ = audio_io.read_wav(input_wav, target_sr=sample_rate_hz)
  if level_db_spl is not None:
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms > 0.0:
      target_rms = 10.0 ** (
        (level_db_spl - audio_constants.DYNAMIC_RANGE_DB) / 20.0
      )
      audio = (audio * (target_rms / rms)).astype(np.float32)

  target_cochlea = carfac.build(fs=sample_rate_hz)
  if fitted_profile_path is not None:
    source_cochlea = impaired_cochlea.from_fitted_profile(
      fitted_profile_path, sample_rate=sample_rate_hz
    )
    # Match middle-ear front end so auralization isolates cochlear impairment.
    target_cochlea = dataclasses.replace(
      target_cochlea, middle_ear_filter=source_cochlea.middle_ear_filter
    )
    default_mode: inversion.InversionMode = "perceptual"
  elif source_ohc_health is not None:
    source_cochlea = carfac.build(
      fs=sample_rate_hz, ohc_health=source_ohc_health
    )
    default_mode = "perceptual"
  else:
    source_cochlea = target_cochlea
    default_mode = "exact"

  resolved_mode: inversion.InversionMode = (
    default_mode if mode is None else mode
  )
  target_nap = np.asarray(
    source_cochlea(jnp.asarray(audio[:, np.newaxis], dtype=jnp.float32))[
      :, :, 0
    ]
  )

  config = inversion.InversionConfig(
    mode=resolved_mode,
    adam_steps=adam_steps,
    lbfgs_steps=lbfgs_steps,
    chunk_seconds=chunk_seconds,
    neural_checkpoint=neural_checkpoint,
  )
  result = inversion.invert_nap(target_nap, target_cochlea, config)
  audio_io.save_audio_to_file(result.waveform, output_wav, sample_rate_hz)

  nap_corr = float(
    np.corrcoef(result.reconstructed_nap.ravel(), target_nap.ravel())[0, 1]
  )
  print(
    f"Mode: {resolved_mode} | Final loss: {result.final_loss:.6f} | "
    f"NAP corr: {nap_corr:.4f}"
  )
  if source_cochlea is target_cochlea:
    ctx = metric_context.Context(
      reference=audio,
      estimation=result.waveform,
      sample_rate=sample_rate_hz,
      hearing_loss_db=np.zeros(6),
    )
    print(
      f"Round-trip SI-SDR: {waveform_metrics.si_sdr(ctx):.2f} dB | "
      f"STOI: {perceptual.stoi(ctx):.4f} | "
      f"HASPI: {perceptual.haspi(ctx):.4f}"
    )
  return result


def main(argv: Sequence[str]) -> None:
  """CLI entrypoint for WAV NAP inversion."""
  del argv
  if _INPUT_WAV.value is None or _OUTPUT_WAV.value is None:
    raise ValueError("input_wav and output_wav are required")
  mode_val = _MODE.value
  resolved_mode: inversion.InversionMode | None = None
  if mode_val == "exact":
    resolved_mode = "exact"
  elif mode_val == "perceptual":
    resolved_mode = "perceptual"
  chunk_s = _CHUNK_SECONDS.value if _CHUNK_SECONDS.value > 0.0 else None
  run_invert_wav(
    input_wav=_INPUT_WAV.value,
    output_wav=_OUTPUT_WAV.value,
    sample_rate_hz=_SAMPLE_RATE_HZ.value,
    level_db_spl=_LEVEL_DB_SPL.value,
    source_ohc_health=_SOURCE_OHC_HEALTH.value,
    fitted_profile_path=_FITTED_PROFILE_PATH.value,
    mode=resolved_mode,
    adam_steps=_ADAM_STEPS.value,
    lbfgs_steps=_LBFGS_STEPS.value,
    chunk_seconds=chunk_s,
    neural_checkpoint=_NEURAL_CHECKPOINT.value,
  )


if __name__ == "__main__":
  define_flags()
  app.run(main)
