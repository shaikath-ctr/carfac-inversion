"""Clinical audiology standards, CARFAC fitting, and NAL-R baselines."""

import dataclasses
import functools
import statistics
from collections.abc import Sequence
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from hp_acoustic.audiology import audiogram as audiogram_lib
from hp_acoustic.audiology import fit as audiology_fit
from hp_acoustic.models import carfac
from jax_haspi_hasqi import nalr

from carfac_inversion import tpu

# Standard 6-frequency audiogram grid used by HASPI v2, HASQI v2, and NAL-R.
AUDIOMETRIC_6_HZ: tuple[float, ...] = (
  250.0,
  500.0,
  1000.0,
  2000.0,
  4000.0,
  6000.0,
)

# Full 10-frequency grid of Bisgaard, Vlaming, and Dahlquist (2010).
BISGAARD_FREQUENCIES_HZ: tuple[float, ...] = (
  250.0,
  375.0,
  500.0,
  750.0,
  1000.0,
  1500.0,
  2000.0,
  3000.0,
  4000.0,
  6000.0,
)

_BISGAARD_TO_6_IDX: tuple[int, ...] = (0, 2, 4, 6, 8, 9)

BISGAARD_AUDIOGRAMS_DB_HL: dict[str, tuple[float, ...]] = {
  "N0": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  "N1": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 15.0, 20.0, 30.0, 40.0),
  "N2": (20.0, 20.0, 20.0, 22.5, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0),
  "N3": (35.0, 35.0, 35.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0),
  "N4": (55.0, 55.0, 55.0, 55.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0),
  "N5": (65.0, 67.5, 70.0, 72.5, 75.0, 80.0, 80.0, 80.0, 80.0, 80.0),
  "N6": (75.0, 77.5, 80.0, 82.5, 85.0, 90.0, 90.0, 95.0, 100.0, 100.0),
  "N7": (90.0, 92.5, 95.0, 100.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0),
  "S1": (10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 15.0, 30.0, 55.0, 70.0),
  "S2": (20.0, 20.0, 20.0, 22.5, 25.0, 35.0, 55.0, 75.0, 95.0, 95.0),
  "S3": (30.0, 30.0, 35.0, 47.5, 60.0, 70.0, 75.0, 80.0, 80.0, 85.0),
}

# ISO 7029:2017 Table 1 coefficients across (250, 500, 1000, 2000, 4000, 6000, 8000) Hz.
_ISO7029_ALPHA: dict[str, tuple[float, ...]] = {
  "male": (1.39e-4, 4.59e-4, 7.02e-4, 1.56e-3, 3.40e-3, 4.53e-3, 5.06e-3),
  "female": (3.98e-4, 2.61e-4, 2.21e-4, 3.12e-4, 7.37e-4, 1.47e-3, 2.53e-3),
}
_ISO7029_BETA: dict[str, tuple[float, ...]] = {
  "male": (2.832, 2.537, 2.494, 2.404, 2.325, 2.315, 2.328),
  "female": (2.568, 2.708, 2.805, 2.792, 2.660, 2.539, 2.439),
}
_ISO7029_SU0 = np.array((5.0, 5.0, 5.2, 5.8, 6.8, 7.6, 8.2), dtype=np.float64)
_ISO7029_SL0 = np.array((4.2, 4.2, 4.4, 4.8, 5.5, 6.1, 6.6), dtype=np.float64)
_STD_NORMAL = statistics.NormalDist()
_HALF_OHC_REF_HL_6 = np.array(
  (21.0, 26.0, 30.0, 32.0, 33.0, 33.0), dtype=np.float32
)


def bisgaard_audiogram_6(code: str) -> np.ndarray:
  """Returns dB HL thresholds at ``AUDIOMETRIC_6_HZ`` for a Bisgaard code."""
  norm_code = code.strip().upper()
  if norm_code not in BISGAARD_AUDIOGRAMS_DB_HL:
    valid = ", ".join(sorted(BISGAARD_AUDIOGRAMS_DB_HL))
    raise ValueError(
      f"Unknown Bisgaard profile {code!r}; expected one of {valid}."
    )
  full = BISGAARD_AUDIOGRAMS_DB_HL[norm_code]
  return np.asarray([full[i] for i in _BISGAARD_TO_6_IDX], dtype=np.float32)


def iso7029_audiogram_6(
  age_years: float,
  sex: str = "male",
  percentile: float = 0.50,
) -> np.ndarray:
  """Computes ISO 7029:2017 presbycusis thresholds at ``AUDIOMETRIC_6_HZ``.

  Args:
    age_years: Listener age in years (relative shift begins above ``18``).
    sex: Biological sex table selector, ``"male"`` or ``"female"``.
    percentile: Population threshold percentile in ``[0.01, 0.99]`` where larger
      percentiles correspond to greater hearing loss.

  Returns:
    Threshold shifts in dB HL of shape ``(6,)`` (``float32``).
  """
  sex_key = sex.strip().lower()
  if sex_key not in _ISO7029_ALPHA:
    raise ValueError(f"sex must be 'male' or 'female', got {sex!r}.")
  if not (0.01 <= float(percentile) <= 0.99):
    raise ValueError(f"percentile must lie in [0.01, 0.99], got {percentile}.")
  if age_years < 0.0:
    raise ValueError(f"age_years must be non-negative, got {age_years}.")

  alpha = np.asarray(_ISO7029_ALPHA[sex_key], dtype=np.float64)
  beta = np.asarray(_ISO7029_BETA[sex_key], dtype=np.float64)
  dy = max(float(age_years) - 18.0, 0.0)
  h_md = alpha * (dy**beta)
  z = float(_STD_NORMAL.inv_cdf(float(percentile)))
  sigma = _ISO7029_SU0 + 0.18 * h_md if z >= 0.0 else _ISO7029_SL0 + 0.12 * h_md
  return np.maximum(h_md[:6] + z * sigma[:6], 0.0).astype(np.float32)


def _interpolate_aud6_to_cfs(
  audiogram_6_db_hl: Sequence[float] | np.ndarray,
  channel_freqs_hz: Sequence[float] | np.ndarray,
) -> np.ndarray:
  """Interpolates 6-band dB HL onto arbitrary CARFAC CFs in log-frequency."""
  aud6 = np.asarray(audiogram_6_db_hl, dtype=np.float64)
  if aud6.shape != (6,):
    raise ValueError(f"Expected 6 audiogram bands, got shape {aud6.shape}.")
  cfs = np.asarray(channel_freqs_hz, dtype=np.float64)
  if cfs.ndim != 1 or cfs.size == 0:
    raise ValueError(f"channel_freqs_hz must be non-empty 1D, got {cfs.shape}.")
  xp = np.log10(np.asarray(AUDIOMETRIC_6_HZ, dtype=np.float64))
  x_q = np.log10(np.clip(cfs, AUDIOMETRIC_6_HZ[0], AUDIOMETRIC_6_HZ[-1]))
  return np.interp(x_q, xp, aud6)


def audiogram_to_carfac_params(
  audiogram_6_db_hl: Sequence[float] | np.ndarray,
  channel_freqs_hz: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Maps a 6-band audiogram onto CARFAC ``(ohc_health, v_half_shift)``."""
  cfs = np.asarray(channel_freqs_hz, dtype=np.float64)
  loss_db = np.maximum(_interpolate_aud6_to_cfs(audiogram_6_db_hl, cfs), 0.0)
  log_frac = np.clip(
    (np.log10(np.maximum(cfs, 80.0)) - np.log10(80.0))
    / (np.log10(8000.0) - np.log10(80.0)),
    0.0,
    1.0,
  )
  cr = 1.25 + 2.25 * log_frac
  max_ohc_db = 1.25 * 70.0 * (1.0 - 1.0 / cr)
  ohc_loss_db = np.minimum(0.8 * loss_db, 0.8 * max_ohc_db)
  ihc_loss_db = np.maximum(loss_db - ohc_loss_db, 0.0)
  frac_ohc = np.clip(ohc_loss_db / np.maximum(0.8 * max_ohc_db, 1e-6), 0.0, 1.0)
  ohc_health = np.where(
    loss_db <= 1e-4, 1.0, np.clip(1.0 - (frac_ohc**0.85), 0.0, 1.0)
  ).astype(np.float32)
  fiber_scales = np.array([0.7, 1.0, 1.3], dtype=np.float32)
  v_half_shift = (
    0.00045 * ihc_loss_db[:, None] * fiber_scales[None, :]
  ).astype(np.float32)
  return ohc_health, v_half_shift


@dataclasses.dataclass(frozen=True)
class ClinicalListenerProfile:
  """Clinical audiogram and fitted CARFAC OHC/synaptic state for one ear."""

  name: str
  audiogram_6_db_hl: np.ndarray
  ohc_health: np.ndarray
  v_half_shift: np.ndarray
  channel_freqs_hz: np.ndarray
  sample_rate_hz: float

  @property
  def is_healthy(self) -> bool:
    """Returns ``True`` when the profile represents unimpaired normal hearing."""
    return bool(
      np.max(self.audiogram_6_db_hl) <= 1e-4
      and np.min(self.ohc_health) > 0.999
      and np.max(self.v_half_shift) <= 1e-7
    )

  def conditioning_features(self) -> np.ndarray:
    """Returns ``(n_channels, 3)`` float32 conditioning features for neural FiLM."""
    n_ch = int(self.channel_freqs_hz.shape[0])
    if self.is_healthy:
      return np.zeros((n_ch, 3), dtype=np.float32)
    ohc_deficit = (1.0 - np.asarray(self.ohc_health, dtype=np.float32)).clip(
      0.0, 1.0
    )
    ihc_feat = (
      np.mean(np.asarray(self.v_half_shift, dtype=np.float32), axis=-1) * 20.0
    )
    aud_norm = (
      _interpolate_aud6_to_cfs(
        self.audiogram_6_db_hl, self.channel_freqs_hz
      ).astype(np.float32)
      / 100.0
    )
    return np.stack([ohc_deficit, ihc_feat, aud_norm], axis=-1).astype(
      np.float32
    )


def profile_from_audiogram_6(
  name: str,
  audiogram_6_db_hl: Sequence[float] | np.ndarray,
  *,
  sample_rate_hz: float = 32000.0,
  channel_freqs_hz: np.ndarray | None = None,
  fit_steps: int = 0,
) -> ClinicalListenerProfile:
  """Constructs a :class:`ClinicalListenerProfile` from 6-band thresholds."""
  aud6 = np.maximum(np.asarray(audiogram_6_db_hl, dtype=np.float32), 0.0)
  if aud6.shape != (6,):
    raise ValueError(f"Expected 6 audiogram bands, got shape {aud6.shape}.")
  cfs = (
    np.asarray(channel_freqs_hz, dtype=np.float32)
    if channel_freqs_hz is not None
    else np.asarray(
      carfac.channel_frequencies_for_rate(float(sample_rate_hz)),
      dtype=np.float32,
    )
  )
  ohc, shift = audiogram_to_carfac_params(aud6, cfs)
  if fit_steps > 0 and float(np.max(aud6)) > 1e-4:
    aud_obj = audiogram_lib.from_tuples(
      list(zip(AUDIOMETRIC_6_HZ, [float(v) for v in aud6], strict=True)),
      side=audiogram_lib.Side.LEFT,
    )
    with tpu.host_cpu_context():
      fitted = audiology_fit.fit_ohc_health(
        aud_obj, sample_rate=float(sample_rate_hz), steps=int(fit_steps)
      )
    ohc = np.asarray(fitted.ohc_health, dtype=np.float32)
  return ClinicalListenerProfile(
    name=name,
    audiogram_6_db_hl=aud6,
    ohc_health=ohc,
    v_half_shift=shift,
    channel_freqs_hz=cfs,
    sample_rate_hz=float(sample_rate_hz),
  )


def build_clinical_profile(
  spec: str,
  sample_rate_hz: float = 32000.0,
  *,
  channel_freqs_hz: np.ndarray | None = None,
  fit_steps: int = 0,
) -> ClinicalListenerProfile:
  """Parses a profile spec string into a :class:`ClinicalListenerProfile`.

  Supported specs:
    - ``"healthy"`` or ``"N0"``
    - ``"bisgaard:N1"``..``"bisgaard:S3"`` or bare ``"N1"``..``"S3"``
    - ``"iso7029:<age>[:<sex>[:<percentile>]]"``
    - ``"flat:<ohc>"`` and ``"sloping"``
  """
  raw = spec.strip()
  low = raw.lower()
  cfs = (
    np.asarray(channel_freqs_hz, dtype=np.float32)
    if channel_freqs_hz is not None
    else np.asarray(
      carfac.channel_frequencies_for_rate(float(sample_rate_hz)),
      dtype=np.float32,
    )
  )
  n_ch = int(cfs.shape[0])

  if low in ("healthy", "n0", "bisgaard:n0"):
    return ClinicalListenerProfile(
      name="healthy",
      audiogram_6_db_hl=np.zeros((6,), dtype=np.float32),
      ohc_health=np.ones((n_ch,), dtype=np.float32),
      v_half_shift=np.zeros((n_ch, 3), dtype=np.float32),
      channel_freqs_hz=cfs,
      sample_rate_hz=float(sample_rate_hz),
    )

  if low.startswith("bisgaard:") or raw.upper() in BISGAARD_AUDIOGRAMS_DB_HL:
    code = raw.partition(":")[2] if low.startswith("bisgaard:") else raw
    norm_code = code.strip().upper()
    aud6 = bisgaard_audiogram_6(norm_code)
    return profile_from_audiogram_6(
      f"bisgaard:{norm_code}",
      aud6,
      sample_rate_hz=sample_rate_hz,
      channel_freqs_hz=cfs,
      fit_steps=fit_steps,
    )

  if low.startswith("iso7029:"):
    parts = raw.split(":")
    if len(parts) < 2 or len(parts) > 4:
      raise ValueError(
        f"Expected 'iso7029:<age>[:<sex>[:<percentile>]]', got {spec!r}."
      )
    age = float(parts[1])
    sex = parts[2].strip().lower() if len(parts) >= 3 and parts[2] else "male"
    pct = float(parts[3]) if len(parts) == 4 and parts[3] else 0.50
    aud6 = iso7029_audiogram_6(age, sex=sex, percentile=pct)
    return profile_from_audiogram_6(
      f"iso7029:{age:g}:{sex}:{pct:g}",
      aud6,
      sample_rate_hz=sample_rate_hz,
      channel_freqs_hz=cfs,
      fit_steps=fit_steps,
    )

  if low.startswith("flat:"):
    ohc_val = float(np.clip(float(raw.partition(":")[2]), 0.0, 1.0))
    ohc = np.full((n_ch,), ohc_val, dtype=np.float32)
    shift = np.zeros((n_ch, 3), dtype=np.float32)
    scale = (1.0 - ohc_val) / 0.5
    aud6 = np.clip(_HALF_OHC_REF_HL_6 * scale, 0.0, 70.0).astype(np.float32)
    return ClinicalListenerProfile(
      name=f"flat:{ohc_val:g}",
      audiogram_6_db_hl=aud6,
      ohc_health=ohc,
      v_half_shift=shift,
      channel_freqs_hz=cfs,
      sample_rate_hz=float(sample_rate_hz),
    )

  if low == "sloping":
    frac = np.clip(
      (np.log10(np.maximum(cfs, 20.0)) - np.log10(250.0))
      / (np.log10(6000.0) - np.log10(250.0)),
      0.0,
      1.0,
    )
    ohc = (0.95 - 0.75 * frac).astype(np.float32)
    shift = np.zeros((n_ch, 3), dtype=np.float32)
    xp = np.sort(cfs)
    fp = (1.0 - ohc[np.argsort(cfs)]) * 55.0
    aud6 = np.interp(np.asarray(AUDIOMETRIC_6_HZ), xp, fp).astype(np.float32)
    return ClinicalListenerProfile(
      name="sloping",
      audiogram_6_db_hl=aud6,
      ohc_health=ohc,
      v_half_shift=shift,
      channel_freqs_hz=cfs,
      sample_rate_hz=float(sample_rate_hz),
    )

  raise ValueError(
    f"Unsupported clinical profile spec {spec!r}; expected 'healthy', "
    "'bisgaard:<code>', 'iso7029:<age>[:<sex>[:<pct>]]', 'flat:<ohc>', or 'sloping'."
  )


def build_cochlea_from_profile(
  profile: ClinicalListenerProfile,
  *,
  base_cochlea: carfac.CarfacCochleaModel | None = None,
) -> carfac.CarfacCochleaModel:
  """Constructs an impaired ``CarfacCochleaModel`` matching ``profile``."""
  if base_cochlea is None:
    return carfac.build(
      fs=int(round(profile.sample_rate_hz)),
      ohc_health=profile.ohc_health,
      v_half_shift=profile.v_half_shift,
    )
  ohc_j = jnp.asarray(profile.ohc_health, dtype=jnp.float32)
  shift_j = jnp.asarray(profile.v_half_shift, dtype=jnp.float32)
  new_ears = []
  for ear_w in base_cochlea.weights.ears:
    car_w = dataclasses.replace(ear_w.car, ohc_health=ohc_j)
    if ear_w.syn is not None:
      syn_w = dataclasses.replace(
        ear_w.syn, v_halfs=ear_w.syn.v_halfs + shift_j
      )
      new_ears.append(dataclasses.replace(ear_w, car=car_w, syn=syn_w))
    else:
      new_ears.append(dataclasses.replace(ear_w, car=car_w))
  return dataclasses.replace(
    base_cochlea,
    weights=dataclasses.replace(base_cochlea.weights, ears=new_ears),
  )


@functools.partial(jax.jit, static_argnames=("n_fir", "sample_rate_hz"))
def _apply_nalr_jit(
  wav: jnp.ndarray,
  aud6: jnp.ndarray,
  *,
  n_fir: int,
  sample_rate_hz: float,
) -> jnp.ndarray:
  """JIT-compiled NAL-R FIR design and linear-phase convolution."""
  fir, _ = nalr.build(
    aud6,
    n_fir=n_fir,
    sample_rate=sample_rate_hz,
  )
  conv = cast(jnp.ndarray, nalr.apply(fir, wav))
  delay = int(n_fir) // 2
  return conv[delay : delay + wav.shape[0]]


def apply_nalr(
  waveform: np.ndarray,
  audiogram_6_db_hl: Sequence[float] | np.ndarray,
  sample_rate_hz: float,
  *,
  n_fir: int = 140,
  mpo_clip: float = 0.99,
) -> np.ndarray:
  """Applies linear-phase NAL-R amplification clipped at ``mpo_clip``."""
  wav = np.asarray(waveform, dtype=np.float32)
  if wav.ndim != 1:
    raise ValueError(f"Expected 1D waveform for NAL-R, got shape {wav.shape}.")
  aud6 = np.asarray(audiogram_6_db_hl, dtype=np.float32)
  if aud6.shape != (6,):
    raise ValueError(f"Expected 6 audiogram bands, got shape {aud6.shape}.")
  with tpu.host_cpu_context():
    aligned = _apply_nalr_jit(
      jnp.asarray(wav, dtype=jnp.float32),
      jnp.asarray(aud6, dtype=jnp.float32),
      n_fir=int(n_fir),
      sample_rate_hz=float(sample_rate_hz),
    )
  out = np.asarray(aligned, dtype=np.float32)
  if mpo_clip > 0.0:
    out = np.clip(out, -float(mpo_clip), float(mpo_clip))
  return out
