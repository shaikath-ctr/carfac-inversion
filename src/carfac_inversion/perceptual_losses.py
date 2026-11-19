"""Float32-safe differentiable HASQI v2 and HASPI v2 perceptual losses."""

import functools
import math
from typing import Any
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
from jax_haspi_hasqi import ear_model
from jax_haspi_hasqi import filters as hasqi_filters
from jax_haspi_hasqi import hasqi as hasqi_lib
from jax_haspi_hasqi import modulation as mod_lib
from jax_haspi_hasqi import nalr
from jax_haspi_hasqi import neural_net

# Safe float32 floor so squared products (>= 1e-32) stay in normal float32 range.
_EPS_F32 = 1e-16
_DIV_EPS_F32 = 1e-12
_LOG_EPS_F32 = 1e-15
_STD_EPS_F32 = 1e-20

_HASQI_LOUDNESS_WEIGHT = 0.579
_HASQI_SLOPE_WEIGHT = 0.421
_HASQI_LOUDNESS_SCALE = 2.5
_HASQI_SILENCE_THRESHOLD = 2.5
_HASQI_SEGMENT_COVARIANCE = 16
_HASQI_CEPSTRAL_COEFFS = 6

_HASPI_LOW_PASS_CUTOFF = 320.0
_HASPI_CEPSTRAL_BASIS = 6
_HASPI_SILENCE_THRESHOLD = 2.5
_HASPI_NERVE_DITHER = 0.1


def _safe_rms(x: jnp.ndarray, denom_len: jnp.ndarray | int) -> jnp.ndarray:
  return jnp.sqrt(jnp.sum(x**2) / jnp.maximum(denom_len, 1) + 1e-24)


def _safe_envelope_sl(
  envelope: jnp.ndarray,
  motion: jnp.ndarray,
  attn_ihc: jnp.ndarray,
  level1: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  in_db = jnp.maximum(
    level1 - attn_ihc + 20.0 * jnp.log10(jnp.maximum(envelope, _LOG_EPS_F32)),
    0.0,
  )
  gain = (in_db + _DIV_EPS_F32) / jnp.maximum(envelope, _DIV_EPS_F32)
  return in_db, gain * motion


def _safe_ihc_adaptation(
  envelope_db: jnp.ndarray,
  motion: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  out, _ = ear_model.inner_hair_cell_adaptation(
    envelope_db, jnp.zeros_like(envelope_db)
  )
  gain = (out + _DIV_EPS_F32) / jnp.maximum(envelope_db, _DIV_EPS_F32)
  return out, gain * motion


def _safe_ear_model_aligned(
  reference: jnp.ndarray,
  processed: jnp.ndarray,
  hearing_loss: jnp.ndarray,
  itype: int,
  level1: float,
  nchan: int = 32,
  m_delay: int = 1,
  valid_len: jnp.ndarray | None = None,
) -> tuple[
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
]:
  """Differentiable float32-safe counterpart of ``ear_model._ear_model_aligned``."""
  reference = jnp.asarray(reference, dtype=jnp.float32)
  processed = jnp.asarray(processed, dtype=jnp.float32)
  hearing_loss = jnp.asarray(hearing_loss, dtype=jnp.float32)
  nsamp = reference.shape[0]
  eff_len: jnp.ndarray | int = nsamp if valid_len is None else valid_len
  valid_mask = (
    jnp.ones((nsamp,), dtype=bool)
    if valid_len is None
    else (jnp.arange(nsamp) < valid_len)
  )

  centre_freq_np = ear_model.center_frequency(nchan)
  centre_freq = jnp.asarray(centre_freq_np, dtype=jnp.float32)
  attn_ohc_y, bandwidth_min_y, low_knee_y, ratio_y, attn_ihc_y = (
    ear_model.loss_parameters(hearing_loss, centre_freq_np)
  )
  loss_x = jnp.zeros_like(hearing_loss) if itype == 0 else hearing_loss
  attn_ohc_x, bandwidth_min_x, low_knee_x, ratio_x, attn_ihc_x = (
    ear_model.loss_parameters(loss_x, centre_freq_np)
  )
  _, bandwidth_1, _, _, _ = ear_model.loss_parameters(
    jnp.full(6, 100.0, dtype=jnp.float32), centre_freq_np
  )

  if itype == 1:
    nalr_taps = 140
    fir, _ = nalr.build(hearing_loss, nalr_taps, 24000.0)
    reference = nalr.apply(fir, reference)[nalr_taps : nalr_taps + nsamp]
    reference = jnp.where(valid_mask, reference, 0.0)

  reference_mid = jnp.where(valid_mask, ear_model.middle_ear(reference), 0.0)
  processed_mid = jnp.where(valid_mask, ear_model.middle_ear(processed), 0.0)

  def channel(
    index: jnp.ndarray,
  ) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
  ]:
    reference_control, _, processed_control, _ = (
      ear_model.gammatone_basilar_membrane(
        reference_mid,
        bandwidth_1[index],
        processed_mid,
        bandwidth_1[index],
        centre_freq[index],
      )
    )
    reference_control = jnp.where(valid_mask, reference_control, 0.0)
    processed_control = jnp.where(valid_mask, processed_control, 0.0)
    reference_bandwidth = ear_model.bandwidth_adjust(
      reference_control,
      bandwidth_min_x[index],
      bandwidth_1[index],
      level1,
      valid_len=valid_len,
    )
    processed_bandwidth = ear_model.bandwidth_adjust(
      processed_control,
      bandwidth_min_y[index],
      bandwidth_1[index],
      level1,
      valid_len=valid_len,
    )
    xenv, xbm, yenv, ybm = ear_model.gammatone_basilar_membrane(
      reference_mid,
      reference_bandwidth,
      processed_mid,
      processed_bandwidth,
      centre_freq[index],
    )
    xenv = jnp.where(valid_mask, xenv, 0.0)
    xbm = jnp.where(valid_mask, xbm, 0.0)
    yenv = jnp.where(valid_mask, yenv, 0.0)
    ybm = jnp.where(valid_mask, ybm, 0.0)

    ref_comp, ref_bm = ear_model.env_compress_basilar_membrane(
      xenv,
      xbm,
      reference_control,
      attn_ohc_x[index],
      low_knee_x[index],
      ratio_x[index],
      level1,
    )
    proc_comp, proc_bm = ear_model.env_compress_basilar_membrane(
      yenv,
      ybm,
      processed_control,
      attn_ohc_y[index],
      low_knee_y[index],
      ratio_y[index],
      level1,
    )
    proc_comp = ear_model.envelope_align(
      ref_comp, proc_comp, valid_len=valid_len
    )
    proc_bm = ear_model.envelope_align(ref_bm, proc_bm, valid_len=valid_len)

    ref_comp, ref_bm = _safe_envelope_sl(
      ref_comp, ref_bm, attn_ihc_x[index], level1
    )
    proc_comp, proc_bm = _safe_envelope_sl(
      proc_comp, proc_bm, attn_ihc_y[index], level1
    )
    ref_comp = jnp.where(valid_mask, ref_comp, 0.0)
    ref_bm = jnp.where(valid_mask, ref_bm, 0.0)
    proc_comp = jnp.where(valid_mask, proc_comp, 0.0)
    proc_bm = jnp.where(valid_mask, proc_bm, 0.0)

    ref_db, ref_bm = _safe_ihc_adaptation(ref_comp, ref_bm)
    proc_db, proc_bm = _safe_ihc_adaptation(proc_comp, proc_bm)
    ref_db = jnp.where(valid_mask, ref_db, 0.0)
    ref_bm = jnp.where(valid_mask, ref_bm, 0.0)
    proc_db = jnp.where(valid_mask, proc_db, 0.0)
    proc_bm = jnp.where(valid_mask, proc_bm, 0.0)
    return (
      ref_db,
      ref_bm,
      proc_db,
      proc_bm,
      _safe_rms(xenv, eff_len),
      _safe_rms(yenv, eff_len),
      _safe_rms(reference_control, eff_len),
      _safe_rms(processed_control, eff_len),
      reference_bandwidth,
    )

  (
    reference_db,
    reference_bm,
    processed_db,
    processed_bm,
    reference_average,
    processed_average,
    reference_control_average,
    processed_control_average,
    reference_bandwidth,
  ) = jax.vmap(channel)(jnp.arange(nchan))

  if m_delay > 0:
    reference_db = jnp.where(
      valid_mask[None, :],
      ear_model.group_delay_compensate(
        reference_db, bandwidths=reference_bandwidth, centre_freq=centre_freq
      ),
      0.0,
    )
    processed_db = jnp.where(
      valid_mask[None, :],
      ear_model.group_delay_compensate(
        processed_db, bandwidths=reference_bandwidth, centre_freq=centre_freq
      ),
      0.0,
    )
    reference_bm = jnp.where(
      valid_mask[None, :],
      ear_model.group_delay_compensate(
        reference_bm, bandwidths=reference_bandwidth, centre_freq=centre_freq
      ),
      0.0,
    )
    processed_bm = jnp.where(
      valid_mask[None, :],
      ear_model.group_delay_compensate(
        processed_bm, bandwidths=reference_bandwidth, centre_freq=centre_freq
      ),
      0.0,
    )

  reference_sl = ear_model.convert_rms_to_sl(
    reference_average,
    reference_control_average,
    attn_ohc_x,
    low_knee_x,
    ratio_x,
    attn_ihc_x,
    level1,
  )
  processed_sl = ear_model.convert_rms_to_sl(
    processed_average,
    processed_control_average,
    attn_ohc_y,
    low_knee_y,
    ratio_y,
    attn_ihc_y,
    level1,
  )
  return (
    reference_db,
    reference_bm,
    processed_db,
    processed_bm,
    reference_sl,
    processed_sl,
  )


def _safe_mel_cepstrum_correlation(
  reference: jnp.ndarray,
  distorted: jnp.ndarray,
  threshold: float = _HASQI_SILENCE_THRESHOLD,
) -> jnp.ndarray:
  nbands = reference.shape[0]
  basis = jnp.asarray(
    hasqi_lib._mel_cepstrum_basis(nbands, _HASQI_CEPSTRAL_COEFFS),
    dtype=jnp.float32,
  )
  loudness = (
    jnp.sum(jnp.power(jnp.float32(10.0), reference / 20.0), axis=0) / nbands
  )
  loudness = 20.0 * jnp.log10(jnp.maximum(loudness, jnp.float32(_LOG_EPS_F32)))
  keep = loudness > threshold
  nsamp = jnp.sum(keep)
  denom = jnp.maximum(nsamp, 1).astype(jnp.float32)

  def _cep(env: jnp.ndarray) -> jnp.ndarray:
    coeffs = basis.T @ jnp.where(keep, env, 0.0)
    mean = jnp.sum(coeffs, axis=1, keepdims=True) / denom
    return jnp.where(keep, coeffs - mean, 0.0)

  ref_cep = _cep(reference)
  dist_cep = _cep(distorted)
  ref_sum = jnp.sum(ref_cep**2, axis=1)
  dist_sum = jnp.sum(dist_cep**2, axis=1)
  valid = (ref_sum >= _EPS_F32) & (dist_sum >= _EPS_F32)
  denom_corr = jnp.sqrt(jnp.where(valid, ref_sum * dist_sum, 1.0))
  corrs = jnp.where(
    valid,
    jnp.abs(jnp.sum(ref_cep * dist_cep, axis=1)) / denom_corr,
    0.0,
  )
  average = jnp.sum(corrs[1:]) / (_HASQI_CEPSTRAL_COEFFS - 1)
  return jnp.where(nsamp > 1, average, 0.0)


def _safe_spectrum_diff(
  reference_sl: jnp.ndarray, processed_sl: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  nbands = reference_sl.shape[0]
  ref = jnp.power(jnp.float32(10.0), reference_sl / 20.0)
  proc = jnp.power(jnp.float32(10.0), processed_sl / 20.0)
  ref = ref / jnp.maximum(jnp.sum(ref), _EPS_F32)
  proc = proc / jnp.maximum(jnp.sum(proc), _EPS_F32)

  def _smooth_std(diff: jnp.ndarray) -> jnp.ndarray:
    centered = diff - jnp.mean(diff)
    return nbands * jnp.sqrt(jnp.mean(centered**2) + _STD_EPS_F32)

  d_loud_std = _smooth_std(ref - proc)
  d_slope_std = _smooth_std(jnp.diff(ref) - jnp.diff(proc))
  return d_loud_std, d_slope_std


def _safe_bm_covary(
  reference_bm: jnp.ndarray,
  processed_bm: jnp.ndarray,
  segment_size: int = _HASQI_SEGMENT_COVARIANCE,
  sample_rate: float = 24000.0,
  valid_len: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes normalized BM cross-covariance with float32-safe variance gating."""
  reference_bm = jnp.asarray(reference_bm, dtype=jnp.float32)
  processed_bm = jnp.asarray(processed_bm, dtype=jnp.float32)
  max_lag = int(np.around(1.0 * (0.001 * sample_rate)))
  nwin = int(np.around(segment_size * (0.001 * sample_rate)))
  nwin += nwin % 2 == 1
  nhalf = nwin // 2

  window_np = np.hanning(nwin)
  half_window_np = window_np[nhalf:nwin]

  def _inv_corr(vals: np.ndarray) -> jnp.ndarray:
    corr = np.correlate(vals, vals, "full")
    start = len(vals) - 1 - max_lag
    return jnp.asarray(
      1.0 / corr[start : max_lag + len(vals)], dtype=jnp.float32
    )

  window_corr = _inv_corr(window_np)
  half_corr = _inv_corr(half_window_np)
  window_pow = jnp.float32(1.0 / np.sum(window_np**2))
  half_pow = jnp.float32(1.0 / np.sum(half_window_np**2))
  window = jnp.asarray(window_np, dtype=jnp.float32)
  half_window = jnp.asarray(half_window_np, dtype=jnp.float32)

  npts = reference_bm.shape[1]
  starts, max_nseg = hasqi_lib._segment_layout(npts, nwin)

  def _covary(
    ref_seg: jnp.ndarray,
    proc_seg: jnp.ndarray,
    power: jnp.ndarray,
    correction: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    ref_seg = ref_seg - jnp.mean(ref_seg, axis=-1, keepdims=True)
    proc_seg = proc_seg - jnp.mean(proc_seg, axis=-1, keepdims=True)
    ref_ms = jnp.sum(ref_seg**2, axis=-1) * power
    proc_ms = jnp.sum(proc_seg**2, axis=-1) * power
    length = ref_seg.shape[-1]
    lags = jnp.arange(length - 1 - max_lag, max_lag + length)
    corr = hasqi_filters.correlate_at_lags(ref_seg, proc_seg, lags)
    peak = jnp.max(jnp.abs(corr * correction))
    valid = (ref_ms > _EPS_F32) & (proc_ms > _EPS_F32)
    cov = jnp.where(
      valid,
      peak / jnp.sqrt(jnp.where(valid, ref_ms * proc_ms, 1.0)),
      0.0,
    )
    return cov, ref_ms

  over_ch = jax.vmap(_covary, in_axes=(0, 0, None, None))
  over_seg = jax.vmap(
    jax.vmap(_covary, in_axes=(0, 0, None, None)), in_axes=(0, 0, None, None)
  )
  first = over_ch(
    reference_bm[:, :nhalf] * half_window,
    processed_bm[:, :nhalf] * half_window,
    half_pow,
    half_corr,
  )
  middle = over_seg(
    hasqi_lib._gather_segments(reference_bm, starts, nwin) * window,
    hasqi_lib._gather_segments(processed_bm, starts, nwin) * window,
    window_pow,
    window_corr,
  )
  if valid_len is None:
    last_start = np.array([nhalf * (max_nseg - 1)])
    nseg = None
  else:
    nseg = jnp.where(
      valid_len > 0, 1 + (valid_len // nwin) + ((valid_len - nhalf) // nwin), 0
    )
    last_start = jnp.array([jnp.maximum(nhalf * (nseg - 1), 0)])

  last = over_ch(
    hasqi_lib._gather_segments(reference_bm, last_start, nhalf)[:, 0]
    * window[:nhalf],
    hasqi_lib._gather_segments(processed_bm, last_start, nhalf)[:, 0]
    * window[:nhalf],
    half_pow,
    half_corr,
  )

  def _stack(idx: int) -> jnp.ndarray:
    out = jnp.concatenate(
      [first[idx][:, None], middle[idx], last[idx][:, None]], axis=1
    )
    if valid_len is not None and nseg is not None:
      s_idx = jnp.arange(max_nseg)
      out = jnp.where(s_idx[None, :] == (nseg - 1), last[idx][:, None], out)
      out = jnp.where(s_idx[None, :] < nseg, out, 0.0)
    return out

  return jnp.clip(_stack(0), 0.0, 1.0), 2.0 * _stack(1)


def _safe_modulation_cross_correlation(
  ref_mod: jnp.ndarray, proc_mod: jnp.ndarray, nkept: jnp.ndarray
) -> jnp.ndarray:
  nsamp = ref_mod.shape[2]
  enough = nkept > 1
  active = jnp.where(enough, nkept, 0)
  denom = jnp.maximum(active, 1).astype(jnp.float32)
  mask = (jnp.arange(nsamp) < active)[None, None, :]
  ref_m = jnp.where(mask, ref_mod, 0.0)
  proc_m = jnp.where(mask, proc_mod, 0.0)
  ref_c = jnp.where(
    mask, ref_m - jnp.sum(ref_m, axis=2, keepdims=True) / denom, 0.0
  )
  proc_c = jnp.where(
    mask, proc_m - jnp.sum(proc_m, axis=2, keepdims=True) / denom, 0.0
  )
  ref_sum = jnp.sum(ref_c**2, axis=2)
  proc_sum = jnp.sum(proc_c**2, axis=2)
  valid = (ref_sum >= _EPS_F32) & (proc_sum >= _EPS_F32)
  cov = jnp.where(
    valid,
    jnp.abs(jnp.sum(ref_c * proc_c, axis=2))
    / jnp.sqrt(jnp.where(valid, ref_sum * proc_sum, 1.0)),
    0.0,
  )
  return jnp.where(enough, jnp.mean(cov[1:6], axis=0), 0.0)


def _prepare_24k(
  reference: jnp.ndarray,
  processed: jnp.ndarray,
  sample_rate_hz: float,
  align_inputs: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
  ref_24 = ear_model.resample_24khz(
    jnp.asarray(reference, dtype=jnp.float32), sample_rate_hz
  )
  proc_24 = ear_model.resample_24khz(
    jnp.asarray(processed, dtype=jnp.float32), sample_rate_hz
  )
  n = min(ref_24.shape[0], proc_24.shape[0])
  ref_24 = ref_24[:n]
  proc_24 = proc_24[:n]
  if not align_inputs:
    return ref_24, proc_24, None
  aligned_ref, aligned_proc, (start, stop, _) = ear_model.input_align(
    ref_24, proc_24
  )
  valid_len = jnp.maximum(stop - start + 1, 0)
  return aligned_ref, aligned_proc, valid_len


def differentiable_hasqi_v2(
  reference: jnp.ndarray,
  processed: jnp.ndarray,
  hearing_loss_db: jnp.ndarray,
  *,
  sample_rate_hz: float = 32000.0,
  level1: float = 104.0 + 20.0 * math.log10(math.sqrt(2.0)),
  align_inputs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Differentiable float32-safe HASQI v2 ``(combined, nonlinear, linear)``."""
  ref_24, proc_24, valid_len = _prepare_24k(
    reference, processed, sample_rate_hz, align_inputs
  )
  hl = jnp.asarray(hearing_loss_db, dtype=jnp.float32)
  ref_db, ref_bm, proc_db, proc_bm, ref_sl, proc_sl = _safe_ear_model_aligned(
    ref_24, proc_24, hl, itype=1, level1=level1, valid_len=valid_len
  )
  ref_sm = hasqi_lib.env_smooth(
    ref_db, _HASQI_SEGMENT_COVARIANCE, 24000.0, valid_len=valid_len
  )
  proc_sm = hasqi_lib.env_smooth(
    proc_db, _HASQI_SEGMENT_COVARIANCE, 24000.0, valid_len=valid_len
  )
  cep_corr = _safe_mel_cepstrum_correlation(
    ref_sm, proc_sm, _HASQI_SILENCE_THRESHOLD
  )
  cross_cov, ref_ms = _safe_bm_covary(
    ref_bm, proc_bm, _HASQI_SEGMENT_COVARIANCE, 24000.0, valid_len=valid_len
  )
  _, sync_cov = hasqi_lib.ave_covary2(
    cross_cov, ref_ms, _HASQI_SILENCE_THRESHOLD
  )
  d_loud_std, d_slope_std = _safe_spectrum_diff(ref_sl, proc_sl)
  d_loud = jnp.clip(1.0 - d_loud_std / _HASQI_LOUDNESS_SCALE, 0.0, 1.0)
  d_slope = jnp.clip(1.0 - d_slope_std, 0.0, 1.0)
  nonlinear = (cep_corr**2) * sync_cov[4]
  linear = _HASQI_LOUDNESS_WEIGHT * d_loud + _HASQI_SLOPE_WEIGHT * d_slope
  return nonlinear * linear, nonlinear, linear


def differentiable_haspi_v2(
  reference: jnp.ndarray,
  processed: jnp.ndarray,
  hearing_loss_db: jnp.ndarray,
  *,
  sample_rate_hz: float = 32000.0,
  level1: float = 104.0 + 20.0 * math.log10(math.sqrt(2.0)),
  align_inputs: bool = False,
) -> jnp.ndarray:
  """Differentiable float32-safe HASPI v2 intelligibility score in ``[0, 1]``."""
  ref_24, proc_24, valid_len = _prepare_24k(
    reference, processed, sample_rate_hz, align_inputs
  )
  hl = jnp.asarray(hearing_loss_db, dtype=jnp.float32)
  ref_db, _, proc_db, _, _, _ = _safe_ear_model_aligned(
    ref_24, proc_24, hl, itype=0, level1=level1, valid_len=valid_len
  )
  freq_sub = 8.0 * _HASPI_LOW_PASS_CUTOFF
  ref_lp, proc_lp = mod_lib.env_filter(
    ref_db,
    proc_db,
    _HASPI_LOW_PASS_CUTOFF,
    freq_sub,
    24000.0,
    valid_len=valid_len,
  )
  ref_cep, proc_cep, nkept = mod_lib.cepstral_correlation_coef(
    ref_lp,
    proc_lp,
    _HASPI_SILENCE_THRESHOLD,
    _HASPI_NERVE_DITHER,
    _HASPI_CEPSTRAL_BASIS,
  )
  ref_mod, proc_mod, _ = mod_lib.fir_modulation_filter(
    ref_cep, proc_cep, freq_sub
  )
  corrs = _safe_modulation_cross_correlation(ref_mod, proc_mod, nkept)
  return jnp.where(
    nkept > 1,
    neural_net.intelligibility(corrs),
    jnp.float32(0.0),
  )


@functools.lru_cache(maxsize=4)
def _middle_ear_fir_np(
  num_taps: int = 512, sample_rate: float = 24000.0
) -> np.ndarray:
  """Cached FIR impulse response of the 24 kHz HASPI/HASQI middle-ear filter."""
  low_b, low_a = cast(
    tuple[np.ndarray, np.ndarray],
    scipy.signal.butter(1, 5000.0 / (0.5 * sample_rate), output="ba"),
  )
  high_b, high_a = cast(
    tuple[np.ndarray, np.ndarray],
    scipy.signal.butter(2, 350.0 / (0.5 * sample_rate), "high", output="ba"),
  )
  imp = np.zeros(num_taps, dtype=np.float64)
  imp[0] = 1.0
  low_ir = cast(np.ndarray, scipy.signal.lfilter(low_b, low_a, imp))
  ir = cast(np.ndarray, scipy.signal.lfilter(high_b, high_a, low_ir))
  return np.asarray(ir, dtype=np.float32)


def _fast_middle_ear(signal: jnp.ndarray) -> jnp.ndarray:
  taps = jnp.asarray(_middle_ear_fir_np(), dtype=jnp.float32)
  return jnp.convolve(signal, taps, mode="full")[: signal.shape[0]]


def _fast_resample_24khz(
  signal: jnp.ndarray, sample_rate_hz: float
) -> jnp.ndarray:
  sig = jnp.asarray(signal, dtype=jnp.float32)
  src_khz = int(round(sample_rate_hz / 1000.0))
  if src_khz == 24:
    return sig
  return jnp.asarray(
    hasqi_filters.resample_poly(sig, 24, src_khz), dtype=jnp.float32
  )


def _gammatone_one(
  signal: jnp.ndarray,
  bandwidth: jnp.ndarray,
  centre_freq: jnp.ndarray,
  sample_rate: float = 24000.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  sincf, coscf = ear_model._carrier(signal.shape[0], centre_freq, sample_rate)
  num, _, gain, pole = ear_model._gammatone_coefficients(
    bandwidth, centre_freq, sample_rate
  )
  real = hasqi_filters.cascaded_one_pole(pole, num, signal * coscf)
  imag = hasqi_filters.cascaded_one_pole(pole, num, signal * sincf)
  env = gain * ear_model._safe_sqrt(real * real + imag * imag)
  bm = gain * (real * coscf + imag * sincf)
  return env, bm


def _ear_frontend_single(
  mid_1d: jnp.ndarray,
  hearing_loss_6: jnp.ndarray,
  level1: float,
  nchan: int = 32,
) -> tuple[
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
  jnp.ndarray,
]:
  """Single-ear 32-band gammatone + OHC compression front-end @ 24 kHz."""
  nsamp = mid_1d.shape[0]
  hl = jnp.asarray(hearing_loss_6, dtype=jnp.float32)
  centre_freq_np = ear_model.center_frequency(nchan)
  centre_freq = jnp.asarray(centre_freq_np, dtype=jnp.float32)
  attn_ohc, bandwidth_min, low_knee, ratio, attn_ihc = (
    ear_model.loss_parameters(hl, centre_freq_np)
  )
  _, bandwidth_1, _, _, _ = ear_model.loss_parameters(
    jnp.full(6, 100.0, dtype=jnp.float32), centre_freq_np
  )

  def _channel(
    index: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    ctrl_env, _ = _gammatone_one(mid_1d, bandwidth_1[index], centre_freq[index])
    bw = ear_model.bandwidth_adjust(
      ctrl_env, bandwidth_min[index], bandwidth_1[index], level1
    )
    env, bm = _gammatone_one(mid_1d, bw, centre_freq[index])
    comp_raw, bm_raw = ear_model.env_compress_basilar_membrane(
      env,
      bm,
      ctrl_env,
      attn_ohc[index],
      low_knee[index],
      ratio[index],
      level1,
    )
    return (
      comp_raw,
      bm_raw,
      _safe_rms(env, nsamp),
      _safe_rms(ctrl_env, nsamp),
      bw,
    )

  comp_raw, bm_raw, avg_env, avg_ctrl, bw = jax.vmap(_channel)(
    jnp.arange(nchan)
  )
  sl = ear_model.convert_rms_to_sl(
    avg_env, avg_ctrl, attn_ohc, low_knee, ratio, attn_ihc, level1
  )
  return comp_raw, bm_raw, sl, bw, attn_ihc, centre_freq


def _post_ear_vmap(
  comp_raw: jnp.ndarray,
  bm_raw: jnp.ndarray,
  attn_ihc: jnp.ndarray,
  level1: float,
  ref_bw: jnp.ndarray,
  centre_freq: jnp.ndarray,
  *,
  with_bm: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  def _post(
    c_r: jnp.ndarray, b_r: jnp.ndarray, a_i: jnp.ndarray
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    c_sl, b_sl = _safe_envelope_sl(c_r, b_r, a_i, level1)
    return _safe_ihc_adaptation(c_sl, b_sl)

  db, bm = jax.vmap(_post)(comp_raw, bm_raw, attn_ihc)
  db = ear_model.group_delay_compensate(
    db, bandwidths=ref_bw, centre_freq=centre_freq
  )
  if with_bm:
    bm = ear_model.group_delay_compensate(
      bm, bandwidths=ref_bw, centre_freq=centre_freq
    )
  return db, bm


def batched_hasqi_haspi_loss(
  reference_b: jnp.ndarray,
  processed_b: jnp.ndarray,
  hearing_loss_b: jnp.ndarray,
  *,
  mask: jnp.ndarray | None = None,
  sample_rate_hz: float = 32000.0,
  level1: float = 104.0 + 20.0 * math.log10(math.sqrt(2.0)),
  hasqi_weight: float = 0.05,
  haspi_weight: float = 0.05,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  """Computes checkpointed batched perceptual loss ``w_q*(1-HASQI) + w_p*(1-HASPI)``."""
  w_q = float(hasqi_weight)
  w_p = float(haspi_weight)

  def _ref_features(ref_1d: jnp.ndarray, hl_1d: jnp.ndarray) -> tuple[Any, ...]:
    ref_24 = jax.lax.stop_gradient(_fast_resample_24khz(ref_1d, sample_rate_hz))
    nsamp = ref_24.shape[0]
    hl = jnp.asarray(hl_1d, dtype=jnp.float32)

    if w_q > 0.0:
      nalr_taps = 140
      fir, _ = nalr.build(hl, nalr_taps, 24000.0)
      ref_nalr = nalr.apply(fir, ref_24)[nalr_taps : nalr_taps + nsamp]
      ref_mid_q = _fast_middle_ear(ref_nalr)
      ref_comp_q, ref_bm_raw_q, ref_sl_q, ref_bw_q, ref_ihc_q, cf = (
        _ear_frontend_single(ref_mid_q, hl, level1)
      )
      ref_db_q, ref_bm_q = _post_ear_vmap(
        ref_comp_q, ref_bm_raw_q, ref_ihc_q, level1, ref_bw_q, cf, with_bm=True
      )
      ref_sm_q = hasqi_lib.env_smooth(
        ref_db_q, _HASQI_SEGMENT_COVARIANCE, 24000.0
      )
      q_bundle: tuple[jnp.ndarray, ...] = (
        ref_comp_q,
        ref_bm_raw_q,
        ref_sm_q,
        ref_bm_q,
        ref_sl_q,
        ref_bw_q,
      )
    else:
      q_bundle = ()

    if w_p > 0.0:
      ref_mid_p = _fast_middle_ear(ref_24)
      ref_comp_p, _, _, ref_bw_p, ref_ihc_p, cf_p = _ear_frontend_single(
        ref_mid_p, jnp.zeros_like(hl), level1
      )
      ref_db_p, _ = _post_ear_vmap(
        ref_comp_p, ref_comp_p, ref_ihc_p, level1, ref_bw_p, cf_p, with_bm=False
      )
      p_bundle: tuple[jnp.ndarray, ...] = (ref_comp_p, ref_db_p, ref_bw_p)
    else:
      p_bundle = ()

    return jax.lax.stop_gradient((q_bundle, p_bundle))

  ref_bundle_b = jax.vmap(_ref_features)(reference_b, hearing_loss_b)

  @jax.checkpoint
  def _per_clip_proc(
    proc_1d: jnp.ndarray,
    hl_1d: jnp.ndarray,
    ref_bundle: tuple[tuple[jnp.ndarray, ...], tuple[jnp.ndarray, ...]],
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    q_bundle, p_bundle = ref_bundle
    if w_q <= 0.0 and w_p <= 0.0:
      zero = jnp.zeros((), dtype=jnp.float32)
      one = jnp.ones((), dtype=jnp.float32)
      return zero, one, one

    hl = jnp.asarray(hl_1d, dtype=jnp.float32)
    proc_24 = _fast_resample_24khz(proc_1d, sample_rate_hz)
    proc_mid = _fast_middle_ear(proc_24)
    proc_comp_raw, proc_bm_raw, proc_sl, _, proc_ihc, cf = _ear_frontend_single(
      proc_mid, hl, level1
    )

    if w_q > 0.0:
      ref_comp_q, ref_bm_raw_q, ref_sm_q, ref_bm_q, ref_sl_q, ref_bw_q = (
        q_bundle
      )
      proc_comp_q = jax.vmap(ear_model.envelope_align)(
        ref_comp_q, proc_comp_raw
      )
      proc_bm_al_q = jax.vmap(ear_model.envelope_align)(
        ref_bm_raw_q, proc_bm_raw
      )
      proc_db_q, proc_bm_q = _post_ear_vmap(
        proc_comp_q,
        proc_bm_al_q,
        proc_ihc,
        level1,
        ref_bw_q,
        cf,
        with_bm=True,
      )
      proc_sm_q = hasqi_lib.env_smooth(
        proc_db_q, _HASQI_SEGMENT_COVARIANCE, 24000.0
      )
      cep_corr = _safe_mel_cepstrum_correlation(
        ref_sm_q, proc_sm_q, _HASQI_SILENCE_THRESHOLD
      )
      cross_cov, ref_ms = _safe_bm_covary(
        ref_bm_q, proc_bm_q, _HASQI_SEGMENT_COVARIANCE, 24000.0
      )
      _, sync_cov = hasqi_lib.ave_covary2(
        cross_cov, ref_ms, _HASQI_SILENCE_THRESHOLD
      )
      d_loud_std, d_slope_std = _safe_spectrum_diff(ref_sl_q, proc_sl)
      d_loud = jnp.clip(1.0 - d_loud_std / _HASQI_LOUDNESS_SCALE, 0.0, 1.0)
      d_slope = jnp.clip(1.0 - d_slope_std, 0.0, 1.0)
      nonlinear = (cep_corr**2) * sync_cov[4]
      linear = _HASQI_LOUDNESS_WEIGHT * d_loud + _HASQI_SLOPE_WEIGHT * d_slope
      q = nonlinear * linear
    else:
      q = jnp.ones((), dtype=jnp.float32)

    if w_p > 0.0:
      ref_comp_p, ref_db_p, ref_bw_p = p_bundle
      proc_comp_p = jax.vmap(ear_model.envelope_align)(
        ref_comp_p, proc_comp_raw
      )
      proc_db_p, _ = _post_ear_vmap(
        proc_comp_p,
        proc_comp_p,
        proc_ihc,
        level1,
        ref_bw_p,
        cf,
        with_bm=False,
      )
      freq_sub = 8.0 * _HASPI_LOW_PASS_CUTOFF
      ref_lp, proc_lp = mod_lib.env_filter(
        ref_db_p,
        proc_db_p,
        _HASPI_LOW_PASS_CUTOFF,
        freq_sub,
        24000.0,
      )
      ref_cep, proc_cep, nkept = mod_lib.cepstral_correlation_coef(
        ref_lp,
        proc_lp,
        _HASPI_SILENCE_THRESHOLD,
        _HASPI_NERVE_DITHER,
        _HASPI_CEPSTRAL_BASIS,
      )
      ref_mod, proc_mod, _ = mod_lib.fir_modulation_filter(
        ref_cep, proc_cep, freq_sub
      )
      corrs = _safe_modulation_cross_correlation(ref_mod, proc_mod, nkept)
      p = jnp.where(
        nkept > 1,
        neural_net.intelligibility(corrs),
        jnp.float32(0.0),
      )
    else:
      p = jnp.ones((), dtype=jnp.float32)

    clip_loss = w_q * (1.0 - q) + w_p * (1.0 - p)
    return clip_loss, q, p

  clip_losses, qs, ps = jax.vmap(_per_clip_proc)(
    processed_b, hearing_loss_b, ref_bundle_b
  )
  if mask is None:
    loss = cast(jnp.ndarray, jnp.mean(clip_losses))
    mean_q = cast(jnp.ndarray, jnp.mean(qs))
    mean_p = cast(jnp.ndarray, jnp.mean(ps))
  else:
    w = jnp.asarray(mask, dtype=jnp.float32)
    denom = jnp.maximum(jnp.sum(w), 1.0)
    loss = cast(jnp.ndarray, jnp.sum(clip_losses * w) / denom)
    mean_q = cast(jnp.ndarray, jnp.sum(qs * w) / denom)
    mean_p = cast(jnp.ndarray, jnp.sum(ps * w) / denom)
  return loss, {"hasqi": mean_q, "haspi": mean_p}
