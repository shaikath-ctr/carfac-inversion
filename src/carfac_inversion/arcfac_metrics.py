"""Perceptual representation metrics matching ``AFHI/arcfac``.

Implements the evaluation suite from ``arcfac.score``, ``arcfac.match``, and
``arcfac.sai_loss`` (NAP & SAI round-trip L1, impairment ``kept``, ``healed``,
``direction``, ``off_axis`` across SPL level bands, and multi-resolution
log-STFT spectral distance) so ``carfac-inversion`` benchmarks directly on
``arcfac``'s native metrics without requiring optional baseline packages.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import scipy.signal
from hp_acoustic.models import sai as sai_lib

from carfac_inversion import tpu

FULL_SCALE_DB_SPL = 104.0
LEVEL_BANDS_DB = (45.0, 60.0, 75.0)
SPACES = ("nap", "sai")
DEFAULT_SAI_HYPERS = sai_lib.SaiHypers()


def level_db_spl(x: np.ndarray) -> float:
  """Returns ``x``'s presentation level in dB SPL under 104 dB full-scale."""
  rms = float(np.sqrt(np.mean(np.square(np.asarray(x, np.float64)))))
  return float(
    20.0 * np.log10(max(rms, 1e-12) * np.sqrt(2.0)) + FULL_SCALE_DB_SPL
  )


def align_with_delay(
  x: np.ndarray, y: np.ndarray, delay: int = 0
) -> tuple[np.ndarray, np.ndarray]:
  """Drops ``y``'s first ``delay`` latency samples and trims both to match."""
  y_arr = np.asarray(y)[max(0, delay) :]
  x_arr = np.asarray(x)
  n = min(len(x_arr), len(y_arr))
  return x_arr[:n], y_arr[:n]


def first_unmasked_frame(
  mask_samples: int, hypers: sai_lib.SaiHypers = DEFAULT_SAI_HYPERS
) -> int:
  """Returns the first SAI frame starting at or after ``mask_samples``."""
  hop = max(1, hypers.win_len // 2)
  return -(-max(0, mask_samples) // hop)


def sai_of_naps(
  naps: jnp.ndarray | np.ndarray,
  hypers: sai_lib.SaiHypers = DEFAULT_SAI_HYPERS,
) -> np.ndarray:
  """Converts NAPs ``(batch, samples, channels)`` to ``(batch, frames, lags, channels)``."""
  with tpu.host_cpu_context():
    sai = sai_lib.from_naps(jnp.asarray(naps, dtype=jnp.float32), hypers)
    return np.asarray(jnp.moveaxis(sai, 0, 1), dtype=np.float32)


def _frame_levels(
  x: np.ndarray, hypers: sai_lib.SaiHypers = DEFAULT_SAI_HYPERS
) -> np.ndarray:
  hop = max(1, hypers.win_len // 2)
  win = hypers.win_len
  frames = -(-max(len(x) - win, 0) // hop) + 1
  padded = np.pad(np.asarray(x, np.float64), (0, frames * hop + win))
  return np.array(
    [level_db_spl(padded[i * hop : i * hop + win]) for i in range(frames)]
  )


def _band_of_time(
  x: np.ndarray,
  space: str,
  mask_samples: int,
  n_time: int,
  hypers: sai_lib.SaiHypers = DEFAULT_SAI_HYPERS,
) -> np.ndarray:
  levels = _frame_levels(x, hypers)
  if space == "nap":
    hop = max(1, hypers.win_len // 2)
    idx = np.minimum((mask_samples + np.arange(n_time)) // hop, len(levels) - 1)
  else:
    idx = np.minimum(
      first_unmasked_frame(mask_samples, hypers) + np.arange(n_time),
      len(levels) - 1,
    )
  return np.digitize(levels[idx], LEVEL_BANDS_DB)


def direction_metrics(
  dy: np.ndarray, di: np.ndarray
) -> tuple[np.ndarray, float, float]:
  """Regresses ``dy`` onto ``di`` per channel, pooled, and off-axis L1 share."""
  c = dy.shape[-1]
  dy2 = dy.reshape(-1, c)
  di2 = di.reshape(-1, c)
  num = np.sum(dy2 * di2, axis=0)
  den = np.sum(di2 * di2, axis=0)
  with np.errstate(invalid="ignore", divide="ignore"):
    per_channel = np.where(den > 0, num / den, np.nan)
  den_sum = float(np.sum(den))
  pooled = float(np.sum(num) / den_sum) if den_sum > 0.0 else float("nan")
  resid = dy2 - (pooled if np.isfinite(pooled) else 0.0) * di2
  total = float(np.sum(np.abs(dy2)))
  scale = 1e-6 * float(np.sum(np.abs(di2)))
  off_axis = float(np.sum(np.abs(resid)) / total) if total > scale else 0.0
  return per_channel, pooled, off_axis


def log_stft_distance(
  y: np.ndarray,
  x: np.ndarray,
  *,
  resolutions: tuple[int, ...] = (512, 1024, 2048),
) -> float:
  """Computes ``arcfac.match`` multi-resolution -60 dB floored log-STFT L1."""
  x_arr = np.asarray(x, dtype=np.float32)
  y_arr = np.asarray(y, dtype=np.float32)
  common = min(len(x_arr), len(y_arr))
  if common < 16:
    return float("nan")
  x_arr = x_arr[:common]
  y_arr = y_arr[:common]
  valid_res = [n for n in resolutions if n <= common]
  if not valid_res:
    valid_res = [common]
  dists: list[float] = []
  for n in valid_res:
    _, _, sy = scipy.signal.stft(y_arr, nperseg=n, noverlap=(3 * n) // 4)
    _, _, sx = scipy.signal.stft(x_arr, nperseg=n, noverlap=(3 * n) // 4)
    floor = max(1e-3 * float(np.max(np.abs(sx))), 1e-12)
    dists.append(
      float(
        np.mean(np.abs(np.log(np.abs(sy) + floor) - np.log(np.abs(sx) + floor)))
      )
    )
  return float(np.mean(dists))


def _ensure_batched_nap(nap: np.ndarray) -> np.ndarray:
  arr = np.asarray(nap, dtype=np.float32)
  return arr[None, :, :] if arr.ndim == 2 else arr


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
  a_f = np.asarray(a, dtype=np.float64).ravel()
  b_f = np.asarray(b, dtype=np.float64).ravel()
  if np.std(a_f) <= 1e-12 or np.std(b_f) <= 1e-12:
    return 0.0
  return float(np.corrcoef(a_f, b_f)[0, 1])


def score_arcfac(
  ref: np.ndarray,
  est: np.ndarray,
  *,
  n_target: np.ndarray,
  n_healthy_ref: np.ndarray,
  n_healthy_est: np.ndarray,
  sample_rate_hz: int = 32000,
  delay: int = 0,
  mask_samples: int | None = None,
  hypers: sai_lib.SaiHypers = DEFAULT_SAI_HYPERS,
  spaces: tuple[str, ...] = SPACES,
) -> dict[str, Any]:
  """Scores waveform ``est`` against reference ``ref`` using ``arcfac.score`` metrics.

  Args:
    ref: Reference input mono waveform ``(T,)``.
    est: Reconstructed or auralized mono waveform ``(T,)``.
    n_target: Target ear NAP on ``ref``, shape ``(T, C)`` or ``(1, T, C)``.
    n_healthy_ref: Healthy ear NAP on ``ref``, shape ``(T, C)`` or ``(1, T, C)``.
    n_healthy_est: Healthy ear NAP on ``est``, shape ``(T, C)`` or ``(1, T, C)``.
    sample_rate_hz: Sample rate in Hz (used for default mask duration).
    delay: Causal latency samples dropped from ``est`` and ``n_healthy_est``.
    mask_samples: Initial cold-start samples excluded from scoring; defaults to
      ``min(round(0.25 * fs), len(ref) // 4)`` (8000 at >= 2s @ 32 kHz).
    hypers: SAI hyperparameters matching ``hp_acoustic.models.sai.SaiHypers``.
    spaces: Representation spaces to score, subset of ``('nap', 'sai')``.

  Returns:
    Nested dictionary matching ``arcfac.score.score`` schema plus
    ``log_stft_to_x`` and ``corr_with_x``.
  """
  d = max(0, int(delay))
  x_a, y_a = align_with_delay(ref, est, delay=d)
  nt_b = _ensure_batched_nap(n_target)[:, : len(x_a)]
  nhx_b = _ensure_batched_nap(n_healthy_ref)[:, : len(x_a)]
  nhy_raw = _ensure_batched_nap(n_healthy_est)[:, d:]
  common = min(len(x_a), nt_b.shape[1], nhx_b.shape[1], nhy_raw.shape[1])
  x_a = x_a[:common]
  y_a = y_a[:common]
  nt_b = nt_b[:, :common]
  nhx_b = nhx_b[:, :common]
  nhy_b = nhy_raw[:, :common]

  if mask_samples is None:
    mask = min(int(round(0.25 * sample_rate_hz)), common // 4)
  else:
    mask = min(max(0, int(mask_samples)), max(0, common - 1))

  out: dict[str, Any] = {
    "samples": common,
    "delay": d,
    "mask": mask,
    "x_db_spl": level_db_spl(x_a),
    "y_db_spl": level_db_spl(y_a),
    "log_stft_to_x": log_stft_distance(y_a, x_a),
    "corr_with_x": _safe_corrcoef(y_a, x_a),
  }

  for space in spaces:
    if space == "nap":
      r_t = np.asarray(nt_b[0, mask:])
      r_hx = np.asarray(nhx_b[0, mask:])
      r_hy = np.asarray(nhy_b[0, mask:])
    elif space == "sai":
      first = first_unmasked_frame(mask, hypers)
      if common < hypers.win_len:
        r_t = np.empty((0, 0, nt_b.shape[-1]), dtype=np.float32)
        r_hx = r_t
        r_hy = r_t
      else:
        s_t = sai_of_naps(nt_b, hypers)[0]
        s_hx = sai_of_naps(nhx_b, hypers)[0]
        s_hy = sai_of_naps(nhy_b, hypers)[0]
        first_c = min(first, len(s_t))
        r_t = s_t[first_c:]
        r_hx = s_hx[first_c:]
        r_hy = s_hy[first_c:]
    else:
      raise ValueError(f"Unsupported space {space!r}; expected 'nap' or 'sai'.")

    if len(r_t) == 0:
      out[space] = {
        "identity": float("nan"),
        "round_trip": float("nan"),
        "round_trip_rel": float("nan"),
      }
      continue

    mean_hx = float(np.mean(np.abs(r_hx)))
    identity = float(np.mean(np.abs(r_hx - r_t)))
    round_trip = float(np.mean(np.abs(r_hy - r_hx)))
    s: dict[str, Any] = {
      "identity": identity,
      "round_trip": round_trip,
      "round_trip_rel": (
        round_trip / mean_hx if mean_hx > 1e-12 else float("nan")
      ),
    }
    if identity > 1e-9 * max(mean_hx, 1e-12):
      dy = r_hy - r_hx
      di = r_t - r_hx
      per_channel, pooled, off_axis = direction_metrics(dy, di)
      bands = _band_of_time(x_a, space, mask, len(dy), hypers)
      s.update(
        kept=float(np.mean(np.abs(r_hy - r_t))) / identity,
        healed=round_trip / identity,
        direction=pooled,
        off_axis=off_axis,
        direction_per_channel=[
          None if not np.isfinite(v) else round(float(v), 4)
          for v in per_channel
        ],
        direction_per_band={},
      )
      edges = (-np.inf, *LEVEL_BANDS_DB, np.inf)
      for b in range(len(edges) - 1):
        rows = bands == b
        if not np.any(rows):
          continue
        _, pooled_b, _ = direction_metrics(dy[rows], di[rows])
        s["direction_per_band"][f"{edges[b]:g}..{edges[b + 1]:g} dB"] = {
          "direction": pooled_b,
          "share_of_time": float(rows.mean()),
        }
    out[space] = s
  return out


def flatten_arcfac_metrics(scores: dict[str, Any]) -> dict[str, float]:
  """Flattens ``score_arcfac`` outputs into a fixed schema of float columns."""
  row: dict[str, float] = {
    "arcfac_log_stft": float(scores.get("log_stft_to_x", float("nan"))),
    "arcfac_corr": float(scores.get("corr_with_x", float("nan"))),
  }
  for space in SPACES:
    sp = scores.get(space, {})
    for key in (
      "identity",
      "round_trip",
      "round_trip_rel",
      "kept",
      "healed",
      "direction",
      "off_axis",
    ):
      val = sp.get(key, float("nan"))
      row[f"arcfac_{space}_{key}"] = (
        float(val) if val is not None else float("nan")
      )
  return row
