"""Tests for ``carfac_inversion.arcfac_metrics`` metric parity."""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from hp_acoustic.models import carfac

from carfac_inversion import arcfac_metrics


class ArcfacMetricsTest(absltest.TestCase):
  """Verifies exact behavior of the ported ``arcfac.score`` metrics."""

  def test_direction_and_off_axis_algebra(self):
    rng = np.random.default_rng(0)
    di = rng.normal(size=(200, 6))
    per_ch, pooled, off_axis = arcfac_metrics.direction_metrics(0.7 * di, di)
    np.testing.assert_allclose(per_ch, 0.7)
    self.assertAlmostEqual(pooled, 0.7, places=6)
    self.assertAlmostEqual(off_axis, 0.0, places=9)

    di_orth = rng.normal(size=(5000, 6))
    dy_orth = rng.normal(size=(5000, 6))
    _, pooled_o, off_axis_o = arcfac_metrics.direction_metrics(dy_orth, di_orth)
    self.assertLess(abs(pooled_o), 0.05)
    self.assertGreater(off_axis_o, 0.95)

  def test_align_and_log_stft_distance(self):
    x = np.sin(2.0 * np.pi * 200.0 * np.arange(8000) / 16000.0).astype(
      np.float32
    )
    y_del = np.concatenate([np.zeros(64, dtype=np.float32), x])
    xa, ya = arcfac_metrics.align_with_delay(x, y_del, delay=64)
    np.testing.assert_array_equal(xa, ya)

    self.assertLess(arcfac_metrics.log_stft_distance(-x, x), 1e-6)
    noise = (
      np.random.default_rng(1).normal(0.0, 0.3, size=len(x)).astype(np.float32)
    )
    self.assertGreater(arcfac_metrics.log_stft_distance(noise, x), 0.5)

  def test_score_arcfac_healthy_and_impaired_invariants(self):
    fs = 16000
    t = np.arange(int(1.0 * fs), dtype=np.float32) / fs
    x = (0.05 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)

    healthy = carfac.build(fs=fs)
    impaired = carfac.build(fs=fs, ohc_health=0.5)
    n_hx = np.asarray(healthy(jnp.asarray(x[:, None]))[:, :, 0])
    n_tx = np.asarray(impaired(jnp.asarray(x[:, None]))[:, :, 0])

    # 1. Healthy target only emits round_trip / round_trip_rel (kept omitted).
    half_hy = np.asarray(healthy(jnp.asarray((0.5 * x)[:, None]))[:, :, 0])
    sc_h = arcfac_metrics.score_arcfac(
      x,
      0.5 * x,
      n_target=n_hx,
      n_healthy_ref=n_hx,
      n_healthy_est=half_hy,
      sample_rate_hz=fs,
      mask_samples=4000,
    )
    self.assertNotIn("kept", sc_h["nap"])
    self.assertGreater(sc_h["nap"]["round_trip_rel"], 0.05)
    flat_h = arcfac_metrics.flatten_arcfac_metrics(sc_h)
    self.assertTrue(np.isnan(flat_h["arcfac_nap_kept"]))
    self.assertTrue(np.isfinite(flat_h["arcfac_sai_round_trip_rel"]))

    # 2. Identity on impaired target gives kept=1.0, healed=0.0, direction=0.0.
    sc_id = arcfac_metrics.score_arcfac(
      x,
      x,
      n_target=n_tx,
      n_healthy_ref=n_hx,
      n_healthy_est=n_hx,
      sample_rate_hz=fs,
      mask_samples=4000,
    )
    for sp in ("nap", "sai"):
      self.assertAlmostEqual(sc_id[sp]["kept"], 1.0, places=6)
      self.assertAlmostEqual(sc_id[sp]["healed"], 0.0, places=6)
      self.assertAlmostEqual(sc_id[sp]["direction"], 0.0, places=6)
    flat_id = arcfac_metrics.flatten_arcfac_metrics(sc_id)
    self.assertEqual(set(flat_h.keys()), set(flat_id.keys()))


if __name__ == "__main__":
  absltest.main()
