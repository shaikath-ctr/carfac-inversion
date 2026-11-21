"""Tests for the head-to-head ``benchmark_arcfac`` comparison runner."""

import csv
import pathlib
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from hp_acoustic.models import carfac

from carfac_inversion import nap_inverter
from carfac_inversion import sai_inverter
from carfac_inversion.scripts import benchmark_arcfac


class BenchmarkArcfacTest(absltest.TestCase):
  """Tests for ``carfac_inversion.scripts.benchmark_arcfac``."""

  def test_run_arcfac_benchmark_smoke_writes_csv_and_summary(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      rows = benchmark_arcfac.run_arcfac_benchmark(
        input_dir=None,
        output_dir=tmp_dir,
        max_clips=1,
        clip_seconds=0.25,
        sample_rate_hz=16000,
        targets=("healthy", "flat:0.5"),
        methods=("carfac_exact", "carfac_perceptual"),
        carfac_adam_steps=2,
        carfac_lbfgs_steps=2,
        chunk_seconds=None,
        seed=0,
      )
      self.assertLen(rows, 4)
      csv_path = pathlib.Path(tmp_dir) / "benchmark_arcfac.csv"
      summary_path = pathlib.Path(tmp_dir) / "summary.md"
      self.assertTrue(csv_path.is_file())
      self.assertTrue(summary_path.is_file())
      with csv_path.open() as f:
        parsed = list(csv.DictReader(f))
      self.assertLen(parsed, 4)
      self.assertIn("arcfac_nap_round_trip_rel", parsed[0])
      self.assertIn("arcfac_sai_direction", parsed[0])

  def test_run_arcfac_benchmark_sai_direct_smoke(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = pathlib.Path(tmp_dir)
      n_ch = int(carfac.channel_frequencies_for_rate(16000).shape[0])
      cfg = sai_inverter.SaiInverterConfig(
        sample_rate_hz=16000.0,
        model_dim=32,
        stem_blocks=1,
        decoder_blocks=1,
        stem_mlp_dim=64,
        decoder_mlp_dim=64,
        output_scale=0.05,
      )
      model = sai_inverter.SaiDirectInverter(config=cfg)
      params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 5, 1365, n_ch), dtype=jnp.float32),
        target_samples=4000,
      )
      ckpt = root / "sai.npz"
      sai_inverter.save_sai_checkpoint(ckpt, params, cfg)
      rows = benchmark_arcfac.run_arcfac_benchmark(
        input_dir=None,
        output_dir=str(root / "out"),
        max_clips=1,
        clip_seconds=0.25,
        sample_rate_hz=16000,
        targets=("healthy",),
        methods=("sai_direct",),
        sai_checkpoint=str(ckpt),
        seed=0,
      )
      self.assertLen(rows, 1)
      self.assertTrue(np.isfinite(float(rows[0]["arcfac_sai_round_trip_rel"])))

  def test_run_arcfac_benchmark_strobe_sai_direct_smoke(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = pathlib.Path(tmp_dir)
      cfg = nap_inverter.DirectInverterConfig(
        architecture="cochlea_hybrid",
        width=16,
        num_blocks=3,
        coarse_dim=16,
        coarse_hop=32,
        lookahead=32,
        fs=16000.0,
      )
      model = nap_inverter.CochleaDirectInverter(config=cfg)
      n_ch = int(carfac.channel_frequencies_for_rate(16000).shape[0])
      params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 128, n_ch), dtype=jnp.float32),
      )
      ckpt = root / "strobe_sai_ft.npz"
      nap_inverter.save_direct_checkpoint(ckpt, params, cfg)
      rows = benchmark_arcfac.run_arcfac_benchmark(
        input_dir=None,
        output_dir=str(root / "out"),
        max_clips=1,
        clip_seconds=0.25,
        sample_rate_hz=16000,
        targets=("healthy",),
        methods=("strobe_sai_direct",),
        strobe_sai_checkpoint=str(ckpt),
        seed=0,
      )
      self.assertLen(rows, 1)
      self.assertTrue(np.isfinite(float(rows[0]["arcfac_nap_round_trip_rel"])))

  def test_missing_arcfac_raises_clear_import_error(self):
    with self.assertRaisesRegex(ImportError, "arcfac"):
      benchmark_arcfac._load_arcfac_modules("/nonexistent/arcfac/path")


if __name__ == "__main__":
  absltest.main()
