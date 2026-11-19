"""Tests for the head-to-head ``benchmark_arcfac`` comparison runner."""

import csv
import pathlib
import tempfile

from absl.testing import absltest

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

  def test_missing_arcfac_raises_clear_import_error(self):
    with self.assertRaisesRegex(ImportError, "arcfac"):
      benchmark_arcfac._load_arcfac_modules("/nonexistent/arcfac/path")


if __name__ == "__main__":
  absltest.main()
