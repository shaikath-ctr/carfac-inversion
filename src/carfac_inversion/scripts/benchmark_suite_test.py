"""Tests for the ``benchmark_suite`` diagnostic benchmark."""

import csv
import pathlib
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from hp_acoustic.data import clips as clip_types
from hp_acoustic.data import records
from hp_acoustic.io import audio_io
from hp_acoustic.metrics import sii

from carfac_inversion import neural
from carfac_inversion.scripts import benchmark_suite


class BenchmarkSuiteTest(absltest.TestCase):
  """Tests for ``carfac_inversion.scripts.benchmark_suite``."""

  def test_load_speech_clips_from_wav_directory(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      wav_path = pathlib.Path(tmp_dir) / "sample.wav"
      t = np.arange(8000, dtype=np.float32) / 16000.0
      audio_io.save_audio_to_file(
        np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32),
        str(wav_path),
        16000,
      )
      loaded = benchmark_suite.load_speech_clips(
        tmp_dir, sample_rate_hz=16000, max_clips=2, clip_seconds=0.25
      )
      self.assertLen(loaded, 1)
      name, audio = loaded[0]
      self.assertEqual(name, "sample")
      self.assertEqual(audio.shape, (4000,))

  def test_load_speech_clips_from_processed_shards(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      test_split = pathlib.Path(tmp_dir) / "test"
      test_split.mkdir(parents=True)
      clip = clip_types.Clip(
        audio=jnp.ones(4000, dtype=jnp.float32) * 0.05,
        sample_rate=16000,
        clip_id="test_clip_0",
      )
      records.write_clips([clip], str(test_split), clips_per_shard=10)
      loaded = benchmark_suite.load_speech_clips(
        tmp_dir,
        sample_rate_hz=16000,
        max_clips=1,
        clip_seconds=0.2,
        split="test",
      )
      self.assertLen(loaded, 1)
      self.assertEqual(loaded[0][1].shape, (3200,))

  def test_run_benchmark_smoke_writes_csv_wavs_and_charts(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      rows = benchmark_suite.run_benchmark(
        input_dir=None,
        output_dir=tmp_dir,
        max_clips=1,
        clip_seconds=0.2,
        sample_rates=(16000,),
        levels_db_spl=(65.0,),
        snrs_db=(None,),
        step_budgets=((0, 0), (3, 2)),
        chunk_seconds=None,
        seed=0,
      )
      self.assertNotEmpty(rows)
      csv_path = pathlib.Path(tmp_dir) / "metrics.csv"
      self.assertTrue(csv_path.is_file())
      with csv_path.open() as f:
        parsed = list(csv.DictReader(f))
      self.assertLen(parsed, len(rows))
      self.assertTrue(
        (
          pathlib.Path(tmp_dir) / "charts" / "pareto_quality_vs_runtime.png"
        ).is_file()
      )
      self.assertTrue(
        (
          pathlib.Path(tmp_dir) / "charts" / "auralization_comparison.png"
        ).is_file()
      )

  def test_optional_enhancers_handle_installed_and_unknown(self):
    noisy = (
      np.random.default_rng(0).normal(0.0, 0.02, size=1000).astype(np.float32)
    )
    for name in ("shaca", "viking", "seanet"):
      out = benchmark_suite._run_optional_enhancer(name, noisy, 16000)
      if out is not None:
        self.assertEqual(out.shape, noisy.shape)
        self.assertTrue(np.all(np.isfinite(out)))
    with self.assertRaisesRegex(ValueError, "Unsupported enhancer"):
      benchmark_suite._run_optional_enhancer("unknown", noisy, 16000)

  def test_scale_to_spl_uses_carfac_rms_one_convention(self):
    t = np.arange(1600, dtype=np.float32) / 16000.0
    sine = np.sin(2.0 * np.pi * 500.0 * t).astype(np.float32)
    scaled = benchmark_suite._scale_to_spl(sine, 65.0)
    expected_rms = 10.0 ** ((65.0 - sii.CARFAC_RMS_ONE_DB_SPL) / 20.0)
    self.assertAlmostEqual(
      float(np.sqrt(np.mean(scaled**2))), expected_rms, delta=1e-6
    )
    zeros = np.zeros(128, dtype=np.float32)
    np.testing.assert_array_equal(
      benchmark_suite._scale_to_spl(zeros, 65.0), zeros
    )

  def test_run_benchmark_routes_perceptual_neural_checkpoint(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = pathlib.Path(tmp_dir)
      exact_ckpt = root / "exact.npz"
      perc_ckpt = root / "perc.npz"
      for path, mode in ((exact_ckpt, "exact"), (perc_ckpt, "perceptual")):
        cfg = neural.NeuralRefinerConfig(
          dim=32, dilations=(1, 2), unet_base_dim=8, target_mode=mode
        )
        model = neural.NapResidualRefiner(cfg)
        params = model.init(
          jax.random.PRNGKey(0),
          jnp.zeros((1, 1024), dtype=jnp.float32),
          jnp.zeros((1, 1024, cfg.n_coch_bins), dtype=jnp.float32),
          jnp.zeros((1, 1024, cfg.n_coch_bins), dtype=jnp.float32),
        )
        neural.save_refiner_checkpoint(path, params, cfg)

      with self.assertNoLogs(level="WARNING"):
        rows = benchmark_suite.run_benchmark(
          input_dir=None,
          output_dir=str(root / "out"),
          max_clips=1,
          clip_seconds=0.15,
          sample_rates=(16000,),
          levels_db_spl=(65.0,),
          snrs_db=(10.0,),
          step_budgets=((0, 0),),
          chunk_seconds=None,
          neural_checkpoint=str(exact_ckpt),
          perceptual_neural_checkpoint=str(perc_ckpt),
          seed=0,
        )
      scenarios = {r["scenario"] for r in rows}
      self.assertIn("auralization", scenarios)
      self.assertIn("roundtrip", scenarios)


if __name__ == "__main__":
  absltest.main()
