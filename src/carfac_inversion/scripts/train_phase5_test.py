"""Smoke tests for Phase 5 clinical conditioning and prescription training."""

import pathlib
import tempfile

from absl.testing import absltest

from carfac_inversion import nap_inverter
from carfac_inversion.scripts import train_phase4
from carfac_inversion.scripts import train_phase5


class TrainPhase5Test(absltest.TestCase):
  """Verifies Phase 5 warm-starting and differentiable HASQI/HASPI training."""

  def test_phase5_warmstart_and_training_smoke(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      p4_dir = root / "p4_base"
      p4_res = train_phase4.run_phase4_training(
        output_dir=str(p4_dir),
        architecture="cochlea_hybrid",
        width=16,
        num_blocks=3,
        coarse_dim=16,
        coarse_hop=32,
        lookahead=32,
        sample_rate_hz=16000,
        seconds=0.35,
        mask_seconds=0.064,
        stft_weight=0.0,
        steps=2,
        gan_steps=0,
        batch_size=2,
        steps_per_loop=2,
        eval_every=2,
        eval_clips_count=1,
        score_seconds=0.25,
        seed=0,
      )
      p5_dir = root / "p5_run"
      res = train_phase5.run_phase5_training(
        output_dir=str(p5_dir),
        init_checkpoint=p4_res["checkpoint_path"],
        architecture="cochlea_hybrid",
        width=16,
        num_blocks=3,
        coarse_dim=16,
        coarse_hop=32,
        lookahead=32,
        max_log_gain=4.5,
        sample_rate_hz=16000,
        seconds=0.35,
        mask_seconds=0.064,
        stft_weight=0.05,
        hasqi_weight=0.05,
        haspi_weight=0.05,
        steps=2,
        batch_size=2,
        grad_accum_steps=1,
        steps_per_loop=2,
        eval_every=2,
        eval_clips_count=1,
        score_seconds=0.25,
        seed=1,
      )
      ckpt = pathlib.Path(res["checkpoint_path"])
      self.assertTrue(ckpt.is_file())
      self.assertTrue((p5_dir / "config.json").is_file())
      self.assertTrue((p5_dir / "log.csv").is_file())
      self.assertTrue((p5_dir / "scores" / "step0.json").is_file())
      self.assertTrue((p5_dir / "scores" / "step2.json").is_file())
      _, _, p4_cfg = nap_inverter.load_direct_checkpoint(
        p4_res["checkpoint_path"]
      )
      _, _, cfg = nap_inverter.load_direct_checkpoint(ckpt)
      self.assertTrue(cfg.conditioned)
      self.assertEqual(cfg.max_log_gain, 4.5)
      self.assertAlmostEqual(cfg.input_scale, p4_cfg.input_scale, places=6)
      self.assertAlmostEqual(cfg.output_scale, p4_cfg.output_scale, places=6)
      self.assertIn("prescribe_hasqi", res["best_scores"])
      self.assertIn("nalr_hasqi", res["best_scores"])


if __name__ == "__main__":
  absltest.main()
