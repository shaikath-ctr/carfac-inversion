"""Unit tests for Phase 5b direct generative SAI-to-audio TPU training."""

import pathlib
import tempfile

from absl.testing import absltest

from carfac_inversion import sai_inverter
from carfac_inversion.scripts import train_sai


class TrainSaiTest(absltest.TestCase):
  """Verifies Stage 1 SAI cycle + Stage 2 MRD/MPD GAN smoke training."""

  def test_sai_training_stage1_and_gan_smoke(self):
    with tempfile.TemporaryDirectory() as tmp:
      out_dir = pathlib.Path(tmp) / "sai_smoke"
      res = train_sai.run_sai_training(
        output_dir=str(out_dir),
        model_dim=32,
        stem_blocks=1,
        decoder_blocks=1,
        sample_rate_hz=16000,
        seconds=0.32,
        mask_seconds=0.064,
        stft_weight=0.10,
        steps=4,
        healthy_warmup_steps=2,
        gan_steps=2,
        batch_size=2,
        grad_accum_steps=2,
        steps_per_loop=2,
        eval_every=2,
        eval_clips_count=1,
        score_seconds=0.32,
        seed=0,
      )
      ckpt = pathlib.Path(res["checkpoint_path"])
      self.assertTrue(ckpt.is_file())
      self.assertTrue((out_dir / "checkpoint_stage1.npz").is_file())
      self.assertTrue((out_dir / "checkpoint_gan.npz").is_file())
      self.assertTrue((out_dir / "checkpoint_latest.npz").is_file())
      self.assertTrue((out_dir / "config.json").is_file())
      self.assertTrue((out_dir / "log.csv").is_file())
      self.assertTrue((out_dir / "scores" / "step0.json").is_file())
      self.assertTrue((out_dir / "scores" / "step4.json").is_file())
      self.assertTrue((out_dir / "scores" / "step6.json").is_file())
      _, _, loaded_cfg = sai_inverter.load_sai_checkpoint(ckpt)
      self.assertEqual(loaded_cfg.model_dim, 32)
      self.assertGreater(res["params"], 0)


if __name__ == "__main__":
  absltest.main()
