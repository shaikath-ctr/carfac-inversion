"""Unit tests for the Phase 4 training pipeline and per-clip ear synthesis."""

import pathlib
import tempfile

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from hp_acoustic.models import carfac

from carfac_inversion import nap_inverter
from carfac_inversion.scripts import train_phase4


class TrainPhase4Test(absltest.TestCase):
  """Verifies per-clip ear vmap parity and Stage 1 + Stage 2 GAN smoke runs."""

  def test_apply_ear_with_profile_matches_carfac_build(self):
    fs = 16000
    base = carfac.build(fs=fs)
    n_ch = int(base.num_channels)
    ohc = np.linspace(0.3, 0.9, n_ch, dtype=np.float32)
    shift = np.full((n_ch, 3), 0.01, dtype=np.float32)
    ref_model = carfac.build(fs=fs, ohc_health=ohc, v_half_shift=shift)

    rng = np.random.default_rng(0)
    wav = 0.03 * rng.standard_normal(512).astype(np.float32)
    expected = np.asarray(ref_model(jnp.asarray(wav[:, None]))[:, :, 0])
    got = np.asarray(
      train_phase4._apply_ear_with_profile(
        base,
        jnp.asarray(wav),
        jnp.asarray(ohc),
        jnp.asarray(shift),
      )
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

  def test_phase4_training_stage1_and_gan_smoke(self):
    with tempfile.TemporaryDirectory() as tmp:
      out_dir = pathlib.Path(tmp) / "p4_smoke"
      res = train_phase4.run_phase4_training(
        output_dir=str(out_dir),
        architecture="cochlea_hybrid",
        width=16,
        num_blocks=3,
        coarse_dim=16,
        coarse_hop=32,
        lookahead=32,
        sample_rate_hz=16000,
        seconds=0.32,
        mask_seconds=0.064,
        stft_weight=0.05,
        sai_weight=0.25,
        steps=4,
        gan_steps=2,
        batch_size=2,
        grad_accum_steps=2,
        steps_per_loop=2,
        eval_every=2,
        eval_clips_count=1,
        score_seconds=0.25,
        seed=0,
      )
      ckpt = pathlib.Path(res["checkpoint_path"])
      self.assertTrue(ckpt.is_file())
      self.assertTrue((out_dir / "checkpoint_stage1.npz").is_file())
      self.assertTrue((out_dir / "checkpoint_gan.npz").is_file())
      self.assertTrue((out_dir / "config.json").is_file())
      self.assertTrue((out_dir / "log.csv").is_file())
      self.assertTrue((out_dir / "scores" / "step0.json").is_file())
      self.assertTrue((out_dir / "scores" / "step4.json").is_file())
      self.assertTrue((out_dir / "scores" / "step6.json").is_file())
      _, _, loaded_cfg = nap_inverter.load_direct_checkpoint(ckpt)
      self.assertEqual(loaded_cfg.architecture, "cochlea_hybrid")
      self.assertGreater(res["params"], 0)

  def test_phase4_strobe_sai_input_smoke(self):
    with tempfile.TemporaryDirectory() as tmp:
      out_dir = pathlib.Path(tmp) / "p4_strobe_smoke"
      res = train_phase4.run_phase4_training(
        output_dir=str(out_dir),
        architecture="cochlea_hybrid",
        width=16,
        num_blocks=3,
        coarse_dim=16,
        coarse_hop=32,
        lookahead=32,
        sample_rate_hz=16000,
        seconds=0.32,
        mask_seconds=0.064,
        lam=0.0,
        stft_weight=0.05,
        excess_stft_weight=0.05,
        sai_weight=0.10,
        strobe_sai_input=True,
        steps=2,
        gan_steps=1,
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
      self.assertTrue((out_dir / "checkpoint_gan.npz").is_file())
      self.assertTrue((out_dir / "config.json").is_file())
      cfg_text = (out_dir / "config.json").read_text()
      self.assertIn('"strobe_sai_input": true', cfg_text)

  def test_phase4_strobe_sai_phase_free_losses_smoke(self):
    with tempfile.TemporaryDirectory() as tmp:
      out_dir = pathlib.Path(tmp) / "p4_strobe_phase_free"
      res = train_phase4.run_phase4_training(
        output_dir=str(out_dir),
        architecture="cochlea_hybrid",
        width=16,
        num_blocks=3,
        coarse_dim=16,
        coarse_hop=32,
        lookahead=32,
        sample_rate_hz=16000,
        seconds=0.32,
        mask_seconds=0.064,
        nap_weight=0.0,
        relative_nap=False,
        lam=0.0,
        lam_impaired=0.0,
        stft_weight=0.25,
        mel_weight=1.0,
        lf_complex_weight=0.5,
        rms_env_weight=0.5,
        excess_stft_weight=0.1,
        excess_all_ears=True,
        sai_weight=0.0,
        zero_resting_gate=True,
        strobe_sai_input=True,
        steps=2,
        gan_steps=1,
        batch_size=2,
        grad_accum_steps=1,
        steps_per_loop=2,
        eval_every=2,
        eval_clips_count=1,
        score_seconds=0.25,
        seed=2,
      )
      self.assertTrue((out_dir / "checkpoint_last.npz").is_file())
      self.assertTrue((out_dir / "checkpoint_gan.npz").is_file())
      self.assertGreater(res["params"], 0)
      cfg_text = (out_dir / "config.json").read_text()
      self.assertIn('"nap_weight": 0.0', cfg_text)
      self.assertIn('"excess_all_ears": true', cfg_text)
      self.assertIn('"mel_weight": 1.0', cfg_text)


if __name__ == "__main__":
  absltest.main()
