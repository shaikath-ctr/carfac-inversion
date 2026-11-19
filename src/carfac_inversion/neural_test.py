"""Tests for the Phase 3 neural residual refiner and training pipeline."""

import pathlib
import tempfile
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from carfac.jax import carfac as carfac_jax
from hp_acoustic.io import audio_io
from hp_acoustic.models import carfac

from carfac_inversion import inversion
from carfac_inversion import neural
from carfac_inversion.scripts import invert_nap as invert_nap_cli
from carfac_inversion.scripts import train_neural_refiner


class NeuralRefinerTest(absltest.TestCase):
  """Tests for ``carfac_inversion.neural`` and Phase 3 CLI integration."""

  def test_channel_projection_matrix_convex_and_ordered(self):
    for fs in (16000, 32000):
      m = carfac.build(fs=fs)
      cfs = np.asarray(m.channel_frequencies, dtype=np.float64)
      w = neural.channel_projection_matrix(cfs, n_bins=64)
      self.assertEqual(w.shape, (len(cfs), 64))
      np.testing.assert_allclose(w.sum(axis=0), np.ones(64), atol=1e-5)
      self.assertTrue(np.all(w >= 0.0))

    with self.assertRaisesRegex(ValueError, "non-empty"):
      neural.channel_projection_matrix(np.zeros((0,), dtype=np.float64))

  def test_config_json_roundtrip(self):
    cfg = neural.NeuralRefinerConfig(
      dim=96, dilations=(1, 2, 4), unet_base_dim=16, target_mode="perceptual"
    )
    restored = neural.NeuralRefinerConfig.from_json(cfg.to_json())
    self.assertEqual(cfg, restored)

    # Older exact checkpoints omit target_mode and should default to "exact".
    legacy_json = cfg.to_json().replace(', "target_mode": "perceptual"', "")
    legacy = neural.NeuralRefinerConfig.from_json(legacy_json)
    self.assertEqual(legacy.target_mode, "exact")

  def test_expected_mode_mismatch_logs_warning(self):
    cfg = neural.NeuralRefinerConfig(
      dim=32, dilations=(1, 2), unet_base_dim=8, target_mode="exact"
    )
    model = neural.NapResidualRefiner(cfg)
    params = model.init(
      jax.random.PRNGKey(2),
      jnp.zeros((1, 1024), dtype=jnp.float32),
      jnp.zeros((1, 1024, cfg.n_coch_bins), dtype=jnp.float32),
      jnp.zeros((1, 1024, cfg.n_coch_bins), dtype=jnp.float32),
    )
    mono_m = carfac.build(fs=16000)
    t = np.arange(2048, dtype=np.float32) / 16000.0
    tone = (0.01 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    nap = np.asarray(mono_m(jnp.asarray(tone[:, None]))[:, :, 0])

    with tempfile.TemporaryDirectory() as tmp_dir:
      ckpt = pathlib.Path(tmp_dir) / "exact_only.npz"
      neural.save_refiner_checkpoint(ckpt, params, cfg)
      with self.assertLogs(level="WARNING") as logs:
        out = neural.refine_waveform(
          nap,
          tone,
          mono_m,
          ckpt,
          chunk_seconds=None,
          expected_mode="perceptual",
        )
      self.assertEqual(out.shape, tone.shape)
      self.assertTrue(
        any("target_mode='exact'" in msg for msg in logs.output),
        msg=f"Unexpected warning logs: {logs.output}",
      )

  def test_zero_init_identity_silence_and_arbitrary_length(self):
    cfg = neural.NeuralRefinerConfig(dim=64, dilations=(1, 2), unet_base_dim=12)
    model = neural.NapResidualRefiner(cfg)
    rng = jax.random.PRNGKey(0)
    ws = jax.random.normal(rng, (2, 4096), dtype=jnp.float32) * 0.005
    tgt_n = jnp.ones((2, 4096, cfg.n_coch_bins), dtype=jnp.float32) * 0.02
    ws_n = tgt_n * 0.8
    params = model.init(rng, ws, tgt_n, ws_n)

    out = cast(jnp.ndarray, model.apply(params, ws, tgt_n, ws_n))
    self.assertEqual(out.shape, ws.shape)
    self.assertLess(float(jnp.max(jnp.abs(out - ws))), 1e-6)

    zeros_out = cast(
      jnp.ndarray,
      model.apply(
        params,
        jnp.zeros((1, 2048), dtype=jnp.float32),
        jnp.zeros((1, 2048, cfg.n_coch_bins), dtype=jnp.float32),
        jnp.zeros((1, 2048, cfg.n_coch_bins), dtype=jnp.float32),
      ),
    )
    self.assertEqual(float(jnp.max(jnp.abs(zeros_out))), 0.0)

    # Non-hop-multiple odd length should match shape cleanly.
    odd_out = cast(
      jnp.ndarray,
      model.apply(params, ws[:1, :3993], tgt_n[:1, :3993], ws_n[:1, :3993]),
    )
    self.assertEqual(odd_out.shape, (1, 3993))

    def _loss(p):
      pred = cast(jnp.ndarray, model.apply(p, ws, tgt_n, ws_n))
      return jnp.mean((pred - ws) ** 2)

    grads = jax.grad(_loss)(params)
    self.assertTrue(
      all(
        bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads)
      )
    )

  def test_checkpoint_roundtrip_ola_binaural_and_invert_nap(self):
    cfg = neural.NeuralRefinerConfig(dim=48, dilations=(1, 2), unet_base_dim=8)
    model = neural.NapResidualRefiner(cfg)
    rng = jax.random.PRNGKey(1)
    dummy_x = jnp.zeros((1, 1024), dtype=jnp.float32)
    dummy_n = jnp.zeros((1, 1024, cfg.n_coch_bins), dtype=jnp.float32)
    params = model.init(rng, dummy_x, dummy_n, dummy_n)

    with tempfile.TemporaryDirectory() as tmp_dir:
      ckpt = pathlib.Path(tmp_dir) / "test_refiner.npz"
      neural.save_refiner_checkpoint(ckpt, params, cfg)
      _, loaded_params, loaded_cfg = neural.load_refiner_checkpoint(ckpt)
      self.assertEqual(loaded_cfg, cfg)
      for a, b in zip(
        jax.tree_util.tree_leaves(params),
        jax.tree_util.tree_leaves(loaded_params),
        strict=True,
      ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))

      # Mono OLA + invert_nap + CLI integration
      mono_m = carfac.build(fs=16000)
      t = np.arange(8000, dtype=np.float32) / 16000.0
      tone = (0.01 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
      nap = np.asarray(mono_m(jnp.asarray(tone[:, None]))[:, :, 0])
      res = inversion.invert_nap(
        nap,
        mono_m,
        inversion.InversionConfig(
          adam_steps=0,
          lbfgs_steps=0,
          chunk_seconds=0.3,
          chunk_overlap_s=0.05,
          neural_checkpoint=str(ckpt),
        ),
      )
      self.assertEqual(res.waveform.shape, (8000,))
      self.assertTrue(np.all(np.isfinite(res.waveform)))

      # Binaural 2-ear support
      carfac_params = carfac_jax.CarfacDesignParameters.with_n_ears(2, fs=16000)
      carfac_params.ears[0].ihc.ihc_style = "two_cap_with_syn"
      carfac_params.ears[1].ihc.ihc_style = "two_cap_with_syn"
      hypers, weights, state = carfac_jax.design_and_init_carfac(carfac_params)
      bin_m = carfac.CarfacCochleaModel(
        hypers=hypers, weights=weights, initial_state=state, fs=16000.0
      )
      bin_w = np.stack([tone, tone * 0.5], axis=1)
      bin_nap = np.asarray(bin_m(jnp.asarray(bin_w)))
      bin_out = neural.refine_waveform(
        bin_nap, bin_w, bin_m, ckpt, chunk_seconds=0.3, chunk_overlap_s=0.05
      )
      self.assertEqual(bin_out.shape, (8000, 2))
      np.testing.assert_allclose(bin_out, bin_w, atol=1e-5)

      # CLI with --neural_checkpoint
      in_wav = pathlib.Path(tmp_dir) / "in.wav"
      out_wav = pathlib.Path(tmp_dir) / "out.wav"
      audio_io.save_audio_to_file(tone, str(in_wav), 16000)
      cli_res = invert_nap_cli.run_invert_wav(
        input_wav=str(in_wav),
        output_wav=str(out_wav),
        sample_rate_hz=16000,
        adam_steps=0,
        lbfgs_steps=0,
        neural_checkpoint=str(ckpt),
      )
      self.assertTrue(out_wav.is_file())
      self.assertEqual(cli_res.waveform.shape, tone.shape)

    with self.assertRaises(FileNotFoundError):
      neural.load_refiner_checkpoint(pathlib.Path(tmp_dir) / "missing.npz")

  def test_mini_training_stage1_and_stage2_smoke(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = pathlib.Path(tmp_dir)
      tr_p = root / "train.npz"
      va_p = root / "val.npz"
      ckpt_p = root / "trained.npz"
      rng = np.random.default_rng(0)
      t = np.arange(4096, dtype=np.float32) / 16000.0
      sig = (0.01 * np.sin(2.0 * np.pi * 440.0 * t))[None, :].repeat(2, axis=0)
      ws = sig * 0.8 + 0.001 * rng.standard_normal(sig.shape).astype(np.float32)
      np.savez(tr_p, target_wavs=sig, ws_wavs=ws)
      np.savez(va_p, target_wavs=sig[:1], ws_wavs=ws[:1])

      tiny_cfg = neural.NeuralRefinerConfig(
        dim=32, dilations=(1, 2), unet_base_dim=8
      )
      result = train_neural_refiner.run_training(
        train_cache=str(tr_p),
        val_cache=str(va_p),
        output_checkpoint=str(ckpt_p),
        stage1_steps=2,
        stage2_steps=1,
        batch_size=2,
        stage2_batch_size=1,
        eval_every=1,
        model_config=tiny_cfg,
      )
      self.assertTrue(ckpt_p.is_file())
      self.assertIsNotNone(result.stage1_checkpoint_path)
      assert result.stage1_checkpoint_path is not None
      self.assertTrue(pathlib.Path(result.stage1_checkpoint_path).is_file())
      self.assertTrue(np.isfinite(result.best_stage1_val_loss))
      self.assertTrue(np.isfinite(result.best_stage2_val_loss))

  def test_mini_training_perceptual_smoke(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      root = pathlib.Path(tmp_dir)
      tr_p = root / "train_p.npz"
      va_p = root / "val_p.npz"
      ckpt_p = root / "perc.npz"
      rng = np.random.default_rng(1)
      t = np.arange(4096, dtype=np.float32) / 16000.0
      sig = (0.01 * np.sin(2.0 * np.pi * 440.0 * t))[None, :].repeat(2, axis=0)
      impaired = carfac.build(fs=16000, ohc_health=0.5)
      healthy = carfac.build(fs=16000)
      tgt_naps = np.asarray(
        jax.vmap(lambda w: impaired(w[:, None])[:, :, 0])(jnp.asarray(sig)),
        dtype=np.float32,
      )
      ws = sig * 0.6 + 0.0005 * rng.standard_normal(sig.shape).astype(
        np.float32
      )
      np.savez(tr_p, target_naps=tgt_naps, ws_wavs=ws)
      np.savez(va_p, target_naps=tgt_naps[:1], ws_wavs=ws[:1])

      tiny_cfg = neural.NeuralRefinerConfig(
        dim=32, dilations=(1, 2), unet_base_dim=8
      )
      # Oracle-NAP Stage 1 warmup followed by Stage 2 perceptual fine-tuning.
      oracle_p = root / "oracle.npz"
      oracle_ckpt_p = root / "oracle_s1.npz"
      np.savez(oracle_p, target_wavs=sig, target_naps=tgt_naps, ws_wavs=ws)
      s1_res = train_neural_refiner.run_training(
        train_cache=str(oracle_p),
        val_cache=str(oracle_p),
        output_checkpoint=str(oracle_ckpt_p),
        objective="exact",
        stage1_steps=1,
        stage2_steps=0,
        batch_size=2,
        eval_every=1,
        model_config=tiny_cfg,
      )
      self.assertIsNotNone(s1_res.stage1_checkpoint_path)
      assert s1_res.stage1_checkpoint_path is not None
      self.assertTrue(pathlib.Path(s1_res.stage1_checkpoint_path).is_file())

      result = train_neural_refiner.run_training(
        train_cache=str(tr_p),
        val_cache=str(va_p),
        output_checkpoint=str(ckpt_p),
        objective="perceptual",
        init_checkpoint=str(s1_res.stage1_checkpoint_path),
        perceptual_steps=2,
        perceptual_batch_size=2,
        eval_every=1,
        model_config=tiny_cfg,
      )
      self.assertTrue(ckpt_p.is_file())
      self.assertTrue(np.isfinite(result.best_stage2_val_loss))
      self.assertTrue(np.isfinite(result.best_val_nap_corr))
      _, _, loaded_cfg = neural.load_refiner_checkpoint(ckpt_p)
      self.assertEqual(loaded_cfg.target_mode, "perceptual")

      res = inversion.invert_nap(
        tgt_naps[0],
        healthy,
        inversion.InversionConfig(
          mode="perceptual",
          adam_steps=0,
          lbfgs_steps=0,
          chunk_seconds=None,
          neural_checkpoint=str(ckpt_p),
        ),
      )
      self.assertEqual(res.waveform.shape, (4096,))
      self.assertTrue(np.all(np.isfinite(res.waveform)))

      with self.assertRaisesRegex(ValueError, "Unsupported objective"):
        train_neural_refiner.run_training(
          train_cache=str(tr_p),
          val_cache=str(va_p),
          output_checkpoint=str(ckpt_p),
          objective="invalid",
        )

  def test_refine_waveform_chunked_ola_with_preroll(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      ckpt_p = pathlib.Path(tmp_dir) / "ola.npz"
      cfg = neural.NeuralRefinerConfig(
        dim=32, dilations=(1, 2), unet_base_dim=8
      )
      model = neural.NapResidualRefiner(cfg)
      params = model.init(
        jax.random.PRNGKey(9),
        jnp.zeros((1, 1024), dtype=jnp.float32),
        jnp.zeros((1, 1024, cfg.n_coch_bins), dtype=jnp.float32),
        jnp.zeros((1, 1024, cfg.n_coch_bins), dtype=jnp.float32),
      )
      neural.save_refiner_checkpoint(ckpt_p, params, cfg)

      healthy = carfac.build(fs=16000)
      t = np.arange(4800, dtype=np.float32) / 16000.0
      ws = (0.015 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
      tgt_nap = np.asarray(healthy(jnp.asarray(ws[:, None]))[:, :, 0])

      out_preroll = neural.refine_waveform(
        tgt_nap,
        ws,
        healthy,
        ckpt_p,
        chunk_seconds=0.15,
        chunk_overlap_s=0.04,
        chunk_preroll_s=0.128,
      )
      out_nopreroll = neural.refine_waveform(
        tgt_nap,
        ws,
        healthy,
        ckpt_p,
        chunk_seconds=0.15,
        chunk_overlap_s=0.04,
        chunk_preroll_s=0.0,
      )
      self.assertEqual(out_preroll.shape, ws.shape)
      self.assertEqual(out_nopreroll.shape, ws.shape)
      self.assertTrue(np.all(np.isfinite(out_preroll)))
      self.assertTrue(np.all(np.isfinite(out_nopreroll)))


if __name__ == "__main__":
  absltest.main()
