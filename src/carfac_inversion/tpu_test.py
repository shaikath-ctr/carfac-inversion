"""Tests for TPU runtime, sharding, prefetching, and multi-step scan parity."""

import pathlib
import tempfile
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl.testing import absltest
from hp_acoustic.models import carfac
from jax.experimental.compilation_cache import compilation_cache

from carfac_inversion import neural
from carfac_inversion import tpu
from carfac_inversion.scripts import train_neural_refiner


class TpuUtilsTest(absltest.TestCase):
  """Tests for ``carfac_inversion.tpu`` and multi-step ``jax.lax.scan``."""

  def test_configure_runtime_validation_and_cache(self):
    with self.assertRaisesRegex(ValueError, "matmul_precision"):
      tpu.configure_runtime(matmul_precision="invalid_mode")

    with tempfile.TemporaryDirectory() as tmp_dir:
      cache_dir = pathlib.Path(tmp_dir) / "xla_cache"
      try:
        backend, n_devices = tpu.configure_runtime(
          compilation_cache_dir=str(cache_dir),
          matmul_precision="high",
        )
        self.assertTrue(cache_dir.is_dir())
        self.assertNotEmpty(backend)
        self.assertGreaterEqual(n_devices, 1)
      finally:
        compilation_cache.reset_cache()
        jax.config.update("jax_default_matmul_precision", "default")

    with tpu.host_cpu_context():
      arr = jnp.zeros((2, 2), dtype=jnp.float32)
      self.assertEqual(arr.shape, (2, 2))

  def test_mesh_replicate_and_shard_batch(self):
    mesh = tpu.make_data_mesh()
    rep = tpu.replicated_sharding(mesh)
    tree = {"w": np.ones((4, 3), dtype=np.float32), "b": np.zeros((3,))}
    rep_tree = tpu.replicate_tree(tree, mesh)
    self.assertEqual(cast(jax.Array, rep_tree["w"]).sharding, rep)
    # Replicating an already replicated tree keeps its sharding intact.
    rep_tree2 = tpu.replicate_tree(rep_tree, mesh)
    self.assertEqual(cast(jax.Array, rep_tree2["w"]).sharding, rep)

    n_local = len(mesh.local_devices)
    unstacked = {
      "x": np.arange(n_local * 2 * 5, dtype=np.float32).reshape(n_local * 2, 5)
    }
    sh_unstacked = tpu.shard_batch(unstacked, mesh, stacked=False)
    self.assertEqual(
      cast(jax.Array, sh_unstacked["x"]).sharding,
      tpu.batch_sharding(mesh, 2, stacked=False),
    )

    stacked = {
      "x": np.arange(3 * n_local * 2 * 5, dtype=np.float32).reshape(
        3, n_local * 2, 5
      )
    }
    sh_stacked = tpu.shard_batch(stacked, mesh, stacked=True)
    self.assertEqual(
      cast(jax.Array, sh_stacked["x"]).sharding,
      tpu.batch_sharding(mesh, 3, stacked=True),
    )

    with self.assertRaisesRegex(ValueError, "rank"):
      tpu.batch_sharding(mesh, 1, stacked=True)
    with self.assertRaisesRegex(ValueError, "rank"):
      tpu.shard_batch(np.float32(1.0), mesh, stacked=False)

  def test_threaded_device_prefetcher_stacks_drops_remainder_and_errors(self):
    mesh = tpu.make_data_mesh()
    n_local = len(mesh.local_devices)
    batches = [
      (np.full((n_local, 4), float(i), dtype=np.float32),) for i in range(5)
    ]
    with tpu.ThreadedDevicePrefetcher(
      iter(batches), mesh, steps_per_loop=2, max_prefetch=2
    ) as prefetcher:
      chunks = list(prefetcher)
    # 5 batches with steps_per_loop=2 produces 2 stacked chunks and drops 5th.
    self.assertLen(chunks, 2)
    self.assertEqual(chunks[0][0].shape, (2, n_local, 4))
    np.testing.assert_allclose(np.asarray(chunks[0][0])[:, 0, 0], [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(chunks[1][0])[:, 0, 0], [2.0, 3.0])

    # Verify early exit via close() does not hang and stops cleanly.
    infinite = ((np.zeros((n_local, 2), dtype=np.float32),) for _ in range(999))
    with tpu.ThreadedDevicePrefetcher(
      infinite, mesh, steps_per_loop=1, max_prefetch=1
    ) as early:
      first = next(early)
      self.assertEqual(first[0].shape, (n_local, 2))

    # Verify exceptions in producer thread propagate to consumer.
    def _failing_iter():
      yield (np.zeros((n_local, 2), dtype=np.float32),)
      raise RuntimeError("boom in producer")

    with tpu.ThreadedDevicePrefetcher(
      _failing_iter(), mesh, steps_per_loop=1, max_prefetch=1
    ) as failing:
      _ = next(failing)
      with self.assertRaisesRegex(RuntimeError, "boom in producer"):
        _ = next(failing)

  def test_make_multi_step_fns_parity_and_buffer_donation(self):
    mesh = tpu.make_data_mesh()
    n_local = len(mesh.local_devices)
    tx = optax.adamw(learning_rate=1e-2, weight_decay=1e-4)

    def step_impl(p, os_state, batch):
      x, y = batch

      def loss_fn(pr):
        pred = x @ pr["w"]
        loss = jnp.mean((pred - y) ** 2)
        return loss, {"loss": loss}

      (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
      updates, next_os = tx.update(grads, os_state, p)
      return optax.apply_updates(p, updates), next_os, metrics

    single_fn, multi_fn = tpu.make_multi_step_fns(
      step_impl, donate_argnums=(0, 1)
    )

    init_host = {"w": np.array([[0.5], [-0.25]], dtype=np.float32)}
    p_seq = tpu.replicate_tree(init_host, mesh)
    os_seq = tpu.replicate_tree(tx.init(init_host), mesh)
    p_scan = tpu.replicate_tree(init_host, mesh)
    os_scan = tpu.replicate_tree(tx.init(init_host), mesh)

    rng = np.random.default_rng(42)
    raw_batches = [
      (
        rng.standard_normal((n_local * 2, 2)).astype(np.float32),
        rng.standard_normal((n_local * 2, 1)).astype(np.float32),
      )
      for _ in range(3)
    ]

    seq_losses = []
    for batch_h in raw_batches:
      p_seq, os_seq, m_s = single_fn(
        p_seq, os_seq, tpu.shard_batch(batch_h, mesh, stacked=False)
      )
      seq_losses.append(float(np.asarray(m_s["loss"])[0]))

    stacked_h = jax.tree.map(lambda *xs: np.stack(xs, axis=0), *raw_batches)
    p_scan, os_scan, m_scan = multi_fn(
      p_scan, os_scan, tpu.shard_batch(stacked_h, mesh, stacked=True)
    )
    del os_seq, os_scan

    np.testing.assert_allclose(
      np.asarray(p_scan["w"]), np.asarray(p_seq["w"]), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
      np.asarray(m_scan["loss"]), np.array(seq_losses), rtol=1e-5, atol=1e-5
    )

  def test_run_training_multistep_parity_exact_and_perceptual(self):
    mesh = tpu.make_data_mesh()
    bsz = max(2, len(mesh.local_devices))
    tiny_cfg = neural.NeuralRefinerConfig(
      dim=32, dilations=(1, 2), unet_base_dim=8
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      root = pathlib.Path(tmp_dir)
      t = np.arange(2048, dtype=np.float32) / 16000.0
      sig = (0.01 * np.sin(2.0 * np.pi * 440.0 * t))[None, :].repeat(
        bsz, axis=0
      )
      rng = np.random.default_rng(11)
      ws = sig * 0.75 + 0.001 * rng.standard_normal(sig.shape).astype(
        np.float32
      )

      tr_p = root / "tr.npz"
      va_p = root / "va.npz"
      np.savez(tr_p, target_wavs=sig, ws_wavs=ws)
      np.savez(va_p, target_wavs=sig[:1], ws_wavs=ws[:1])

      # 1. Exact two-stage aligned multi-step parity (K=1 vs K=2).
      ck1 = root / "ex_k1.npz"
      ck2 = root / "ex_k2.npz"
      res_k1 = train_neural_refiner.run_training(
        train_cache=str(tr_p),
        val_cache=str(va_p),
        output_checkpoint=str(ck1),
        stage1_steps=4,
        stage2_steps=2,
        batch_size=bsz,
        stage2_batch_size=bsz,
        eval_every=2,
        seed=0,
        model_config=tiny_cfg,
        steps_per_loop=1,
      )
      res_k2 = train_neural_refiner.run_training(
        train_cache=str(tr_p),
        val_cache=str(va_p),
        output_checkpoint=str(ck2),
        stage1_steps=4,
        stage2_steps=2,
        batch_size=bsz,
        stage2_batch_size=bsz,
        eval_every=2,
        seed=0,
        model_config=tiny_cfg,
        steps_per_loop=2,
      )
      np.testing.assert_allclose(
        res_k1.best_stage1_val_loss,
        res_k2.best_stage1_val_loss,
        rtol=1e-5,
        atol=1e-4,
      )
      np.testing.assert_allclose(
        res_k1.best_stage2_val_loss,
        res_k2.best_stage2_val_loss,
        rtol=1e-5,
        atol=1e-4,
      )
      _, p1, _ = neural.load_refiner_checkpoint(ck1)
      _, p2, _ = neural.load_refiner_checkpoint(ck2)
      for leaf1, leaf2 in zip(
        jax.tree.leaves(p1), jax.tree.leaves(p2), strict=True
      ):
        np.testing.assert_allclose(leaf1, leaf2, rtol=5e-3, atol=5e-4)

      # 2. Exact unaligned eval_every=3 with stage1_steps=5, K=2 matches K=1.
      ck_u1 = root / "ex_u1.npz"
      ck_u2 = root / "ex_u2.npz"
      ru1 = train_neural_refiner.run_training(
        train_cache=str(tr_p),
        val_cache=str(va_p),
        output_checkpoint=str(ck_u1),
        stage1_steps=5,
        stage2_steps=0,
        batch_size=bsz,
        eval_every=3,
        seed=3,
        model_config=tiny_cfg,
        steps_per_loop=1,
      )
      ru2 = train_neural_refiner.run_training(
        train_cache=str(tr_p),
        val_cache=str(va_p),
        output_checkpoint=str(ck_u2),
        stage1_steps=5,
        stage2_steps=0,
        batch_size=bsz,
        eval_every=3,
        seed=3,
        model_config=tiny_cfg,
        steps_per_loop=2,
      )
      self.assertEqual(
        [r["step"] for r in ru1.history], [r["step"] for r in ru2.history]
      )
      np.testing.assert_allclose(
        ru1.best_stage1_val_loss,
        ru2.best_stage1_val_loss,
        rtol=1e-5,
        atol=1e-4,
      )

      # 3. Perceptual (Phase 3b) multi-step parity (K=1 vs K=2).
      impaired = carfac.build(fs=16000, ohc_health=0.5)
      tgt_naps = np.asarray(
        jax.vmap(lambda w: impaired(w[:, None])[:, :, 0])(jnp.asarray(sig)),
        dtype=np.float32,
      )
      tr_perc = root / "tr_p.npz"
      va_perc = root / "va_p.npz"
      np.savez(tr_perc, target_naps=tgt_naps, ws_wavs=ws)
      np.savez(va_perc, target_naps=tgt_naps[:1], ws_wavs=ws[:1])

      pk1 = root / "perc_k1.npz"
      pk2 = root / "perc_k2.npz"
      rp1 = train_neural_refiner.run_training(
        train_cache=str(tr_perc),
        val_cache=str(va_perc),
        output_checkpoint=str(pk1),
        objective="perceptual",
        perceptual_steps=4,
        perceptual_batch_size=bsz,
        eval_every=2,
        seed=5,
        model_config=tiny_cfg,
        steps_per_loop=1,
      )
      rp2 = train_neural_refiner.run_training(
        train_cache=str(tr_perc),
        val_cache=str(va_perc),
        output_checkpoint=str(pk2),
        objective="perceptual",
        perceptual_steps=4,
        perceptual_batch_size=bsz,
        eval_every=2,
        seed=5,
        model_config=tiny_cfg,
        steps_per_loop=2,
      )
      np.testing.assert_allclose(
        rp1.best_stage2_val_loss,
        rp2.best_stage2_val_loss,
        rtol=1e-5,
        atol=1e-4,
      )
      np.testing.assert_allclose(
        rp1.best_val_nap_corr,
        rp2.best_val_nap_corr,
        rtol=1e-5,
        atol=1e-4,
      )
      _, pp1, _ = neural.load_refiner_checkpoint(pk1)
      _, pp2, _ = neural.load_refiner_checkpoint(pk2)
      for leaf1, leaf2 in zip(
        jax.tree.leaves(pp1), jax.tree.leaves(pp2), strict=True
      ):
        np.testing.assert_allclose(leaf1, leaf2, rtol=5e-3, atol=5e-4)


if __name__ == "__main__":
  absltest.main()
