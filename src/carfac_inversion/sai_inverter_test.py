"""Unit tests for Phase 5b float32 SAI, COLA ISTFT, and SaiDirectInverter."""

import math
import pathlib
import tempfile
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from carfac_inversion import arcfac_metrics
from carfac_inversion import sai_inverter


class SaiInverterTest(absltest.TestCase):
  """Verifies SAI parity, exact COLA ISTFT round-trip, and model checkpoints."""

  def test_sai_from_naps_f32_matches_arcfac_metrics(self):
    rng = np.random.default_rng(7)
    naps = rng.normal(0.0, 0.25, size=(2, 16000, 12)).astype(np.float32)
    expected = arcfac_metrics.sai_of_naps(naps)
    got = np.asarray(sai_inverter.sai_from_naps_f32(jnp.asarray(naps)))
    self.assertEqual(got.shape, expected.shape)
    self.assertEqual(got.dtype, np.float32)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)

  def test_cola_istft_roundtrip_arbitrary_length(self):
    rng = np.random.default_rng(11)
    t_len = 5000
    x = rng.normal(0.0, 0.1, size=(2, t_len)).astype(np.float32)
    t_pad = int(math.ceil(t_len / 128) * 128)
    x_pad = jnp.pad(jnp.asarray(x), ((0, 0), (0, t_pad - t_len)))
    for stft_fn, istft_fn in (
      (sai_inverter._stft_512_128_complex, sai_inverter.cola_istft_512_128),
      (sai_inverter._stft_1024_128_complex, sai_inverter.cola_istft_1024_128),
    ):
      spec = stft_fn(x_pad)
      recon = np.asarray(istft_fn(spec, t_len))
      self.assertEqual(recon.shape, (2, t_len))
      max_err = float(np.max(np.abs(recon - x)))
      self.assertLess(max_err, 1e-5)

  def test_sai_direct_inverter_grads_and_checkpoint_roundtrip(self):
    cfg = sai_inverter.SaiInverterConfig(
      sample_rate_hz=16000.0,
      model_dim=32,
      stem_blocks=2,
      decoder_blocks=2,
      stem_mlp_dim=64,
      decoder_mlp_dim=64,
      output_scale=0.05,
    )
    model = sai_inverter.SaiDirectInverter(config=cfg)
    rng = np.random.default_rng(3)
    sai_1s = jnp.asarray(
      np.abs(rng.normal(0.0, 0.05, size=(2, 22, 1365, 16))).astype(np.float32)
    )
    params = model.init(
      jax.random.PRNGKey(0),
      sai_1s,
      target_samples=16000,
      noise_key=jax.random.PRNGKey(1),
    )

    def _loss(p):
      y = cast(
        jnp.ndarray,
        model.apply(
          p, sai_1s, target_samples=16000, noise_key=jax.random.PRNGKey(2)
        ),
      )
      return jnp.mean(y**2)

    loss_val, grads = jax.value_and_grad(_loss)(params)
    self.assertTrue(bool(jnp.isfinite(loss_val)))
    for g in jax.tree_util.tree_leaves(grads):
      self.assertTrue(bool(jnp.all(jnp.isfinite(g))))

    with tempfile.TemporaryDirectory() as tmp:
      ckpt = pathlib.Path(tmp) / "sai_ckpt.npz"
      sai_inverter.save_sai_checkpoint(ckpt, params, cfg)
      loaded_model, loaded_p, loaded_cfg = sai_inverter.load_sai_checkpoint(
        ckpt
      )
      self.assertEqual(loaded_cfg.model_dim, 32)
      y_2s = sai_inverter.invert_sai_direct(
        np.abs(rng.normal(0.0, 0.05, size=(45, 1365, 16))).astype(np.float32),
        ckpt,
        target_samples=32000,
      )
      self.assertEqual(y_2s.shape, (32000,))
      self.assertTrue(bool(np.all(np.isfinite(y_2s))))
      y_ref = np.asarray(
        jax.jit(
          lambda s: loaded_model.apply(loaded_p, s, target_samples=16000)
        )(jnp.asarray(sai_1s[:1]))[0]
      )
      y_dir = sai_inverter.invert_sai_direct(
        np.asarray(sai_1s[0]), ckpt, target_samples=16000
      )
      np.testing.assert_allclose(y_dir, y_ref, rtol=1e-5, atol=1e-5)

  def test_strobe_sai_and_causal_ola_frame_causality(self):
    rng = np.random.default_rng(23)
    naps = np.maximum(
      0.0, rng.normal(0.05, 0.05, size=(1, 6400, 8)).astype(np.float32)
    )
    sai = sai_inverter.strobe_sai_from_naps_f32(naps, sample_rate_hz=16000.0)
    self.assertEqual(sai.ndim, 4)
    self.assertEqual(sai.shape[0], 1)
    self.assertEqual(sai.shape[3], 8)

    est_a = np.asarray(
      sai_inverter.causal_strobe_sai_to_nap(
        sai, sample_rate_hz=16000.0, target_samples=6400
      )
    )
    self.assertEqual(est_a.shape, (1, 6400, 8))
    self.assertTrue(np.all(np.isfinite(est_a)))

    # Perturbing frames >= 12 must leave earlier causal samples untouched.
    hop = sai_inverter.default_strobe_sai_params(16000.0, 8)[
      "input_segment_width"
    ]
    max_shift = max(8, hop // 2)
    sai_b = sai.at[:, 12:, :, :].set(sai[:, 12:, :, :] + 0.5)
    est_b = np.asarray(
      sai_inverter.causal_strobe_sai_to_nap(
        sai_b, sample_rate_hz=16000.0, target_samples=6400
      )
    )
    cutoff = 12 * hop - max_shift
    np.testing.assert_allclose(
      est_a[:, :cutoff, :], est_b[:, :cutoff, :], atol=0.0
    )

    # Explicit fixed decay preserves backward compatibility and shape.
    est_fixed = np.asarray(
      sai_inverter.causal_strobe_sai_to_nap(
        sai, sample_rate_hz=16000.0, target_samples=6400, persistence_decay=0.45
      )
    )
    self.assertEqual(est_fixed.shape, est_a.shape)
    self.assertTrue(np.all(np.isfinite(est_fixed)))

  def test_build_strobe_sai_extractor_and_nap_fn_jit_parity(self):
    rng = np.random.default_rng(31)
    naps = np.maximum(
      0.0, rng.normal(0.05, 0.05, size=(2, 3200, 8)).astype(np.float32)
    )
    ref_sai = np.asarray(
      sai_inverter.strobe_sai_from_naps_f32(naps, sample_rate_hz=16000.0)
    )
    ext_fn = jax.jit(sai_inverter.build_strobe_sai_extractor(16000.0, 8))
    got_sai = np.asarray(ext_fn(jnp.asarray(naps, dtype=jnp.float32)))
    np.testing.assert_allclose(got_sai, ref_sai, rtol=1e-5, atol=1e-5)

    nap_fn = jax.jit(sai_inverter.build_causal_strobe_sai_nap_fn(16000.0, 8))
    est_nap = np.asarray(nap_fn(jnp.asarray(naps, dtype=jnp.float32)))
    self.assertEqual(est_nap.shape, naps.shape)
    self.assertTrue(np.all(np.isfinite(est_nap)))


if __name__ == "__main__":
  absltest.main()
