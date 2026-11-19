"""Unit tests for Phase 4 direct causal NAP-to-audio inverter architectures."""

import pathlib
import tempfile
from typing import cast

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

from carfac_inversion import nap_inverter


class NapInverterTest(parameterized.TestCase):
  """Verifies causality, checkpointing, and forward/backward shapes."""

  @parameterized.parameters("tcn", "cochlea_hybrid")
  def test_strict_causality_and_arbitrary_length(self, architecture: str):
    cfg = nap_inverter.DirectInverterConfig(
      architecture=architecture,
      width=16,
      num_blocks=4,
      coarse_dim=16,
      coarse_hop=32,
      coarse_dilations=(1, 2),
      lookahead=32,
      output_highpass_hz=20.0,
      fs=16000.0,
    )
    model = nap_inverter.CochleaDirectInverter(config=cfg)
    rng = np.random.default_rng(7)
    x_a = jnp.asarray(
      0.05 * rng.standard_normal((2, 500, 12)), dtype=jnp.float32
    )
    x_b = x_a.at[:, 250:, :].set(x_a[:, 250:, :] + 0.5)

    params = model.init(jax.random.PRNGKey(0), x_a[:1])
    y_a = np.asarray(model.apply(params, x_a))
    y_b = np.asarray(model.apply(params, x_b))

    self.assertEqual(y_a.shape, (2, 500))
    self.assertTrue(np.all(np.isfinite(y_a)))
    np.testing.assert_allclose(y_a[:, :250], y_b[:, :250], atol=1e-6, rtol=1e-6)
    self.assertGreater(float(np.max(np.abs(y_a[:, 250:] - y_b[:, 250:]))), 1e-4)

  def test_checkpoint_roundtrip_and_aligned_inference(self):
    cfg = nap_inverter.DirectInverterConfig(
      architecture="cochlea_hybrid",
      width=16,
      num_blocks=3,
      coarse_dim=16,
      coarse_hop=16,
      coarse_dilations=(1, 2),
      lookahead=16,
      fs=32000.0,
    )
    model = nap_inverter.CochleaDirectInverter(config=cfg)
    nap = np.random.default_rng(1).standard_normal((320, 8)).astype(np.float32)
    params = model.init(jax.random.PRNGKey(3), jnp.asarray(nap[None]))

    with tempfile.TemporaryDirectory() as tmp:
      ckpt = pathlib.Path(tmp) / "phase4_test.npz"
      nap_inverter.save_direct_checkpoint(ckpt, params, cfg)
      _, _, loaded_cfg = nap_inverter.load_direct_checkpoint(ckpt)
      self.assertEqual(loaded_cfg, cfg)

      out_2d = nap_inverter.invert_nap_direct(nap, ckpt, align_lookahead=True)
      self.assertEqual(out_2d.shape, (320,))
      self.assertTrue(np.all(np.isfinite(out_2d)))

      out_pre = nap_inverter.invert_nap_direct(
        nap, ckpt, align_lookahead=True, preroll_samples=32
      )
      self.assertEqual(out_pre.shape, (320,))
      self.assertTrue(np.all(np.isfinite(out_pre)))

      nap_stereo = np.stack([nap, 0.5 * nap], axis=-1)
      out_3d = nap_inverter.invert_nap_direct(
        nap_stereo, ckpt, align_lookahead=False
      )
      self.assertEqual(out_3d.shape, (320, 2))

  def test_tonotopic_stem_matches_2d_conv(self):
    rng = np.random.default_rng(42)
    x = jnp.asarray(0.1 * rng.standard_normal((2, 24, 15, 2)), jnp.float32)
    stem = nap_inverter._CausalTonotopicStem(width=8, hidden=4)
    params = stem.init(jax.random.PRNGKey(5), x)
    got = np.asarray(stem.apply(params, x))

    p = params["params"]
    x_pad = jnp.pad(x, ((0, 0), (2, 0), (2, 2), (0, 0)))
    h = cast(
      jnp.ndarray,
      nn.Conv(4, (3, 5), padding="VALID").apply(
        {"params": {"kernel": p["conv1_kernel"], "bias": p["conv1_bias"]}},
        x_pad,
      ),
    )
    h = nn.gelu(h)
    h = cast(
      jnp.ndarray,
      nn.Conv(4, (1, 5), padding="SAME").apply(
        {"params": {"kernel": p["conv2_kernel"], "bias": p["conv2_bias"]}},
        h,
      ),
    )
    h = nn.gelu(h)
    expected = np.asarray(
      nn.Dense(8).apply({"params": p["Dense_0"]}, h.reshape((2, 24, 15 * 4)))
    )
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

  def test_conditioned_warmstart_equivalence_and_causality(self):
    base_cfg = nap_inverter.DirectInverterConfig(
      architecture="cochlea_hybrid",
      width=16,
      num_blocks=3,
      coarse_dim=16,
      coarse_hop=32,
      coarse_dilations=(1, 2),
      lookahead=32,
      fs=16000.0,
      conditioned=False,
    )
    cond_cfg = nap_inverter.DirectInverterConfig(
      architecture="cochlea_hybrid",
      width=16,
      num_blocks=3,
      coarse_dim=16,
      coarse_hop=32,
      coarse_dilations=(1, 2),
      lookahead=32,
      fs=16000.0,
      conditioned=True,
      cond_dim=3,
      max_log_gain=4.5,
    )
    rng = np.random.default_rng(11)
    nap = jnp.asarray(0.05 * rng.standard_normal((2, 256, 8)), jnp.float32)
    cond = jnp.asarray(rng.uniform(0.0, 1.0, (2, 8, 3)), jnp.float32)

    base_model = nap_inverter.CochleaDirectInverter(config=base_cfg)
    cond_model = nap_inverter.CochleaDirectInverter(config=cond_cfg)
    base_params = base_model.init(jax.random.PRNGKey(0), nap[:1])
    init_cond = cond_model.init(jax.random.PRNGKey(1), nap[:1], cond=cond[:1])
    merged = nap_inverter.merge_warmstart_params(init_cond, base_params)

    y_base = np.asarray(base_model.apply(base_params, nap))
    y_merged = np.asarray(cond_model.apply(merged, nap, cond=cond))
    np.testing.assert_allclose(y_merged, y_base, atol=1e-5, rtol=1e-5)

    nap_fut = nap.at[:, 128:, :].set(nap[:, 128:, :] + 0.4)
    y_fut = np.asarray(cond_model.apply(merged, nap_fut, cond=cond))
    np.testing.assert_allclose(
      y_merged[:, :128], y_fut[:, :128], atol=1e-6, rtol=1e-6
    )


if __name__ == "__main__":
  absltest.main()
