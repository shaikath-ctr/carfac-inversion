"""Unit tests for float32-safe differentiable HASQI v2 and HASPI v2 losses."""

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax_haspi_hasqi import haspi as haspi_lib
from jax_haspi_hasqi import hasqi as hasqi_lib

from carfac_inversion import clinical
from carfac_inversion import perceptual_losses


class PerceptualLossesTest(absltest.TestCase):
  """Verifies forward parity with jax-haspi-hasqi and finite autodiff."""

  def _sample_signals(self) -> tuple[jnp.ndarray, jnp.ndarray]:
    rng = np.random.default_rng(4)
    fs = 32000.0
    t = np.arange(16000, dtype=np.float32) / fs
    env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t)
    ref = (
      0.015
      * env
      * (
        np.sin(2.0 * np.pi * 300.0 * t) + 0.5 * np.sin(2.0 * np.pi * 1800.0 * t)
      )
    )
    proc = ref + 0.002 * rng.standard_normal(len(t)).astype(np.float32)
    return jnp.asarray(ref, dtype=jnp.float32), jnp.asarray(
      proc, dtype=jnp.float32
    )

  def test_forward_parity_with_jax_haspi_hasqi(self):
    ref, proc = self._sample_signals()
    level1 = 104.0 + 20.0 * float(np.log10(np.sqrt(2.0)))
    for code in ("N0", "N3", "S2"):
      hl = jnp.asarray(clinical.bisgaard_audiogram_6(code), dtype=jnp.float32)
      q_ref, nl_ref, lin_ref, _ = hasqi_lib.hasqi_v2(
        ref, 32000.0, proc, 32000.0, hl, level1=level1
      )
      q_got, nl_got, lin_got = perceptual_losses.differentiable_hasqi_v2(
        ref, proc, hl, sample_rate_hz=32000.0, level1=level1, align_inputs=True
      )
      self.assertAlmostEqual(float(q_got), float(q_ref), delta=5e-4)
      self.assertAlmostEqual(float(nl_got), float(nl_ref), delta=5e-4)
      self.assertAlmostEqual(float(lin_got), float(lin_ref), delta=5e-4)

      p_ref, _ = haspi_lib.haspi_v2(
        ref, 32000.0, proc, 32000.0, hl, level1=level1
      )
      p_got = perceptual_losses.differentiable_haspi_v2(
        ref, proc, hl, sample_rate_hz=32000.0, level1=level1, align_inputs=True
      )
      self.assertAlmostEqual(float(p_got), float(p_ref), delta=1.5e-3)

  def test_gradients_finite_on_noisy_exact_and_silence(self):
    ref, proc = self._sample_signals()
    hl = jnp.asarray(clinical.bisgaard_audiogram_6("N3"), dtype=jnp.float32)

    def _loss(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
      loss_val, _ = perceptual_losses.batched_hasqi_haspi_loss(
        x[None, :], y[None, :], hl[None, :], hasqi_weight=0.5, haspi_weight=0.5
      )
      return loss_val

    vg = jax.jit(jax.value_and_grad(_loss))
    # 1) Noisy signal
    v_noisy, g_noisy = vg(proc, ref)
    self.assertTrue(np.isfinite(float(v_noisy)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(g_noisy))))
    self.assertGreater(float(jnp.linalg.norm(g_noisy)), 0.0)

    # 2) Exact identical reference == processed (std(0) edge case)
    v_exact, g_exact = vg(ref, ref)
    self.assertTrue(np.isfinite(float(v_exact)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(g_exact))))

    # 3) All-zero silence edge case
    zeros = jnp.zeros_like(ref)
    v_zero, g_zero = vg(zeros, zeros)
    self.assertTrue(np.isfinite(float(v_zero)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(g_zero))))
    self.assertEqual(float(jnp.linalg.norm(g_zero)), 0.0)

  def test_batched_mask_zeroes_excluded_examples(self):
    ref, proc = self._sample_signals()
    hl = jnp.asarray(clinical.bisgaard_audiogram_6("N3"), dtype=jnp.float32)
    refs = jnp.stack([ref, ref])
    procs = jnp.stack([proc, jnp.zeros_like(proc)])
    hls = jnp.stack([hl, hl])
    loss, aux = perceptual_losses.batched_hasqi_haspi_loss(
      refs, procs, hls, mask=jnp.array([1.0, 0.0], dtype=jnp.float32)
    )
    loss_1, _ = perceptual_losses.batched_hasqi_haspi_loss(
      ref[None], proc[None], hl[None]
    )
    self.assertAlmostEqual(float(loss), float(loss_1), delta=1e-5)
    self.assertGreater(float(aux["hasqi"]), 0.0)
    self.assertGreater(float(aux["haspi"]), 0.0)

  def test_batched_shared_ear_matches_individual_scores(self):
    ref, proc = self._sample_signals()
    hl = jnp.asarray(clinical.bisgaard_audiogram_6("N3"), dtype=jnp.float32)
    q_ref, _, _ = perceptual_losses.differentiable_hasqi_v2(
      ref, proc, hl, sample_rate_hz=32000.0, align_inputs=False
    )
    p_ref = perceptual_losses.differentiable_haspi_v2(
      ref, proc, hl, sample_rate_hz=32000.0, align_inputs=False
    )
    _, aux = perceptual_losses.batched_hasqi_haspi_loss(
      ref[None], proc[None], hl[None], sample_rate_hz=32000.0
    )
    self.assertAlmostEqual(float(aux["hasqi"]), float(q_ref), delta=5e-3)
    self.assertAlmostEqual(float(aux["haspi"]), float(p_ref), delta=5e-3)


if __name__ == "__main__":
  absltest.main()
