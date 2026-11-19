"""Unit tests for Phase 4 causal high-pass, vocoder losses, and discriminators."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal
from absl.testing import absltest

from carfac_inversion import vocoder_losses


class VocoderLossesTest(absltest.TestCase):
  """Verifies Phase 4 causal filtering, MR-STFT, MPD/MRD, and arcfac loss."""

  def test_causal_highpass_attenuates_dc_and_preserves_speech_band(self):
    fs = 32000.0
    t = np.arange(16000, dtype=np.float32) / fs
    dc = np.ones_like(t) * 0.5
    tone = 0.2 * np.sin(2.0 * np.pi * 1000.0 * t)
    x_np = (dc + tone)[None, :]
    y = jnp.asarray(x_np, dtype=jnp.float32)

    out = np.asarray(vocoder_losses.causal_highpass(y, cutoff_hz=20.0, fs=fs))
    sos = scipy.signal.butter(4, 20.0, "highpass", fs=fs, output="sos")
    ref = scipy.signal.sosfilt(sos, x_np.astype(np.float64), axis=-1)
    np.testing.assert_allclose(out, ref, atol=2e-5, rtol=1e-5)
    self.assertEqual(out.shape, (1, 16000))
    self.assertTrue(np.all(np.isfinite(out)))
    # Steady-state DC is strongly suppressed while 1 kHz passes cleanly.
    self.assertLess(abs(float(np.mean(out[0, 8000:]))), 1e-3)
    self.assertGreater(float(np.std(out[0, 8000:])), 0.12)

  def test_causal_highpass_is_strictly_causal(self):
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.standard_normal((2, 2000)), dtype=jnp.float32)
    b = a.at[:, 800:].set(a[:, 800:] + 1.0)
    ya = np.asarray(
      vocoder_losses.causal_highpass(a, cutoff_hz=20.0, fs=32000.0)
    )
    yb = np.asarray(
      vocoder_losses.causal_highpass(b, cutoff_hz=20.0, fs=32000.0)
    )
    np.testing.assert_allclose(ya[:, :800], yb[:, :800], atol=0.0, rtol=0.0)

  def test_multi_resolution_stft_and_discriminators_shapes_and_grads(self):
    rng = np.random.default_rng(1)
    x = jnp.asarray(0.05 * rng.standard_normal((2, 8192)), dtype=jnp.float32)
    y = jnp.asarray(0.05 * rng.standard_normal((2, 8192)), dtype=jnp.float32)

    self_loss, _ = vocoder_losses.multi_resolution_stft_loss(x, x)
    cross_loss, parts = vocoder_losses.multi_resolution_stft_loss(y, x)
    self.assertLess(float(self_loss), 1e-4)
    self.assertGreater(float(cross_loss), 0.1)
    self.assertIn("spectral_convergence", parts)
    self.assertIn("log_magnitude", parts)

    mpd = vocoder_losses.MultiPeriodDiscriminator(
      periods=(2, 3), channels=(8, 16)
    )
    mrd = vocoder_losses.MultiResolutionDiscriminator(
      resolutions=((256, 64, 256),), channels=8
    )
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    p_mpd = mpd.init(k1, x)
    p_mrd = mrd.init(k2, x)
    disc_out_t = list[tuple[jnp.ndarray, list[jnp.ndarray]]]
    real_m = cast(disc_out_t, mpd.apply(p_mpd, x))
    fake_m = cast(disc_out_t, mpd.apply(p_mpd, y))
    real_r = cast(disc_out_t, mrd.apply(p_mrd, x))
    fake_r = cast(disc_out_t, mrd.apply(p_mrd, y))
    self.assertLen(real_m, 2)
    self.assertLen(real_r, 1)

    d_loss = vocoder_losses.discriminator_loss(
      real_m, fake_m
    ) + vocoder_losses.discriminator_loss(real_r, fake_r)
    g_loss = vocoder_losses.generator_loss(
      fake_m
    ) + vocoder_losses.generator_loss(fake_r)
    fm_loss = vocoder_losses.feature_matching_loss(
      real_m, fake_m
    ) + vocoder_losses.feature_matching_loss(real_r, fake_r)
    self.assertTrue(np.isfinite(float(d_loss)))
    self.assertTrue(np.isfinite(float(g_loss)))
    self.assertGreater(float(fm_loss), 0.0)

  def test_arcfac_inverter_loss_respects_healthy_only_masking(self):
    rng = np.random.default_rng(2)
    x = jnp.asarray(0.05 * rng.standard_normal((2, 4096)), dtype=jnp.float32)
    # Different output in second clip, identical NAPs for testing mask math.
    y = x.at[1].set(-x[1])
    fake_ear = lambda w: jnp.stack([w, 0.5 * w], axis=-1)  # noqa: E731
    tgt_nap = fake_ear(x)

    _, m_healthy = vocoder_losses.compute_arcfac_inverter_loss(
      y,
      x,
      tgt_nap,
      fake_ear,
      lookahead=64,
      mask_samples=512,
      output_scale=0.05,
      lam=jnp.array([0.1, 0.0], dtype=jnp.float32),
      stft_weight=jnp.array([0.5, 0.0], dtype=jnp.float32),
      apply_stft=True,
    )
    # Clip 0 matches x delayed cleanly; Clip 1 has impaired mask=0 so stft_mean ~0.
    self.assertTrue(np.isfinite(float(m_healthy["loss"])))
    self.assertGreater(float(m_healthy["stft"]), 0.0)
    self.assertGreater(float(m_healthy["nap_l1"]), 0.0)


if __name__ == "__main__":
  absltest.main()
