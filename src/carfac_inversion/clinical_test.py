"""Unit tests for clinical audiograms, CARFAC fitting, and NAL-R baselines."""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from hp_acoustic.models import carfac

from carfac_inversion import clinical


class ClinicalTest(absltest.TestCase):
  """Verifies Bisgaard, ISO 7029:2017, CARFAC parameter mapping, and NAL-R."""

  def test_bisgaard_profiles_cover_standard_codes(self):
    codes = (
      "N0",
      "N1",
      "N2",
      "N3",
      "N4",
      "N5",
      "N6",
      "N7",
      "S1",
      "S2",
      "S3",
    )
    for code in codes:
      aud6 = clinical.bisgaard_audiogram_6(code)
      self.assertEqual(aud6.shape, (6,))
      self.assertTrue(np.all(np.isfinite(aud6)))
      prof = clinical.build_clinical_profile(
        f"bisgaard:{code}", sample_rate_hz=16000.0
      )
      cond = prof.conditioning_features()
      self.assertEqual(cond.shape, (65, 3))
      self.assertEqual(cond.dtype, np.float32)
      if code == "N0":
        self.assertTrue(prof.is_healthy)
        np.testing.assert_array_equal(cond, np.zeros((65, 3), dtype=np.float32))
      else:
        self.assertFalse(prof.is_healthy)
        self.assertGreater(float(np.max(cond)), 0.0)

    s2 = clinical.build_clinical_profile("S2", sample_rate_hz=32000.0)
    self.assertEqual(s2.channel_freqs_hz.shape, (77,))
    self.assertGreater(float(np.max(s2.v_half_shift)), 0.0)
    self.assertLess(float(np.min(s2.ohc_health)), 0.2)

    with self.assertRaises(ValueError):
      clinical.bisgaard_audiogram_6("X9")

  def test_iso7029_age_and_percentile_monotonicity(self):
    a18 = clinical.iso7029_audiogram_6(18.0, sex="male", percentile=0.50)
    a50 = clinical.iso7029_audiogram_6(50.0, sex="male", percentile=0.50)
    a75 = clinical.iso7029_audiogram_6(75.0, sex="male", percentile=0.50)
    np.testing.assert_array_equal(a18, np.zeros((6,), dtype=np.float32))
    self.assertTrue(np.all(a75 > a50))
    self.assertTrue(np.all(a50 > a18))

    p10 = clinical.iso7029_audiogram_6(70.0, sex="female", percentile=0.10)
    p50 = clinical.iso7029_audiogram_6(70.0, sex="female", percentile=0.50)
    p90 = clinical.iso7029_audiogram_6(70.0, sex="female", percentile=0.90)
    self.assertTrue(np.all(p90 > p50))
    self.assertTrue(np.all(p50 > p10))

    prof = clinical.build_clinical_profile(
      "iso7029:70:male:0.5", sample_rate_hz=32000.0
    )
    self.assertEqual(prof.conditioning_features().shape, (77, 3))
    cm = clinical.build_cochlea_from_profile(prof)
    self.assertEqual(int(cm.num_channels), 77)

  def test_fit_steps_smoke_and_cochlea_replacement(self):
    base = carfac.build(fs=16000)
    prof = clinical.build_clinical_profile(
      "bisgaard:N2", sample_rate_hz=16000.0, fit_steps=5
    )
    self.assertEqual(prof.ohc_health.shape, (65,))
    self.assertTrue(np.all(prof.ohc_health >= 0.0))
    self.assertTrue(np.all(prof.ohc_health <= 1.0))

    cm = clinical.build_cochlea_from_profile(prof, base_cochlea=base)
    out = np.asarray(cm(jnp.zeros((256, 1), dtype=jnp.float32)))
    self.assertEqual(out.shape, (256, 65, 1))
    self.assertTrue(np.all(np.isfinite(out)))

  def test_apply_nalr_identity_and_hf_amplification(self):
    fs = 32000.0
    t = np.arange(8000, dtype=np.float32) / fs
    wav = (
      0.01 * np.sin(2.0 * np.pi * 400.0 * t)
      + 0.01 * np.sin(2.0 * np.pi * 4000.0 * t)
    ).astype(np.float32)

    n0_out = clinical.apply_nalr(wav, clinical.bisgaard_audiogram_6("N0"), fs)
    np.testing.assert_allclose(n0_out, wav, atol=1e-6, rtol=1e-6)

    n3_out = clinical.apply_nalr(wav, clinical.bisgaard_audiogram_6("N3"), fs)
    self.assertEqual(n3_out.shape, wav.shape)
    self.assertGreater(
      float(np.sqrt(np.mean(n3_out**2))), float(np.sqrt(np.mean(wav**2)))
    )


if __name__ == "__main__":
  absltest.main()
