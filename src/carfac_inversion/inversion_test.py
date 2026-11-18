"""Tests for CARFAC NAP-to-audio inversion."""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from carfac.jax import carfac as carfac_jax
from hp_acoustic.filters import middle_ear
from hp_acoustic.metrics import context as metric_context
from hp_acoustic.metrics import perceptual
from hp_acoustic.metrics import sii
from hp_acoustic.metrics import waveform as waveform_metrics
from hp_acoustic.models import carfac
from hp_acoustic.models import cochlea
from hp_acoustic.models import gammatone

from carfac_inversion import inversion


def _make_speech_like_signal(
  sample_rate_hz: int = 16000,
  duration_s: float = 0.75,
  level_db_spl: float = 65.0,
) -> np.ndarray:
  """Synthesizes a voiced-plus-fricative utterance scaled to ``level_db_spl``."""
  num_samples = int(round(sample_rate_hz * duration_s))
  t = np.arange(num_samples, dtype=np.float64) / sample_rate_hz

  f0 = 125.0 + 25.0 * np.sin(2.0 * np.pi * 2.0 * t)
  phase = np.cumsum(2.0 * np.pi * f0 / sample_rate_hz)
  voiced_env = np.exp(
    -0.5 * ((t - 0.28 * duration_s) / (0.15 * duration_s)) ** 2
  )

  f1_t = 680.0 - 300.0 * np.clip(t / (0.6 * duration_s), 0.0, 1.0)
  f2_t = 1200.0 + 950.0 * np.clip(t / (0.6 * duration_s), 0.0, 1.0)
  voiced = np.zeros_like(t)
  for k in range(1, 32):
    fk = k * f0
    r1 = 1.0 / (1.0 + ((fk - f1_t) / 70.0) ** 2)
    r2 = 0.7 / (1.0 + ((fk - f2_t) / 95.0) ** 2)
    voiced += (r1 + r2) * (fk < 0.45 * sample_rate_hz) * np.sin(k * phase)
  voiced *= voiced_env

  # High-frequency fricative burst bandlimited to 3.5-6.5 kHz in FFT space.
  rng = np.random.default_rng(7)
  freqs = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate_hz)
  noise_spec = np.fft.rfft(rng.standard_normal(num_samples))
  fric_band = (freqs >= 3500.0) & (freqs <= min(6500.0, 0.45 * sample_rate_hz))
  fric = np.fft.irfft(noise_spec * fric_band, n=num_samples)
  fric = fric / (np.std(fric) + 1e-12)
  fric *= 0.35 * np.exp(
    -0.5 * ((t - 0.78 * duration_s) / (0.06 * duration_s)) ** 2
  )

  sig = voiced + fric
  ramp = max(1, int(0.005 * sample_rate_hz))
  taper = np.sin(0.5 * np.pi * np.arange(ramp) / ramp) ** 2
  sig[:ramp] *= taper
  sig[-ramp:] *= taper[::-1]

  target_rms = 10.0 ** ((level_db_spl - sii.CARFAC_RMS_ONE_DB_SPL) / 20.0)
  sig *= target_rms / (np.sqrt(np.mean(sig**2)) + 1e-12)
  return sig.astype(np.float32)


class _SimpleLinearCochlea(cochlea.CochleaModel):
  """Minimal non-CARFAC ``CochleaModel`` verifying protocol fallback."""

  def __init__(self, sample_rate_hz: float = 16000.0, num_channels: int = 8):
    self._sample_rate_hz = sample_rate_hz
    self._num_channels = num_channels
    self._weights = jnp.linspace(0.5, 1.5, num_channels, dtype=jnp.float32)

  @property
  def sample_rate(self) -> float:
    return self._sample_rate_hz

  @property
  def num_channels(self) -> int:
    return self._num_channels

  def __call__(self, waveform: jnp.ndarray) -> jnp.ndarray:
    if waveform.ndim == 1:
      waveform = waveform[:, jnp.newaxis]
    return (
      waveform[:, jnp.newaxis, :] * self._weights[jnp.newaxis, :, jnp.newaxis]
    )


class CarfacInversionTest(parameterized.TestCase):
  """Unit and perceptual quality tests for ``carfac_inversion``."""

  def setUp(self):
    super().setUp()
    self.sample_rate_hz = 16000
    self.healthy = carfac.build(fs=self.sample_rate_hz)

  def test_config_validation_rejects_invalid_values(self):
    with self.assertRaises(ValueError):
      inversion.InversionConfig(mode="invalid")  # pyrefly: ignore[bad-argument-type]
    with self.assertRaises(ValueError):
      inversion.InversionConfig(adam_steps=-1)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(lbfgs_steps=-1)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(min_warm_start_cf_hz=0.0)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(
        min_warm_start_cf_hz=500.0, hf_crossover_hz=400.0
      )
    with self.assertRaises(ValueError):
      inversion.InversionConfig(cutoff_low_hz=-1.0)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(cutoff_high_hz=0.0)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(hf_env_weight=-0.1)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(adam_lr_scale=0.0)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(lbfgs_memory_size=0)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(perceptual_nap_weight=-0.1)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(perceptual_nap_weight=1.1)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(chunk_seconds=0.0)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(chunk_overlap_s=-0.1)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(chunk_preroll_s=-0.1)
    with self.assertRaises(ValueError):
      inversion.InversionConfig(chunk_seconds=0.5, chunk_overlap_s=0.5)

  def test_target_nap_validation_rejects_bad_inputs(self):
    with self.assertRaisesRegex(ValueError, "2 dimensions"):
      inversion.invert_nap(np.zeros((100,)), self.healthy)
    with self.assertRaisesRegex(ValueError, "at least 16 samples"):
      inversion.invert_nap(
        np.zeros((8, self.healthy.num_channels)), self.healthy
      )
    with self.assertRaisesRegex(ValueError, "channel count"):
      inversion.invert_nap(np.zeros((128, 4)), self.healthy)
    with self.assertRaisesRegex(ValueError, "ear count"):
      inversion.invert_nap(
        np.zeros((128, self.healthy.num_channels, 2)), self.healthy
      )
    bad_nap = np.zeros((128, self.healthy.num_channels), dtype=np.float32)
    bad_nap[10, 0] = np.nan
    with self.assertRaisesRegex(ValueError, "finite"):
      inversion.invert_nap(bad_nap, self.healthy)

  def test_silent_rest_nap_returns_zero_waveform(self):
    rest_nap = np.asarray(self.healthy(jnp.zeros((256, 1)))[:, :, 0])
    result = inversion.invert_nap(rest_nap, self.healthy)

    self.assertEqual(result.waveform.shape, (256,))
    self.assertEqual(result.reconstructed_nap.shape, rest_nap.shape)
    # Resting NAP matches zero input directly so recovered waveform must be 0.
    np.testing.assert_allclose(result.waveform, 0.0, atol=1e-7)
    self.assertEqual(result.final_loss, 0.0)

  def test_dsp_warm_start_preserves_shapes_with_front_end_filters(self):
    model = carfac.build(
      fs=32000,
      middle_ear_filter=middle_ear.MiddleEarFilterConfig(enabled=True),
      conductive_filter=(10.0, 15.0),
    )
    sig = _make_speech_like_signal(sample_rate_hz=32000, duration_s=0.25)
    nap_3d = np.asarray(model(jnp.asarray(sig[:, None], dtype=jnp.float32)))

    warm_3d = inversion.dsp_warm_start(nap_3d, model)
    self.assertEqual(warm_3d.shape, (len(sig), 1))
    self.assertTrue(np.all(np.isfinite(warm_3d)))
    self.assertGreater(float(np.std(warm_3d)), 1e-4)

    warm_2d = inversion.dsp_warm_start(nap_3d[:, :, 0], model)
    self.assertEqual(warm_2d.shape, (len(sig),))
    # Same 2D/3D NAP values and config seed must produce identical warm start.
    np.testing.assert_allclose(warm_2d, warm_3d[:, 0], rtol=1e-6, atol=1e-6)

  def test_short_clip_runs_without_sai_shape_failure(self):
    sig = _make_speech_like_signal(sample_rate_hz=16000, duration_s=0.032)
    nap = np.asarray(self.healthy(jnp.asarray(sig[:, None]))[:, :, 0])
    cfg = inversion.InversionConfig(
      mode="perceptual", adam_steps=5, lbfgs_steps=3
    )
    result = inversion.invert_nap(nap, self.healthy, cfg)
    self.assertEqual(result.waveform.shape, (len(sig),))
    self.assertTrue(np.all(np.isfinite(result.waveform)))

  @parameterized.named_parameters(
    ("50dB_SPL", 50.0, 60, 60, 8.0, 0.88, 0.95),
    ("65dB_SPL", 65.0, 60, 60, 6.0, 0.85, 0.95),
    ("80dB_SPL", 80.0, 120, 120, 1.5, 0.72, 0.55),
  )
  def test_round_trip_speech_reconstruction_quality(
    self,
    level_db_spl: float,
    adam_steps: int,
    lbfgs_steps: int,
    min_si_sdr_db: float,
    min_stoi: float,
    min_haspi: float,
  ):
    sig = _make_speech_like_signal(
      sample_rate_hz=self.sample_rate_hz,
      duration_s=0.75,
      level_db_spl=level_db_spl,
    )
    nap = np.asarray(self.healthy(jnp.asarray(sig[:, None]))[:, :, 0])
    cfg = inversion.InversionConfig(
      mode="exact", adam_steps=adam_steps, lbfgs_steps=lbfgs_steps
    )
    result = inversion.invert_nap(nap, self.healthy, cfg)

    self.assertEqual(result.waveform.shape, sig.shape)
    self.assertEqual(result.reconstructed_nap.shape, nap.shape)
    self.assertLen(result.loss_history, adam_steps + lbfgs_steps)

    nap_corr = float(
      np.corrcoef(result.reconstructed_nap.ravel(), nap.ravel())[0, 1]
    )
    self.assertGreater(nap_corr, 0.98)

    ctx = metric_context.Context(
      reference=sig,
      estimation=result.waveform,
      sample_rate=self.sample_rate_hz,
      hearing_loss_db=np.zeros(6),
    )
    self.assertGreater(waveform_metrics.si_sdr(ctx), min_si_sdr_db)
    self.assertGreater(perceptual.stoi(ctx), min_stoi)
    self.assertGreater(perceptual.haspi(ctx), min_haspi)

  def test_round_trip_32khz_default_rate(self):
    model_32k = carfac.build(fs=32000)
    sig = _make_speech_like_signal(
      sample_rate_hz=32000, duration_s=0.5, level_db_spl=65.0
    )
    nap = np.asarray(model_32k(jnp.asarray(sig[:, None]))[:, :, 0])
    cfg = inversion.InversionConfig(mode="exact", adam_steps=80, lbfgs_steps=80)
    result = inversion.invert_nap(nap, model_32k, cfg)

    nap_corr = float(
      np.corrcoef(result.reconstructed_nap.ravel(), nap.ravel())[0, 1]
    )
    self.assertGreater(nap_corr, 0.99)
    ctx = metric_context.Context(
      reference=sig,
      estimation=result.waveform,
      sample_rate=32000,
      hearing_loss_db=np.zeros(6),
    )
    self.assertGreater(waveform_metrics.si_sdr(ctx), 10.0)
    self.assertGreater(perceptual.stoi(ctx), 0.88)

  def test_binaural_two_ear_inversion(self):
    params = carfac_jax.CarfacDesignParameters.with_n_ears(2, fs=16000)
    params.ears[0].ihc.ihc_style = "two_cap_with_syn"
    params.ears[1].ihc.ihc_style = "two_cap_with_syn"
    hypers, weights, state = carfac_jax.design_and_init_carfac(params)
    model_2ear = carfac.CarfacCochleaModel(
      hypers=hypers, weights=weights, initial_state=state, fs=16000.0
    )

    sig_l = _make_speech_like_signal(
      sample_rate_hz=16000, duration_s=0.25, level_db_spl=60.0
    )
    sig_r = (np.roll(sig_l, 8) * 0.7).astype(np.float32)
    stereo = np.stack([sig_l, sig_r], axis=-1)
    nap_stereo = np.asarray(model_2ear(jnp.asarray(stereo)))

    cfg = inversion.InversionConfig(mode="exact", adam_steps=45, lbfgs_steps=45)
    result = inversion.invert_nap(nap_stereo, model_2ear, cfg)

    self.assertEqual(result.waveform.shape, stereo.shape)
    self.assertEqual(result.reconstructed_nap.shape, nap_stereo.shape)
    self.assertGreater(
      float(np.corrcoef(result.waveform[:, 0], sig_l)[0, 1]), 0.95
    )
    self.assertGreater(
      float(np.corrcoef(result.waveform[:, 1], sig_r)[0, 1]), 0.95
    )

  def test_cross_cochlea_perceptual_auralization(self):
    sig = _make_speech_like_signal(
      sample_rate_hz=self.sample_rate_hz, duration_s=0.75, level_db_spl=65.0
    )
    impaired = carfac.build(fs=self.sample_rate_hz, ohc_health=0.5)
    nap_impaired = np.asarray(impaired(jnp.asarray(sig[:, None]))[:, :, 0])

    cfg = inversion.InversionConfig(
      mode="perceptual", adam_steps=60, lbfgs_steps=60
    )
    result = inversion.invert_nap(nap_impaired, self.healthy, cfg)

    # Compare smoothed temporal envelopes across cochleae (not fine carrier phase).
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / self.sample_rate_hz)
    lpf = 1.0 / (1.0 + (freqs / 200.0) ** 4)
    env_imp = np.fft.irfft(
      np.fft.rfft(np.maximum(nap_impaired, 0.0), axis=0) * lpf[:, None],
      n=len(sig),
      axis=0,
    )
    env_aur = np.fft.irfft(
      np.fft.rfft(np.maximum(result.reconstructed_nap, 0.0), axis=0)
      * lpf[:, None],
      n=len(sig),
      axis=0,
    )
    env_corr = float(np.corrcoef(env_imp.ravel(), env_aur.ravel())[0, 1])
    self.assertGreater(env_corr, 0.93)

    # 0.5 OHC loss elevates threshold, so healthy-equivalent audio must be quieter.
    rms_orig = float(np.sqrt(np.mean(sig**2)))
    rms_aur = float(np.sqrt(np.mean(result.waveform**2)))
    self.assertLess(rms_aur, rms_orig * 0.5)

  def test_gammatone_cochlea_model_inversion(self):
    model = gammatone.gammatone_healthy(fs=24000, n_channels=32)
    sig = _make_speech_like_signal(
      sample_rate_hz=24000, duration_s=0.25, level_db_spl=65.0
    )
    nap = np.asarray(model(jnp.asarray(sig[:, None]))[:, :, 0])

    cfg = inversion.InversionConfig(mode="exact", adam_steps=50, lbfgs_steps=50)
    result = inversion.invert_nap(nap, model, cfg)

    self.assertEqual(result.waveform.shape, sig.shape)
    nap_corr = float(
      np.corrcoef(result.reconstructed_nap.ravel(), nap.ravel())[0, 1]
    )
    self.assertGreater(nap_corr, 0.97)

  def test_generic_cochlea_protocol_fallback(self):
    model = _SimpleLinearCochlea(sample_rate_hz=16000.0, num_channels=8)
    t = np.arange(512, dtype=np.float32) / 16000.0
    # Integer-cycle 437.5 Hz tone avoids periodic FFT edge discontinuities.
    sig = (0.02 * np.sin(2.0 * np.pi * 437.5 * t)).astype(np.float32)
    nap_3d = np.asarray(model(jnp.asarray(sig[:, None])))

    cfg = inversion.InversionConfig(mode="exact", adam_steps=40, lbfgs_steps=30)
    result = inversion.invert_nap(nap_3d, model, cfg)
    self.assertEqual(result.waveform.shape, (512, 1))
    # Linear channel sum reconstructs 437.5 Hz tone to high accuracy inside band.
    np.testing.assert_allclose(result.waveform[:, 0], sig, rtol=1e-2, atol=5e-4)

  def test_ola_chunked_matches_full_clip_quality(self):
    sig = _make_speech_like_signal(
      sample_rate_hz=self.sample_rate_hz, duration_s=1.0, level_db_spl=65.0
    )
    nap = np.asarray(self.healthy(jnp.asarray(sig[:, None]))[:, :, 0])

    full_cfg = inversion.InversionConfig(
      mode="exact", adam_steps=70, lbfgs_steps=70, chunk_seconds=None
    )
    ola_cfg = inversion.InversionConfig(
      mode="exact",
      adam_steps=70,
      lbfgs_steps=70,
      chunk_seconds=0.5,
      chunk_overlap_s=0.1,
      chunk_preroll_s=0.064,
    )
    res_full = inversion.invert_nap(nap, self.healthy, full_cfg)
    res_ola = inversion.invert_nap(nap, self.healthy, ola_cfg)

    self.assertEqual(res_ola.waveform.shape, sig.shape)
    self.assertEqual(res_ola.reconstructed_nap.shape, nap.shape)

    ctx_full = metric_context.Context(
      reference=sig,
      estimation=res_full.waveform,
      sample_rate=self.sample_rate_hz,
      hearing_loss_db=np.zeros(6),
    )
    ctx_ola = metric_context.Context(
      reference=sig,
      estimation=res_ola.waveform,
      sample_rate=self.sample_rate_hz,
      hearing_loss_db=np.zeros(6),
    )
    ola_nap_corr = float(
      np.corrcoef(res_ola.reconstructed_nap.ravel(), nap.ravel())[0, 1]
    )
    self.assertGreater(ola_nap_corr, 0.99)
    ola_stoi = perceptual.stoi(ctx_ola)
    full_stoi = perceptual.stoi(ctx_full)
    self.assertGreater(ola_stoi, 0.85)
    self.assertGreater(ola_stoi, full_stoi - 0.01)
    self.assertGreater(
      waveform_metrics.si_sdr(ctx_ola),
      waveform_metrics.si_sdr(ctx_full) - 1.5,
    )

  def test_apical_low_frequency_reconstruction(self):
    fs = self.sample_rate_hz
    t = np.arange(int(0.4 * fs), dtype=np.float64) / fs
    sig = 0.7 * np.sin(2.0 * np.pi * 75.0 * t) + 0.3 * np.sin(
      2.0 * np.pi * 440.0 * t
    )
    ramp = max(1, int(0.01 * fs))
    taper = np.sin(0.5 * np.pi * np.arange(ramp) / ramp) ** 2
    sig[:ramp] *= taper
    sig[-ramp:] *= taper[::-1]
    sig *= (10.0 ** ((55.0 - sii.CARFAC_RMS_ONE_DB_SPL) / 20.0)) / (
      np.sqrt(np.mean(sig**2)) + 1e-12
    )
    sig_f32 = sig.astype(np.float32)

    nap = np.asarray(self.healthy(jnp.asarray(sig_f32[:, None]))[:, :, 0])
    cfg = inversion.InversionConfig(mode="exact", adam_steps=20, lbfgs_steps=60)
    res = inversion.invert_nap(nap, self.healthy, cfg)

    freqs = np.fft.rfftfreq(len(sig_f32), d=1.0 / fs)
    lf_mask = (freqs >= 55.0) & (freqs <= 200.0)
    ref_lf = np.fft.irfft(np.fft.rfft(sig_f32) * lf_mask, n=len(sig_f32))
    est_lf = np.fft.irfft(np.fft.rfft(res.waveform) * lf_mask, n=len(sig_f32))
    lf_corr = float(np.corrcoef(ref_lf, est_lf)[0, 1])
    self.assertGreater(lf_corr, 0.90)

    ctx = metric_context.Context(
      reference=sig_f32,
      estimation=res.waveform,
      sample_rate=fs,
      hearing_loss_db=np.zeros(6),
    )
    self.assertGreater(waveform_metrics.si_sdr(ctx), 8.0)

  def test_no_optimization_steps_returns_projected_warm_start(self):
    sig = _make_speech_like_signal(sample_rate_hz=16000, duration_s=0.1)
    nap = np.asarray(self.healthy(jnp.asarray(sig[:, None]))[:, :, 0])
    cfg = inversion.InversionConfig(adam_steps=0, lbfgs_steps=0)
    result = inversion.invert_nap(nap, self.healthy, cfg)
    self.assertEmpty(result.loss_history)
    self.assertGreater(float(np.std(result.waveform)), 1e-4)

  def test_per_ear_ohc_scale_and_retuned_low_spl_calibration(self):
    # Verify per-ear and per-channel ohc_scale broadcasting in BM IR measurement.
    params2 = carfac_jax.CarfacDesignParameters.with_n_ears(2, fs=16000)
    params2.ears[0].ihc.ihc_style = "two_cap_with_syn"
    params2.ears[1].ihc.ihc_style = "two_cap_with_syn"
    h2, w2, s2 = carfac_jax.design_and_init_carfac(params2)
    model_2ear = carfac.CarfacCochleaModel(
      hypers=h2, weights=w2, initial_state=s2, fs=16000.0
    )
    irs_vec = inversion._measure_carfac_bm_irs(
      model_2ear, ir_len=256, ohc_scale=np.array([1.0, 0.55])
    )
    irs_ear0 = inversion._measure_carfac_bm_irs(
      model_2ear, ir_len=256, ohc_scale=1.0
    )
    irs_ear1 = inversion._measure_carfac_bm_irs(
      model_2ear, ir_len=256, ohc_scale=0.55
    )
    np.testing.assert_allclose(irs_vec[:, :, 0], irs_ear0[:, :, 0], rtol=1e-6)
    np.testing.assert_allclose(irs_vec[:, :, 1], irs_ear1[:, :, 1], rtol=1e-6)

    # Verify adaptive scalar gain bounds for high-gain CARFAC presets (Option C,
    # Cal_rms1 ~ 124.2 dB SPL) at 20 dB SPL (~6.17e-6 RMS, below 5e-5 Baseline floor).
    params_c = carfac_jax.CarfacDesignParameters(fs=16000.0)
    params_c.ears[0].ihc.ihc_style = "two_cap_with_syn"
    params_c.ears[0].car.use_delay_buffer = True
    params_c.ears[0].car.min_zeta = 0.07
    params_c.ears[0].car.max_zeta = 0.35
    params_c.ears[0].car.erb_per_step = 0.35
    params_c.ears[0].car.zero_ratio = 1.300
    hc, wc, sc = carfac_jax.design_and_init_carfac(params_c)
    model_c = carfac.CarfacCochleaModel(
      hypers=hc, weights=wc, initial_state=sc, fs=16000.0
    )
    true_rms = 10.0 ** ((20.0 - 124.2) / 20.0)
    sig = _make_speech_like_signal(sample_rate_hz=16000, duration_s=0.4)
    sig = (sig * (true_rms / float(np.sqrt(np.mean(sig**2))))).astype(
      np.float32
    )
    nap = np.asarray(model_c(jnp.asarray(sig[:, None]))[:, :, 0])
    ws = inversion.dsp_warm_start(nap, model_c)
    ws_rms = float(np.sqrt(np.mean(ws**2)))
    self.assertGreater(ws_rms, true_rms * 0.5)
    self.assertLess(ws_rms, true_rms * 1.6)
    nap_ws = np.asarray(model_c(jnp.asarray(ws[:, None]))[:, :, 0])
    self.assertGreater(
      float(np.corrcoef(nap_ws.ravel(), nap.ravel())[0, 1]), 0.90
    )


if __name__ == "__main__":
  absltest.main()
