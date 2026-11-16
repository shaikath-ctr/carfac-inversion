"""Tests for the ``invert_nap`` WAV CLI."""

import pathlib
import tempfile

import numpy as np
from absl.testing import absltest
from hp_acoustic.io import audio_io
from hp_acoustic.models import carfac

from carfac_inversion.scripts import invert_nap


class InvertNapScriptTest(absltest.TestCase):
  """Tests for ``carfac_inversion.scripts.invert_nap``."""

  def _write_test_wav(self, path: pathlib.Path, fs: int = 16000) -> None:
    t = np.arange(int(0.1 * fs), dtype=np.float32) / fs
    tone = (0.02 * np.sin(2.0 * np.pi * 500.0 * t)).astype(np.float32)
    audio_io.save_audio_to_file(tone, str(path), fs)

  def test_round_trip_cli_writes_wav(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      in_wav = pathlib.Path(tmp_dir) / "in.wav"
      out_wav = pathlib.Path(tmp_dir) / "out.wav"
      self._write_test_wav(in_wav)

      result = invert_nap.run_invert_wav(
        input_wav=str(in_wav),
        output_wav=str(out_wav),
        sample_rate_hz=16000,
        adam_steps=5,
        lbfgs_steps=3,
      )
      self.assertTrue(out_wav.is_file())
      loaded, sr = audio_io.read_wav(str(out_wav), target_sr=16000)
      self.assertEqual(sr, 16000)
      self.assertEqual(loaded.shape, result.waveform.shape)

  def test_fitted_profile_and_ohc_mutual_exclusion(self):
    with self.assertRaisesRegex(ValueError, "at most one"):
      invert_nap.run_invert_wav(
        input_wav="unused.wav",
        output_wav="unused_out.wav",
        source_ohc_health=0.5,
        fitted_profile_path="profile.npz",
      )

  def test_fitted_profile_auralization_path(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      in_wav = pathlib.Path(tmp_dir) / "in.wav"
      out_wav = pathlib.Path(tmp_dir) / "out.wav"
      profile_path = pathlib.Path(tmp_dir) / "fitted.npz"
      self._write_test_wav(in_wav, fs=16000)

      n_ch = carfac.build(fs=16000).num_channels
      np.savez(
        profile_path,
        ohc_health=np.full(n_ch, 0.5, dtype=np.float32),
        conductive_loss_db=np.array([5.0, 10.0], dtype=np.float32),
        sample_rate=16000.0,
        high_f_factor=0.0,
      )

      result = invert_nap.run_invert_wav(
        input_wav=str(in_wav),
        output_wav=str(out_wav),
        sample_rate_hz=16000,
        fitted_profile_path=str(profile_path),
        adam_steps=5,
        lbfgs_steps=3,
      )
      self.assertTrue(out_wav.is_file())
      self.assertTrue(np.all(np.isfinite(result.waveform)))


if __name__ == "__main__":
  absltest.main()
