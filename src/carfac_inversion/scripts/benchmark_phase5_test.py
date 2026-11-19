"""Smoke tests for Phase 5 clinical hearing-aid prescription benchmark."""

import pathlib
import tempfile

import jax
import jax.numpy as jnp
from absl.testing import absltest

from carfac_inversion import nap_inverter
from carfac_inversion.scripts import benchmark_phase5


class BenchmarkPhase5Test(absltest.TestCase):
  """Verifies benchmark_phase5 runs end-to-end across unaided, NAL-R, & causal."""

  def test_run_phase5_benchmark_smoke(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = pathlib.Path(tmp)
      cfg = nap_inverter.DirectInverterConfig(
        architecture="cochlea_hybrid",
        width=16,
        num_blocks=3,
        coarse_dim=16,
        coarse_hop=32,
        lookahead=32,
        fs=16000.0,
        conditioned=True,
        cond_dim=3,
        max_log_gain=4.5,
      )
      model = nap_inverter.CochleaDirectInverter(config=cfg)
      params = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 256, 65), dtype=jnp.float32),
        cond=jnp.zeros((1, 65, 3), dtype=jnp.float32),
      )
      ckpt = root / "phase5_smoke.npz"
      nap_inverter.save_direct_checkpoint(ckpt, params, cfg)

      out_dir = root / "bench_out"
      rows = benchmark_phase5.run_phase5_benchmark(
        input_dir=None,
        output_dir=str(out_dir),
        max_clips=1,
        clip_seconds=0.35,
        sample_rate_hz=16000,
        levels_db_spl=(65.0,),
        profiles=("bisgaard:N3", "iso7029:70:male:0.5"),
        methods=("unaided", "nalr", "phase5_direct"),
        phase5_checkpoint=str(ckpt),
      )
      self.assertLen(rows, 6)
      self.assertTrue((out_dir / "benchmark_phase5.csv").is_file())
      self.assertTrue((out_dir / "summary.md").is_file())


if __name__ == "__main__":
  absltest.main()
