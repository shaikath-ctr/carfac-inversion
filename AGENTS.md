This repository follows the AFHI engineering conventions. Copies of the
shared files live in `.afhi/`; `.afhi/REVISION` names the adopted commit.

@.afhi/index.md

# Repository guide (`carfac-inversion`)

Differentiable JAX inversion from CARFAC neural activity patterns (`NAPs`)
and stabilized auditory images (`SAIs`) back to 32 kHz audio, paired with
clinical hearing-loss fitting and perceptual (`HASQI`/`HASPI`) objectives.

## Commands

Run checks and entry points through `uv` (or `.venv/bin/` when `uv` is not on
`PATH`):

```bash
uv sync --dev
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
uv run pyrefly check
uv run pytest -v
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

CLI scripts run as module entry points with `absl` flags:

```bash
uv run python -m carfac_inversion.scripts.invert_nap --help
uv run python -m carfac_inversion.scripts.benchmark_arcfac --help
uv run python -m carfac_inversion.scripts.benchmark_suite --help
uv run python -m carfac_inversion.scripts.benchmark_phase5 --help
uv run python -m carfac_inversion.scripts.train_phase4 --help
uv run python -m carfac_inversion.scripts.train_phase5 --help
uv run python -m carfac_inversion.scripts.train_sai --help
```

## Layout

```text
pyproject.toml  uv.lock  .python-version  .pre-commit-config.yaml
AGENTS.md  CLAUDE.md  README.md  PROGRESS.md
.afhi/                                   # synced shared AFHI conventions
docs/                                    # phase evaluation reports & figures
src/carfac_inversion/
  __init__.py                            # public API, __all__, __version__
  inversion.py  inversion_test.py        # physics DSP warm start + L-BFGS solver
  neural.py     neural_test.py           # residual neural refiner + OLA
  nap_inverter.py  nap_inverter_test.py  # causal CochleaDirectInverter
  sai_inverter.py  sai_inverter_test.py  # DDSP/COLA & causal strobe SAI inverters
  vocoder_losses.py  vocoder_losses_test.py      # MR-STFT, spectral excess, MPD/MRD
  perceptual_losses.py  perceptual_losses_test.py # differentiable HASQI / HASPI losses
  clinical.py   clinical_test.py         # Bisgaard / ISO 7029 profiles & fitting
  arcfac_metrics.py  arcfac_metrics_test.py      # SI-SDR, STOI, log-STFT, NAP/SAI RT
  tpu.py        tpu_test.py              # single/multi-host Mesh('data') & prefetch
  scripts/                               # CLI training, inversion & benchmark tools
```

## Rules and exceptions

- **Arrays and JAX (`jnp` vs `np`)**: Use `jax.numpy` (`jnp`) in
  differentiable, `@jax.jit`, `jax.vmap`, and `jax.lax.scan` paths
  (`inversion.py`, `neural.py`, `nap_inverter.py`, `sai_inverter.py`,
  `vocoder_losses.py`, `perceptual_losses.py`, `tpu.py`), keeping device
  tensors strictly `float32`/`complex64` with static shapes across CPU/GPU/TPU
  `Mesh('data')`. Use `numpy` (`np`) for host file I/O, `.npz`/`.wav`
  serialization, grain dataset iterators, and reference metric evaluation
  (`arcfac_metrics.py`, `clinical.py`).
- **Large binaries and datasets**: Keep committed binaries under `1 MB`
  (documentation figures and small test fixtures only; unit tests synthesize
  waveforms in temporary directories). Model checkpoints (`.npz`), speech
  corpora, listener pools, and rendered WAVs stay outside git and are supplied
  via CLI flags (`--data_dir`, `--output_dir`, `--checkpoint`, `--pool`). Never
  commit participant recordings or audiograms.
- **Merge strategy**: Squash-merge pull requests into `main` with Conventional
  Commit titles (`<= 72` characters).

### Standing exceptions to `.afhi/python.md`

1. **Script paths**: Entry-point scripts live directly under
   `src/carfac_inversion/scripts/<tool>.py` with colocated `<tool>_test.py`
   rather than `scripts/<tool>/main_<tool>.py`, preserving existing
   `-m carfac_inversion.scripts.<tool>` CLI invocations.
2. **Optional perceptual dependency**: `perceptual_losses.py` and Phase 5
   scripts resolve `jax_haspi_hasqi` dynamically at runtime so core CARFAC
   NAP/SAI inversion works without optional perceptual-metric wheels installed.
3. **Existing lint/import baselines**: `[tool.ruff.lint]` ignores `PLW0603`
   and `PLW0120` in `pyproject.toml`, `.github/workflows/python-app.yml` runs
   `uv sync --dev`, and existing modules retain established Flax/typing imports
   (`import flax.linen as nn`, `from typing import ...`); new code imports
   modules rather than names.
