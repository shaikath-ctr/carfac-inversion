# carfac-inversion

Audio reconstruction and hearing-loss auralization from CARFAC Neural Activity
Patterns (NAPs) back to waveforms in JAX.

Built on top of [`hp-acoustic`](https://github.com/Australian-Future-Hearing-Initiative/hp-acoustic)
and [`carfac`](https://github.com/google/carfac).

## Overview

CARFAC transforms waveforms through a cascaded asymmetric resonator filterbank
(`car_step`), asymmetric hair-cell rectification (`ihc_detect`), two-capacitor
adaptation (`ihc_step`), an 80 µs low-pass filter (~2 kHz cutoff), and auditory
nerve synapse modeling (`syn_step`). Below ~2–4 kHz, NAPs preserve fine temporal
phase-locking; above ~4 kHz, carrier phase is attenuated into positive temporal
envelopes.

`carfac-inversion` solves the inverse `NAP -> audio` problem in up to three stages:

1. **Physics-Based CARFAC DSP Warm Start (`dsp_warm_start`)**:
   - Measures open-loop CARFAC basilar-membrane impulse responses through any
     configured conductive and middle-ear front-end filters at an AGC-damped
     operating Q (`ohc_scale`).
   - Two-pass apical recovery: Pass 1 inverts channels at or above
     `min_warm_start_cf_hz` (`250 Hz`) so quadratic `ohc_nlf` combination-tone
     distortion is excluded from initial phase recovery; Pass 2 subtracts the
     orthogonalized Pass-1 closed-loop CARFAC response to recover unexplained
     `< 250 Hz` low-frequency drive without amplifying combination-tone
     artifacts.
   - Compensates discrete-time IHC + synapse `lpf_coeff` phase lag and rolloff
     below `hf_crossover_hz` (`4200 Hz`), extracts high-pass-cleaned envelopes
     modulated onto bandlimited noise above `hf_crossover_hz`, and calibrates
     per-channel RMS SPL plus 1-shot spectral tilt equalization.
2. **Optional Dual-Domain Neural Residual Refiner (`neural.py`, Phase 3)**:
   - When `InversionConfig.neural_checkpoint` is set, runs a lightweight
     `1.3M`-parameter Flax ConvNeXt STFT + sample-rate 1D U-Net refiner
     (`NapResidualRefiner`) conditioned on canonical log-CF (`40–7200 Hz`)
     target NAPs, warm-start NAP residuals, and physical SPL statistics.
   - Recovers high-SPL (`>= 80 dB SPL`) AGC compression and synthesizes
     `>4 kHz` harmonic/fricative fine structure in a single feedforward pass
     (`~15 ms/s` neural forward, `~3 s/clip` total at `0+0` vs `~111 s` L-BFGS),
     optionally followed by a brief L-BFGS polish (e.g. `adam_steps=0, lbfgs_steps=15`).
3. **Scale-Invariant JAX Optimization (`invert_nap`)**:
   - Projects iterates onto the audible band `[cutoff_low_hz, cutoff_high_hz]`
     (default low edge `35 Hz` with an `18 Hz` cosine taper) in rFFT space to
     preserve speech fundamentals while rejecting sub-audible DC drift.
   - Optimizes via cosine-decayed Adam (`adam_steps=45`) followed by
     exact-secant RMS-normalized L-BFGS (`lbfgs_steps=135`,
     `lbfgs_memory_size=25`) in two modes:
     - `mode="exact"`: Sample-level NAP MSE plus an HF-channel-normalized
       `>3.5 kHz` envelope anchor (`hf_env_weight=0.008`) for same-cochlea
       round-trip reconstruction.
     - `mode="perceptual"`: Blended global and channel-normalized multi-scale
       NAP envelopes (2 ms and 8 ms), Stabilized Auditory Image (SAI) matching,
       and a sample-level `perceptual_nap_weight` anchor (default `0.30`) for
       cross-cochlea hearing-loss auralization (inverting an impaired cochlea's
       NAP through a healthy cochlea even when OHC loss alters traveling-wave
       group delays).
   - Automatically chunks utterances longer than `chunk_seconds` (default
     `1.5 s`) with `chunk_overlap_s` (`0.128 s`) overlap-add blending and frozen
     preceding-audio pre-roll (`chunk_preroll_s`, `0.064 s`) via module-level
     JIT kernels so XLA compiles once and runtime scales linearly with duration.

## Installation

```bash
uv sync --dev
```

Or editable install into an existing environment with `hp-acoustic` and
`carfac`:

```bash
pip install --no-deps -e .
```

## Quickstart

```python
import jax.numpy as jnp
import numpy as np
from hp_acoustic.models import carfac
from carfac_inversion import InversionConfig
from carfac_inversion import invert_nap

healthy = carfac.build(fs=16000)
impaired = carfac.build(fs=16000, ohc_health=0.5)

# Same-cochlea round-trip reconstruction
nap_healthy = np.asarray(healthy(jnp.asarray(audio[:, None]))[:, :, 0])
roundtrip = invert_nap(nap_healthy, healthy)

# Cross-cochlea hearing-loss auralization
nap_impaired = np.asarray(impaired(jnp.asarray(audio[:, None]))[:, :, 0])
auralized = invert_nap(
  nap_impaired,
  healthy,
  InversionConfig(mode="perceptual"),
)
```

### CLI

Round-trip WAV reconstruction:

```bash
python -m carfac_inversion.scripts.invert_nap \
  --input_wav=input.wav \
  --output_wav=roundtrip.wav \
  --level_db_spl=65.0
```

Cross-cochlea hearing-loss auralization from OHC impairment or a fitted `.npz`
profile:

```bash
python -m carfac_inversion.scripts.invert_nap \
  --input_wav=input.wav \
  --output_wav=auralized.wav \
  --source_ohc_health=0.5 \
  --level_db_spl=65.0
```

Diagnostic benchmark suite across sample rates (`16/32 kHz`), levels, SNRs,
step budgets, and cross-cochlea auralization (supports raw WAV/FLAC directories,
`ProcessedClips` shards, or optional LibriSpeech `dev-clean` preparation):

```bash
python -m carfac_inversion.scripts.benchmark_suite \
  --input_dir=/path/to/wavs_or_shards \
  --output_dir=tmp/carfac_inversion_benchmark \
  --max_clips=3
```

Fast single-pass or hybrid Phase 3 neural reconstruction & training:

```bash
# Train 2-stage (spectral/waveform + differentiable CARFAC cycle) refiner:
python -m carfac_inversion.scripts.train_neural_refiner \
  --train_cache=/path/to/train_cache.npz \
  --val_cache=/path/to/val_cache.npz \
  --output_checkpoint=tmp/phase3_refiner.npz

# Invert in ~3 s/clip (0+0 feedforward) or ~14 s/clip (0+15 L-BFGS polish):
python -m carfac_inversion.scripts.invert_nap \
  --input_wav=input.wav \
  --output_wav=roundtrip_p3.wav \
  --level_db_spl=80.0 \
  --neural_checkpoint=tmp/phase3_refiner.npz \
  --adam_steps=0 --lbfgs_steps=15
```

## Evaluation

See [docs/evaluation_phases.md](docs/evaluation_phases.md) for the complete
phased ablation and benchmark log (Phase 0 initial DSP + optimization,
Phase 1 OLA chunking & real-speech baseline, Phase 2 apical/L-BFGS/HF/perceptual
refinements, and Phase 3 neural residual refiner + LibriSpeech held-out evaluation).

## Development

```bash
ruff format --check .
ruff check .
pyrefly check
pytest -v
```

## References

- Lyon, R. F. (2011). *Cascades of two-pole–two-zero asymmetric resonators are
  good models of peripheral auditory function*. JASA 130(6), 3893–3904.
- [CARFAC](https://github.com/google/carfac)
- [hp-acoustic](https://github.com/Australian-Future-Hearing-Initiative/hp-acoustic)
