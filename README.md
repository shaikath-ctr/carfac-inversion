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

Headline metrics on **4 held-out LibriSpeech speakers** (`[7850, 7976, 8297, 8842]`, `16 kHz`, `1.0 s` windows; 100% unseen during Phase 3 training):

| Condition | Method | Runtime | SI-SDR | STOI | HASPI | HASQI / nl | 4.5–7.5 kHz SNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `50 dB` clean | Phase 2 Warm Start (`0+0`) | `2.08 s` | `3.15 dB` | `0.8337` | `0.9997` | `0.508 / 0.624` | `0.0 dB` |
| `50 dB` clean | Phase 2 L-BFGS (`20+60`) | `53.37 s` | `16.45 dB` | `0.9784` | `0.9997` | `0.733 / 0.768` | `-4.8 dB` |
| `50 dB` clean | **Phase 3 Neural (`0+0`)** | **`3.07 s`** | `19.78 dB` | `0.9907` | `0.9999` | `0.791 / 0.834` | `-1.5 dB` |
| `50 dB` clean | **Phase 3 Hybrid (`0+15`)** | `15.09 s` | **`20.03 dB`** | **`0.9961`** | **`1.0000`** | **`0.842 / 0.881`** | `-1.5 dB` |
| `65 dB` clean | Phase 2 Warm Start (`0+0`) | `2.17 s` | `-3.49 dB` | `0.8033` | `0.9964` | `0.360 / 0.465` | `0.0 dB` |
| `65 dB` clean | Phase 2 L-BFGS (`20+60`) | `51.97 s` | `8.08 dB` | `0.9255` | `0.9967` | `0.483 / 0.527` | `-4.4 dB` |
| `65 dB` clean | **Phase 3 Neural (`0+0`)** | **`3.60 s`** | `17.38 dB` | `0.9865` | `0.9996` | `0.714 / 0.754` | `-1.0 dB` |
| `65 dB` clean | **Phase 3 Hybrid (`0+15`)** | `14.24 s` | **`18.37 dB`** | **`0.9918`** | **`0.9999`** | **`0.779 / 0.817`** | **`-0.9 dB`** |
| `80 dB` clean | Phase 2 Warm Start (`0+0`) | `2.14 s` | `-4.80 dB` | `0.6837` | `0.5757` | `0.127 / 0.162` | `-1.4 dB` |
| `80 dB` clean | Phase 2 L-BFGS (`20+60`) | `50.98 s` | `0.37 dB` | `0.7993` | `0.8013` | `0.220 / 0.272` | `-4.3 dB` |
| `80 dB` clean | **Phase 3 Neural (`0+0`)** | **`3.17 s`** | `12.83 dB` | `0.9464` | `0.9993` | `0.551 / 0.584` | `-1.3 dB` |
| `80 dB` clean | **Phase 3 Hybrid (`0+15`)** | `13.36 s` | **`13.07 dB`** | **`0.9536`** | **`0.9996`** | **`0.609 / 0.644`** | **`-1.2 dB`** |
| `65 dB`, 10 dB SNR | Phase 2 L-BFGS (`20+60`) | `51.70 s` | `7.17 dB` | `0.9296` | `0.9999` | `0.605 / 0.666` | `-0.1 dB` |
| `65 dB`, 10 dB SNR | **Phase 3 Neural (`0+0`)** | **`3.06 s`** | `10.15 dB` | `0.9570` | `0.9999` | `0.620 / 0.651` | `-1.6 dB` |
| `65 dB`, 10 dB SNR | **Phase 3 Hybrid (`0+15`)** | `14.21 s` | **`10.51 dB`** | **`0.9761`** | **`0.9999`** | **`0.647 / 0.675`** | `-1.6 dB` |

Highlights on canonical out-of-domain `carfac/test_data` (`binaural_test.wav` & `long_test.wav`):
- **Round-trip (`mode="exact"`, Phase 2 `45+135` at `~111 s` -> Phase 3 Hybrid `0+15` at `~14 s`)**: `binaural_test` reached `SI-SDR 15.85 dB / STOI 0.9850` at `50 dB`, `STOI 0.9180 -> 0.9786` / `HASQI 0.479 -> 0.512` at `65 dB`, and `HASPI 0.8863 -> 0.9945` / `STOI 0.8219 -> 0.8857` / `HASQI 0.232 -> 0.284` at `80 dB`; `long_test` improved from `9.56 -> 13.57 dB` at `65 dB` and `7.70 -> 11.65 dB` at `80 dB`.
- **Cross-cochlea auralization (`mode="perceptual"`, Phase 2 `perc030`)**: On `sloping` hearing loss (`65 dB`, 10 dB noisy), achieved `NAP corr 0.8721 / 0.9598`, `Env corr 0.9545 / 0.9874`, `SAI corr 0.9741 / 0.9908`, `HASPI 0.9938 / 0.9522`, and `STOI 0.9233 / 0.7282` (`binaural_test` / `long_test`). *(Note: the Phase 3 checkpoint targets same-cochlea `mode="exact"` reconstruction; cross-cochlea `mode="perceptual"` should use Phase 2 `perc030`.)*

See [docs/evaluation_phases.md](docs/evaluation_phases.md) for the full Phase 0–3 ablation log, accepted/rejected hypotheses, training details, and per-clip tables (`28 / 28` pytest tests passing).

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
