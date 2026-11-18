# carfac-inversion

Audio reconstruction and hearing-loss auralization from CARFAC Neural Activity
Patterns (NAPs) back to waveforms in JAX.

Built on top of [`hp-acoustic`](https://github.com/Australian-Future-Hearing-Initiative/hp-acoustic)
and [`carfac`](https://github.com/google/carfac).

## Overview

CARFAC v2 ([Lyon, 2011](#references); [Lyon, 2017](#references);
[Lyon et al., 2024](#references)) transforms waveforms through a cascaded
asymmetric resonator filterbank (`car_step`), asymmetric hair-cell
rectification (`ihc_detect`), two-capacitor adaptation (`ihc_step`), an 80 µs
low-pass filter (~2 kHz cutoff), and auditory nerve synapse modeling (`syn_step`)
using the canonical CARFAC acoustic reference (`0 dBFS` sine peak = `104 dB SPL`,
i.e., `RMS = 1.0 ≈ 107.01 dB SPL` via `sii.CARFAC_RMS_ONE_DB_SPL`). Below
~2–4 kHz, NAPs preserve fine temporal phase-locking; above ~4 kHz, carrier phase
is attenuated into positive temporal envelopes.

`carfac-inversion` solves the inverse `NAP -> audio` problem in up to three stages:

1. **Physics-Based CARFAC DSP Warm Start (`dsp_warm_start`)**:
   - Measures open-loop CARFAC basilar-membrane impulse responses (duration-capped
     at `~0.15 s` up to `8192` samples to accommodate high-Q / `48 kHz` tails)
     through any configured conductive and middle-ear front-end filters at a
     per-ear AGC-damped operating Q (`ohc_scale`), extending scalar gain search
     brackets down to `1e-7` for retuned high-gain CARFAC configurations.
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
2. **Optional Dual-Domain Neural Residual Refiner (`neural.py`, [`train_neural_refiner.py`](src/carfac_inversion/scripts/train_neural_refiner.py))**:
   - When `InversionConfig.neural_checkpoint` is set, runs a `1.3M`-parameter
     (`1,297,790` params) Flax ConvNeXt STFT + sample-rate 1D U-Net refiner
     (`NapResidualRefiner`) conditioned on canonical log-CF (`40–7200 Hz`)
     target NAPs, (`target_nap - ws_nap`) residual statistics, and physical
     SPL (`log10(ws_rms)`).
   - Zero-initialized synthesis heads predict a global SPL multiplier
     (`exp(delta_g)`), complex STFT ratio mask + additive complex STFT
     residual (`n_fft=256, hop=64`), and a sample-aligned 1D U-Net tail
     (`unet_base_dim=24`), initialized at exact identity (`max |out - x_ws| < 1e-6`,
     exact `0.0` on silence).
   - Recovers high-SPL (`>= 80 dB SPL`) AGC compression and synthesizes
     `>4 kHz` harmonic/fricative fine structure in a single feedforward pass
     (`~15 ms/s` neural forward, `~3 s/clip` total at `0+0` vs `~111 s` L-BFGS),
     optionally followed by a brief L-BFGS polish (`0+15`).
   - Trained in two stages on speaker-disjoint LibriSpeech `dev-clean` (`32`
     train speakers / `766` windows spanning `43–86 dB SPL` + clean/noisy;
     `4` val speakers / `64` windows; `4` test speakers): Stage 1 (`1400` steps)
     optimizes re-balanced multi-resolution STFT (`2048/1024/512/128`),
     `3.8–7.8 kHz` HF log-magnitude MSE, RMS-normalized L1, `log_rms`,
     `-SI-SDR`, and `<20 Hz` drift (`val_sisdr -3.41 -> +12.06 dB`); Stage 2
     (`350` steps) adds differentiable CARFAC NAP + `>3.5 kHz` envelope cycle
     consistency (`carfac_nap` val MSE `0.0950 -> 0.0711`).
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

Fast single-pass or hybrid neural reconstruction & training:

```bash
# Train 2-stage (spectral/waveform + differentiable CARFAC cycle) refiner:
python -m carfac_inversion.scripts.train_neural_refiner \
  --train_cache=/path/to/train_cache.npz \
  --val_cache=/path/to/val_cache.npz \
  --output_checkpoint=tmp/phase3_refiner.npz

# Invert in ~3 s/clip (0+0 feedforward) or ~14 s/clip (0+15 L-BFGS polish):
# (Download pre-trained 4.6 MiB `phase3_refiner.npz` from a GitHub Release asset
# when published, or produce it via `train_neural_refiner.py` above.)
python -m carfac_inversion.scripts.invert_nap \
  --input_wav=input.wav \
  --output_wav=roundtrip_p3.wav \
  --level_db_spl=80.0 \
  --neural_checkpoint=tmp/phase3_refiner.npz \
  --adam_steps=0 --lbfgs_steps=15
```

## Approach & Phased Evaluation Methodology

Every architectural improvement was driven by real-speech diagnostics in
[`benchmark_suite.py`](src/carfac_inversion/scripts/benchmark_suite.py) and
recorded with accepted/rejected ablations in
[`docs/evaluation_phases.md`](docs/evaluation_phases.md):

- **Phase 0 (`95e7a51`)**: Core open-loop CARFAC DSP warm start + rFFT-masked
  Adam/L-BFGS passed **17 / 17** synthetic unit tests, surfacing two gaps:
  unchunked XLA re-compilation on variable utterance lengths and no real-speech
  multi-SPL benchmark.
- **Phase 1 (`08d7700`, `a7b4025`)**: Added fixed-window OLA chunking
  (**22 / 22** tests) and built `benchmark_suite.py`. Real-speech evaluation
  exposed 4 concrete bottlenecks: `-2.5` to `-6.6 dB` SI-SDR on low-F0 speech
  (`long_test.wav`), suboptimal `40+40` secants, `80 dB` AGC compression +
  `> 4 kHz` sag, and `< 250 Hz` energy swamping `1–4 kHz` consonants in
  `mode="perceptual"`.
- **Phase 2a (`9970fa4`)**: Rejected naive unconditional `< 250 Hz` inversion
  (`binaural_test` regressed across all SPLs) and inverse-threshold frequency
  preconditioning `W_pre(f)` (`binaural_test` dropped `-4` to `-6 dB`). Even raw
  Pass-2 residual subtraction failed our `32 kHz` unit test (`9.68 dB < 10.0 dB`)
  until diagonal projection diagnosed `99.9%` of `< 250 Hz` NAP energy on that
  synthetic clip was quadratic combination-tone leakage; orthogonalizing Pass-1
  leakage fixed `32 kHz` (**23 / 23** tests) and lifted `long_test` by
  **`+12.6` to `+14.3 dB` SI-SDR**.
- **Phase 2b (`114cfc3`)**: Fixed an ~8x channel-averaging dilution bug in
  `hf_err`, switched `mode="perceptual"` to blended channel-normalized envelopes
  + `perceptual_nap_weight=0.30` (after testing and rejecting Pass-2 HF noise
  injection, all-channel `log1p` anchors, L-BFGS curvature resets,
  `adam_lr_scale` boosts, and `SaiPndLoss`).
- **Phase 3 (`70d15b8`)**: Trained and benchmarked `NapResidualRefiner` against
  L-BFGS optimization on both `carfac/test_data` and 4 completely unseen
  LibriSpeech `dev-clean` speakers (**28 / 28** tests), re-balancing
  `hp_acoustic`'s default `sqrt(win_len/2)` STFT loss scaling after an initial
  smoke check showed `stft=112.7` drowning `SI-SDR` and CARFAC cycle consistency.
- **Calibration Alignment & Retuned CARFAC Robustness (`f571baa`, `9da7cbf`)**:
  Aligned SPL scaling in `benchmark_suite.py` and `invert_nap.py` with
  `sii.CARFAC_RMS_ONE_DB_SPL` (`RMS = 1.0 ≈ 107.01 dB SPL`, `0 dBFS` sine peak
  `= 104 dB SPL`), added dynamic gain bounds + `1e-7` Adam/L-BFGS RMS floors
  for retuned high-gain CARFAC presets (reducing Option C `20 dB` warm-start
  gain error from `5.3x` to `1.01x`), scaled open-loop BM `ir_len` up to `8192`
  at `48 kHz`, and moved `ohc_scale` per ear after ablations showed per-channel
  `ohc_scale` hurt Phase 3 (`-1.7` to `-3.7 dB` SI-SDR). (**31 / 31** tests.)

## What the Metrics Mean

| Metric | Domain & Range | What It Measures & How to Interpret It |
| :--- | :--- | :--- |
| **`SI-SDR`** | Waveform (dB) | **Scale-Invariant Signal-to-Distortion Ratio**: Sample-level carrier phase and waveform accuracy up to a global scalar. High `SI-SDR` (`> 10–20 dB`) indicates accurate `< 2–4 kHz` phase locking. *(In cross-cochlea `mode="perceptual"`, output reflects a healthy ear at impaired cochlear firing levels with different traveling-wave delays, so negative `SI-SDR` is expected and uninformative there.)* |
| **`SNR`** | Waveform (dB) | **Signal-to-Noise Ratio**: Penalizes global SPL/volume mismatches directly (`ws/ref != 1.0`), making it essential at `80 dB SPL` where linear warm starts under-predict level due to OHC AGC compression. |
| **`STOI`** | Perceptual (`0–1`) | **Short-Time Objective Intelligibility**: Correlates temporal amplitude envelopes across ~1/3-octave critical bands (`> 0.90` = high intelligibility); robust to `> 4 kHz` random-carrier phase ambiguity. |
| **`HASPI v2`** | Auditory (`0–1`) | **Hearing-Aid Speech Perception Index**: Auditory-nerve intelligibility model calibrated to an audiogram (`1.0` = ceiling intelligibility). |
| **`HASQI v2` (`nl` / `lin`)** | Auditory (`0–1`) | **Hearing-Aid Speech Quality Index (`HASQI = nl * lin`)**: Separates **`HASQI_lin`** (long-term spectral tilt) from **`HASQI_nl`** (cepstral spectral-envelope smoothness + temporal modulation); sensitive to `4.2–7.5 kHz` fricative naturalness and tonal harmonics vs. noise. |
| **`NAP corr` / `Env corr` / `SAI corr`** | Cochlear (`-1` to `1`) | **Forward Manifold Consistency**: Passes reconstructed audio back through the target cochlea (`carfac`) to verify exact neural firing alignment (`NAP corr`), `200 Hz`-smoothed temporal envelopes (`Env corr`, phase-tolerant), and Stabilized Auditory Images (`SAI corr`, pitch/timbre periodicity)—the primary ground truth for cross-cochlea auralization. |
| **Band SNRs (`60–300` … `4k5–7k5 Hz`)** | Spectral-Time (dB) | **Frequency Diagnostics**: `60–300 Hz` catches missing `< 250 Hz` pitch fundamentals (`F0`); `300–2200 Hz` tracks vowel formants & `80 dB` compression recovery; `4500–7500 Hz` isolates `> 4 kHz` synthesis above CARFAC's ~2 kHz IHC LPF. |

## Evaluation Results

Headline metrics on **4 held-out LibriSpeech speakers** (`[7850, 7976, 8297, 8842]`, `16 kHz`, `1.0 s` windows; 100% unseen during neural training, calibrated under `sii.CARFAC_RMS_ONE_DB_SPL ≈ 107.01 dB SPL`):

| Condition | Method | Runtime | SI-SDR | STOI | HASPI | HASQI / nl | 4.5–7.5 kHz SNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `50 dB` clean | DSP Warm Start (`0+0`) | `3.36 s` | `4.10 dB` | `0.8394` | `0.9998` | `0.538 / 0.644` | `0.0 dB` |
| `50 dB` clean | DSP + L-BFGS (`20+60`) | `53.79 s` | `16.34 dB` | `0.9822` | `0.9999` | `0.789 / 0.822` | `-4.8 dB` |
| `50 dB` clean | **DSP + Neural (`0+0`)** | **`3.08 s`** | `19.39 dB` | `0.9881` | `0.9999` | `0.815 / 0.858` | `-1.6 dB` |
| `50 dB` clean | **DSP + Neural + L-BFGS (`0+15`)** | `15.89 s` | **`19.44 dB`** | **`0.9943`** | **`1.0000`** | **`0.860 / 0.898`** | `-1.6 dB` |
| `65 dB` clean | DSP Warm Start (`0+0`) | `2.17 s` | `-2.64 dB` | `0.8119` | `0.9983` | `0.404 / 0.520` | `0.0 dB` |
| `65 dB` clean | DSP + L-BFGS (`20+60`) | `51.53 s` | `10.69 dB` | `0.9346` | `0.9988` | `0.535 / 0.569` | `-4.7 dB` |
| `65 dB` clean | **DSP + Neural (`0+0`)** | **`3.21 s`** | `17.80 dB` | `0.9887` | `0.9996` | `0.721 / 0.761` | `-1.0 dB` |
| `65 dB` clean | **DSP + Neural + L-BFGS (`0+15`)** | `14.68 s` | **`18.88 dB`** | **`0.9935`** | **`0.9999`** | **`0.785 / 0.822`** | **`-1.0 dB`** |
| `80 dB` clean | DSP Warm Start (`0+0`) | `2.30 s` | `-4.58 dB` | `0.7132` | `0.7036` | `0.168 / 0.215` | `-0.8 dB` |
| `80 dB` clean | DSP + L-BFGS (`20+60`) | `51.22 s` | `1.72 dB` | `0.8294` | `0.8682` | `0.273 / 0.331` | `-3.6 dB` |
| `80 dB` clean | **DSP + Neural (`0+0`)** | **`3.33 s`** | `14.08 dB` | `0.9564` | `0.9995` | `0.598 / 0.632` | `-1.2 dB` |
| `80 dB` clean | **DSP + Neural + L-BFGS (`0+15`)** | `13.59 s` | **`14.42 dB`** | **`0.9629`** | **`0.9997`** | **`0.662 / 0.697`** | **`-1.1 dB`** |
| `65 dB`, 10 dB SNR | DSP + L-BFGS (`20+60`) | `51.72 s` | `8.97 dB` | `0.9450` | `1.0000` | `0.644 / 0.695` | `-0.1 dB` |
| `65 dB`, 10 dB SNR | **DSP + Neural (`0+0`)** | **`3.28 s`** | `10.33 dB` | `0.9675` | `0.9999` | `0.695 / 0.729` | `-1.7 dB` |
| `65 dB`, 10 dB SNR | **DSP + Neural + L-BFGS (`0+15`)** | `14.58 s` | **`10.73 dB`** | **`0.9844`** | **`1.0000`** | **`0.725 / 0.755`** | `-1.7 dB` |

Highlights on canonical out-of-domain `carfac/test_data` (`binaural_test.wav` & `long_test.wav`) & scope boundaries:
- **Round-trip (`mode="exact"`, DSP + `45+135` L-BFGS @ `~110 s` -> DSP + Neural + `0+15` L-BFGS @ `~14–26 s`)**: `binaural_test` reached `SI-SDR 16.80 dB / STOI 0.9864` at `50 dB`, `SI-SDR 9.26 -> 9.44 dB` / `STOI 0.9204 -> 0.9843` at `65 dB`, and `STOI 0.8471 -> 0.9049` / `HASPI 0.9989 -> 0.9986` / `SI-SDR 1.38 -> 1.06 dB` at `80 dB` (`ws/ref 0.247 -> p3/ref 0.860`); `long_test` improved from `10.47 -> 13.59 dB` (`13.91 dB` at `0+0`) at `65 dB` and `8.53 -> 12.11 dB` at `80 dB`.
- **Cross-cochlea auralization (`mode="perceptual"`, `perc030`)**: On `sloping` hearing loss (`65 dB`, 10 dB noisy), achieved `NAP corr 0.8724 / 0.9622`, `Env corr 0.9571 / 0.9874`, `SAI corr 0.9710 / 0.9901`, `HASPI 0.9841 / 0.8904`, and `STOI 0.9351 / 0.7441` (`binaural_test` / `long_test`). *(Scope caveat: the trained neural checkpoint targets same-cochlea `mode="exact"` reconstruction; passing impaired NAPs pulls output back toward healthy speech rather than staying on the impaired cochlear manifold (`nap_corr 0.33–0.49` vs. `perc030` `0.67–0.96`), so cross-cochlea auralization should use `mode="perceptual"` without the `exact`-trained checkpoint until a dedicated cross-cochlea checkpoint is trained.)*

See [`docs/evaluation_phases.md`](docs/evaluation_phases.md) for the complete Phase 0–3 ablation log, accepted/rejected hypotheses, training details, and per-clip tables (`31 / 31` pytest tests passing across `inversion_test.py` (17), `neural_test.py` (5), `benchmark_suite_test.py` (5), and `invert_nap_test.py` (4)).

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
- Lyon, R. F. (2017). *Human and Machine Hearing: Extracting Meaning from Sound*.
  Cambridge University Press.
- Lyon, R. F., Schonberger, R., Slaney, M., Velimirović, M., & Yu, H. (2024).
  *The CARFAC v2 Cochlear Model in Matlab, NumPy, and JAX*.
  [arXiv:2404.17490](https://arxiv.org/abs/2404.17490).
- [CARFAC](https://github.com/google/carfac)
- [hp-acoustic](https://github.com/Australian-Future-Hearing-Initiative/hp-acoustic)
