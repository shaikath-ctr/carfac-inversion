# CARFAC NAP-to-Audio Inversion — Phased Evaluation & Ablation Log

This document records the engineering progression, hypotheses tested (both accepted and falsified), and quantitative benchmarks across **Phase 0**, **Phase 1**, **Phase 2**, and **Phase 3** of [`carfac-inversion`](../README.md), providing an auditable reference for downstream PR review.

---

## Evaluation Setup & Reproducibility

- **Test suite**: [`inversion_test.py`](../src/carfac_inversion/inversion_test.py), [`neural_test.py`](../src/carfac_inversion/neural_test.py), [`invert_nap_test.py`](../src/carfac_inversion/scripts/invert_nap_test.py), [`benchmark_suite_test.py`](../src/carfac_inversion/scripts/benchmark_suite_test.py) (`31` unit & perceptual tests across `16 / 32 kHz` CARFAC, `24 kHz` Gammatone, generic `CochleaModel`, binaural 2-ear, cross-cochlea auralization, OLA chunking parity, apical `< 250 Hz` tones, Phase 3 neural residual refiner, `107.01 dB` RMS=1 calibration, retuned Option-C `20 dB` low-SPL tracking, and CLI paths).
- **Real-speech diagnostic benchmark**: [`benchmark_suite.py`](../src/carfac_inversion/scripts/benchmark_suite.py) run on `1.0 s` crops of real speech from `carfac/test_data` (`binaural_test.wav` — broadband speech with `/s/` fricatives up to `7.5 kHz`; `long_test.wav` — male speech dominated by low-frequency fundamentals in `60–250 Hz`):
  - `sample_rate_hz = 16000`, `clip_seconds = 1.0`, `chunk_seconds = 0.5` (`0.128 s` overlap, `0.064 s` pre-roll).
  - Clean round-trip levels: `50 / 65 / 80 dB SPL`, plus `10 dB` noisy round-trip at `65 dB SPL`.
  - Cross-cochlea auralization (`65 dB SPL`, `SNR = 10 dB` noisy and `SHACA`-enhanced): uniform `ohc_health = 0.5` (`HALF_OHC_HEARING_LOSS_DB`) and synthetic sloping high-frequency loss (`0.95 -> 0.20` across `250–6000 Hz`).
  - Metrics: `SI-SDR`, `SNR`, `STOI`, `HASPI v2`, `HASQI v2` (`nl` + `lin`), `NAP corr`, `200 Hz` envelope corr, `SAI corr`, and per-band SNRs (`60–300 Hz`, `300–1000 Hz`, `1000–2200 Hz`, `2200–4500 Hz`, `4500–7500 Hz`).

---

## Phase 0: Core DSP Warm Start & Scale-Invariant Optimization (`95e7a51`)

### Architecture
- **Stage 1 (`dsp_warm_start`)**: Measures open-loop basilar-membrane impulse responses through configured conductive and middle-ear filters at an AGC-damped operating Q (`ohc_scale`), inverts channels `>= 250 Hz` (`min_warm_start_cf_hz`) with discrete-time IHC + synapse `lpf_coeff` compensation below `4200 Hz`, extracts HF envelopes above `4200 Hz` modulated onto bandlimited noise, and performs scalar gain calibration + 1-shot spectral tilt EQ.
- **Stage 2 (`invert_nap`)**: rFFT projection onto `[60 Hz, min(0.48*fs, 1.15*max_cf)]`, followed by cosine-decayed Adam (`80` steps) and RMS-normalized L-BFGS (`100` steps, `memory_size=15`).
  - `mode="exact"`: Sample-level NAP MSE + `hf_env_weight` (`0.005`) on `>= 3500 Hz`.
  - `mode="perceptual"`: `2 ms` + `8 ms` NAP envelopes (`0.65`), `SAI` MSE (`0.25`), and `perceptual_nap_weight` (`0.10`).

### Initial Verification & Bottlenecks Identified
- **Unit test evolution**: Initial `60+60` step budget failed `test_round_trip_32khz_default_rate` at `nap_corr = 0.9896 < 0.99`; increasing the 32 kHz test budget to `80+80` brought all **17 / 17** initial tests green (and **55 / 55** in `hp-acoustic`).
- **Gaps identified for Phase 1**:
  1. Unchunked full-clip optimization caused XLA re-compilation on every utterance duration and prevented scaling to multi-sentence LibriSpeech clips.
  2. Validation relied solely on synthetic speech-like signals rather than real speech across SPLs, SNRs, frequency bands, and enhancers.

---

## Phase 1: Fixed-Window OLA Chunking & Real-Speech Benchmark Suite (`08d7700`, `a7b4025`)

### Changes
- Added fixed-window overlap-add (`chunk_seconds = 1.5`, `chunk_overlap_s = 0.128` cosine-squared crossfade, `chunk_preroll_s = 0.064` frozen preceding context) and module-level `@jax.jit` kernels (`_eval_window_loss_jit`, `_adam_window_step_jit`, `_lbfgs_window_step_jit`).
- Built [`benchmark_suite.py`](../src/carfac_inversion/scripts/benchmark_suite.py) supporting raw `.wav`/`.flac` directories, `ProcessedClips` shards, LibriSpeech `dev-clean` preparation, optional `SHACA`/`Viking`/`SEANet` enhancement, CSV output, WAV pairs, and 4 diagnostic PNG charts.

### Phase 1 Validation Results
- **Tests**: **22 / 22** passed (`ruff format --check .` & `ruff check .` clean).
- **XLA kernel reuse**: Two sequential `1.2 s` calls with `chunk_seconds=0.6` measured `31.34 s` on call 1 (compile + run) vs. `21.29 s` on call 2 (cached).
- **OLA parity (`test_ola_chunked_matches_full_clip_quality`, `70+70` steps on `1.0 s` speech-like clip)**:
  - Full-clip unchunked: `SI-SDR = 9.81 dB`, `STOI = 0.8377`, `NAP corr = 0.9983`.
  - OLA (`0.5 s` windows): `SI-SDR = 10.24 dB`, `STOI = 0.8523`, `NAP corr = 0.9985`.
- **Phase 1 Real-Speech Baseline (`40+40` steps, `chunk_seconds=0.5`, `1.0 s` clips)** — revealed four concrete bottlenecks for Phase 2:
  1. **Severe low-frequency (`< 250 Hz`) loss on `long_test.wav`**: `SI-SDR` was `-2.53 dB` (`50 dB`), `-3.06 dB` (`65 dB`), and `-6.65 dB` (`80 dB`), with `60–300 Hz` band SNR `0.17–0.60 dB` because `min_warm_start_cf_hz = 250 Hz` zeroed sub-250 Hz speech fundamentals and `cutoff_low_hz = 60 Hz` (`+35 Hz` taper) attenuated `< 95 Hz`.
  2. **Sub-optimal step budget & L-BFGS secants**: `40+40` left quality on the table vs. L-BFGS-heavy budgets (`15+65` / `20+60`), and post-step `_project_wave` inside `_lbfgs_window_step_jit` broke exact secant pairs.
  3. **High-SPL (`80 dB SPL`) compression & `> 4 kHz` sag**: `binaural_test` at `80 dB` scored `SI-SDR = -2.60 dB`, `HASQI = 0.133`; `4.5–7.5 kHz` band SNR was `-2.29 dB` (`binaural_test`) and `-0.84` to `-2.03 dB` (`long_test`).
  4. **Cross-cochlea perceptual drop on `long_test`**: `ohc05_noisy_perc010` dropped to `STOI = 0.5598` (`HASPI = 0.625`) vs. `exact` `STOI = 0.7568` (`HASPI = 0.712`), because global `env_var` normalization let `< 250 Hz` energy swamp `1–4 kHz` consonant bands under OHC loss.

---

## Phase 2a: Orthogonalized Apical Recovery, `35 Hz` Mask & Exact-Secant L-BFGS (`9970fa4`)

### Hypotheses & Ablations Tested
1. **Unconditional apical warm start (`min_warm_start_cf_hz = 45 Hz`) — REJECTED**:
   - Lifted `long_test` `50 dB` warm-start `SI-SDR` from `-4.55 -> +6.60 dB`, but regressed `binaural_test` (`50 dB`: `7.76 -> 7.45 dB`; `65 dB`: `2.72 -> 1.60 dB`; `80 dB`: `-2.80 -> -4.59 dB`) by amplifying quadratic `ohc_nlf` combination tones into sub-bass rumble.
2. **Inverse cochlear-sensitivity frequency preconditioning `W_pre(f)` — REJECTED**:
   - Lifted `long_test` (`50 dB` `13.27 dB`, `65 dB` `7.43 dB`, `80 dB` `1.06 dB` at `40+40`), but degraded `binaural_test` across all levels (`50 dB`: `7.76 -> 2.73 dB`, `65 dB`: `2.72 -> -1.39 dB`, `80 dB`: `-2.80 -> -6.44 dB`) by over-amplifying low-frequency noise.
3. **Two-pass closed-loop residual apical recovery (raw vs. orthogonalized) — ACCEPTED WITH ORTHOGONALIZATION**:
   - Raw residual (`nap_c - nap1`) worked at `16 kHz` (`long_test` `+7.48` to `+11.62 dB` SI-SDR, zero `binaural_test` regression), **but failed `test_round_trip_32khz_default_rate` (`9.68 dB < 10.0 dB`)**:
     - Forward diagnostic proved 99.9% (`corr = 0.999`) of the `32 kHz` synthetic speech `< 250 Hz` NAP was quadratic combination-tone leakage (`corr(nap_c, nap1) = +0.795`), yet slight Pass-1 amplitude/phase differences inflated raw `||nap_c - nap1|| / ||nap_c||` to `0.653` (`unexplained_frac = 0.672`), injecting a 2.5x over-amplified, anti-correlated (`corr = -0.642`) phantom `60–250 Hz` wave (`ws SI-SDR: 0.56 -> 0.20 dB`).
   - **Fix**: Project out the non-negative linear Pass-1 component per apical channel (`orth_res = tgt_ap - beta_ap * leak_ap`) and gate both `unexplained_frac` (`(rms_orth_ap / rms_tgt_ap - 0.65) / 0.25`) and `ch_gate` (`(ch_orth_r / ch_tgt_r - 0.55) / 0.35`). Across all 10 diagnostic clips, orthogonalized residuals cleanly separated combination-tone-dominated clips (`orth_r = 0.255–0.609`, `u = 0.00`) from true `< 250 Hz` signals (`long_test` `0.922–0.938`, `75+440 Hz` `1.000`, `u = 1.00`).
4. **Exact-secant L-BFGS + `35 Hz` (`+18 Hz`) low-cut + `45+135` (`memory_size=25`) step budget — ACCEPTED**:
   - Removing post-`apply_updates` `_project_wave` inside `_lbfgs_window_step_jit` and `_optimize_window` preserved exact secant updates (`loss_scaled` already projects inputs).
   - At `32 kHz` (`test_round_trip_32khz_default_rate`), `m25` (`SI-SDR = 10.39 dB`, `loss = 0.00828` at `80+80`; `10.02 dB`, `STOI = 0.9491` at `45+135`) outperformed both `m15` (`10.33 dB` / `9.89 dB`) and Phase 1 `HEAD` (`10.35 dB` / `9.93 dB`).

### Phase 2a Quantitative Gains (`16 kHz`, `1.0 s`, `45+135` vs. Phase 1 `40+40`)
| Clip / Level | Phase 1 SI-SDR -> Phase 2a | Phase 1 STOI -> Phase 2a | Phase 1 HASQI -> Phase 2a | 60–300 Hz SNR (P1 -> P2a) |
| :--- | :---: | :---: | :---: | :---: |
| `binaural_test` 50 dB | `8.65 -> 14.48 dB` | `0.9212 -> 0.9739` | `0.608 -> 0.715` | `-3.7 -> +3.6 dB` |
| `binaural_test` 65 dB | `3.47 -> 7.19 dB` | `0.8834 -> 0.9175` | `0.315 -> 0.474` | `-0.9 -> +0.8 dB` |
| `binaural_test` 80 dB | `-2.60 -> -0.06 dB` | `0.7429 -> 0.8218` | `0.133 -> 0.228` | `+0.9 -> +0.3 dB` (`nap 0.9628 -> 0.9852`) |
| `long_test` 50 dB | `-2.53 -> +11.41 dB` | `0.9841 -> 0.9966` | `0.483 -> 0.706` | `+0.6 -> +12.6 dB` |
| `long_test` 65 dB | `-3.06 -> +9.52 dB` | `0.9464 -> 0.9732` | `0.324 -> 0.569` | `+0.5 -> +9.6 dB` |
| `long_test` 80 dB | `-6.65 -> +7.54 dB` | `0.7676 -> 0.9475` | `0.151 -> 0.525` | `+0.2 -> +7.5 dB` |

---

## Phase 2b: HF (`> 4 kHz`) Naturalness, High-SPL (`80 dB`) Diagnostics & Cross-Cochlea Perceptual Tuning

### Bottleneck Diagnosis on `9970fa4`
- **HF (`> 4 kHz`) & `HASQI nl`**: On `binaural_test` `50 dB` and `long_test` `50 dB`, `bm_sync5` was already `0.9582` and `0.9800` (`HASQI lin = 0.958` / `0.972`), while `cepstral_correlation` (`0.8826` / `0.8608`) limited `HASQI nl` (`0.746` / `0.726`) and `4.2–7.5 kHz` sagged by `-3.0 dB` / `-6.1 dB`. Root causes:
  1. `hf_err = jnp.mean(hf_mask_j * ...)` in `_window_loss_impl` averaged across **all `n_ch` (`65–77`) channels**, diluting `hf_env_weight = 0.005` by ~8x (`~0.0006` effective weight).
  2. `_TAIL_LEAKAGE_HPF_HZ = 1200 Hz` in `dsp_warm_start` suppressed the `< 250 Hz` positive envelope of `>= 4200 Hz` bands (`4.2–7.5 kHz` warm start started at `-27.5 dB` / `-26.1 dB`).
- **High-SPL (`80 dB SPL`)**: On `binaural_test` `80 dB`, smeared/low-crest warm start already matched `target_ch_rms` at `0.22x` reference RMS (`mid_p/t = 1.05`), so `_calibrate_scalar_gain` anchored at `ws/ref = 0.22`, and optimization climbed to `est/ref = 0.58` (`250–2200 Hz` lagged at `-5.5 dB` vs `-0.4 dB` at `4.2–7.5 kHz`).
- **Cross-cochlea (`mode="perceptual"`)**: Under `ohc05`, `long_test` `perc010` dropped `STOI` (`0.6140` vs. `exact` `0.7960`), and `binaural_test` `ohc05` scored `HASPI = 0.4418`, because dividing envelope MSE solely by global `env_var` let low-frequency energy drown out attenuated `1–4 kHz` consonant bands (`-24 dB` warm-start tilt on `long_test ohc05`).

### Hypotheses & Ablations Tested
1. **Pass-2 residual HF warm-start with voiced/noise blending — REJECTED**:
   - Helped `50 dB` `binaural_test` when paired with `hf_w=0.05` (`HASQI = 0.797`, `nl = 0.815`, `4.2–7.5 kHz = -0.0 dB`), **but severely degraded `80 dB` clips** (`long_test 80 dB`: injected `+17.8 dB` excess HF hiss, dropping `SI-SDR` from `7.16 -> 0.95 dB` and `HASQI` from `0.502 -> 0.316`; `binaural_test 80 dB`: `HASQI 0.246 -> 0.167`).
2. **All-channel `log1p` envelope anchor (`12 ms`) — REJECTED**:
   - Lifted `binaural_test` `50 dB` (`HASQI 0.776`), **but degraded `long_test` `50 dB`** (`HASQI 0.706 -> 0.671–0.686`, `STOI 0.9966 -> 0.9763`) by over-weighting low-energy channels.
3. **Mid-run L-BFGS `u_scale` refresh + gain probe (`45+(45+90)` / `45+(45+45+45)`) — REJECTED**:
   - Resetting L-BFGS cleared `memory_size=25` curvature history and consistently hurt `binaural_test 80 dB` (`est/ref: 0.58 -> 0.42–0.51`, `SI-SDR: -0.06 -> -0.51 / -1.25 dB`).
4. **Saturation-triggered `adam_lr_scale` boost & `60+120` step split — REJECTED**:
   - `60+120` underperformed `45+135` (`binaural_test 50 dB`: `14.07 vs 14.44 dB`, `HASQI 0.731 vs 0.740`); scaling `adam_lr_scale` (`0.08 -> 0.134`) slightly nudged `binaural_test 80 dB` (`0.02 -> 0.09 dB`) at the cost of hurting `long_test 80 dB` (`7.16 -> 6.71 dB`, `HASQI 0.502 -> 0.478`).
5. **Active-HF-channel normalized `hf_err` (`hf_env_weight = 0.008`, `hf_scale` floor `0.04`) — ACCEPTED**:
   - Fixed the ~8x channel dilution cleanly without over-driving `80 dB` `long_test`:
     - `binaural_test 50 dB`: `HASQI 0.715 -> 0.728` (`nl 0.746 -> 0.758`), `4.2–7.5 kHz -3.0 -> -2.5 dB` (`SI-SDR 14.46 dB`).
     - `long_test 50 dB`: `SI-SDR 11.41 -> 13.33 dB` (`+1.92 dB`), `STOI 0.9966 -> 0.9969`, `HASQI 0.704`.
     - `binaural_test 80 dB`: `SI-SDR -0.06 -> +0.04 dB`, `STOI 0.8218 -> 0.8244`, `HASQI 0.228 -> 0.245` (`nl 0.268 -> 0.288`).
     - `long_test 80 dB`: `SI-SDR 6.94 dB`, `STOI 0.9427`, `HASQI 0.496`.
6. **Perceptual `0.70` global + `0.30` channel-normalized envelope + `SAI MSE` + `perceptual_nap_weight = 0.30` — ACCEPTED**:
   - Ablated `SaiPndLoss` vs. `SAI MSE`: `SaiPndLoss` boosted `binaural_test sloping` `HASQI` (`0.219 -> 0.300`) at the expense of `HASPI` (`0.442 -> 0.328` on `ohc05`, `0.998 -> 0.922` on `sloping`), whereas `SAI MSE` + blended channel-normalized envelopes + higher `perceptual_nap_weight` improved `STOI`, `HASPI`, and `HASQI` together (`P_best`: `binaural_test ohc05` `HASPI 0.4418 -> 0.5895`, `STOI 0.8425 -> 0.8549`, `HASQI 0.027 -> 0.039`; `binaural_test sloping` `HASQI 0.219 -> 0.323`; `long_test ohc05` `STOI 0.6140 -> 0.6994`, `HASPI 0.9301 -> 0.9378`; `long_test sloping` `STOI 0.7215 -> 0.7328`).

---

## Final Phase 2 Verification (`task-474` & `task-497`)

- **Lint & Format**: `ruff format --check .` (9 files formatted) and `ruff check .` passed.
- **Test Suite**: **`23 / 23` passed in `547.00 s`** (`PYTHONPATH=src ../.venv/bin/pytest -o addopts="--import-mode=importlib" -v`).
- **End-to-End Real-Speech `benchmark_suite` (`26` rows, `16 kHz`, `1.0 s`, `chunk_seconds=0.5`, Phase 1 `40+40` vs. Phase 2 `45+135`)**:

### Round-Trip & Pareto Comparison (Phase 1 -> Phase 2)

| Clip | Condition | SI-SDR (P1 -> P2) | STOI (P1 -> P2) | HASPI (P1 -> P2) | HASQI / nl (P1 -> P2) | 60–300 Hz SNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `binaural_test` | `clean_50dB` | `8.65 -> 14.45 dB` | `0.9212 -> 0.9741` | `0.9999 -> 1.0000` | `0.608/0.648 -> 0.722/0.753` | `-3.7 -> +3.6 dB` |
| `binaural_test` | `clean_65dB` | `3.47 -> 7.17 dB` | `0.8834 -> 0.9180` | `0.9994 -> 0.9997` | `0.315/0.361 -> 0.479/0.520` | `-0.9 -> +0.8 dB` |
| `binaural_test` | `clean_80dB` | `-2.60 -> -0.13 dB` | `0.7429 -> 0.8219` | `0.6300 -> 0.8863` | `0.133/0.157 -> 0.232/0.272` | `+0.9 -> +0.3 dB` |
| `binaural_test` | `snr10_65dB` | `4.10 -> 6.62 dB` | `0.8562 -> 0.9001` | `0.9907 -> 0.9999` | `0.285/0.320 -> 0.481/0.522` | `-0.2 -> +1.1 dB` |
| `binaural_test` | `pareto 20+60` | `3.47* -> 4.38 dB` | `0.8834* -> 0.8958` | `0.9994* -> 0.9996` | `0.315* -> 0.398/0.448` | `-0.9* -> -0.4 dB` |
| `long_test` | `clean_50dB` | `-2.53 -> +11.45 dB` | `0.9841 -> 0.9965` | `0.9999 -> 1.0000` | `0.483/0.525 -> 0.708/0.727` | `+0.6 -> +12.7 dB` |
| `long_test` | `clean_65dB` | `-3.06 -> +9.56 dB` | `0.9464 -> 0.9631` | `0.9980 -> 0.9987` | `0.324/0.381 -> 0.558/0.578` | `+0.5 -> +9.6 dB` |
| `long_test` | `clean_80dB` | `-6.65 -> +7.70 dB` | `0.7676 -> 0.9463` | `0.9064 -> 0.9983` | `0.151/0.214 -> 0.527/0.582` | `+0.2 -> +7.8 dB` |
| `long_test` | `snr10_65dB` | `-3.15 -> +5.96 dB` | `0.9720 -> 0.9944` | `0.9999 -> 0.9999` | `0.454/0.543 -> 0.665/0.720` | `+0.4 -> +6.5 dB` |
| `long_test` | `pareto 0+0` | `-7.11 -> +3.13 dB` | `0.8287 -> 0.9322` | `0.8758 -> 0.9875` | `0.130/0.191 -> 0.271/0.339` | `+0.0 -> +3.6 dB` |
| `long_test` | `pareto 20+60` | `-3.06* -> +5.44 dB` | `0.9464* -> 0.9672` | `0.9980* -> 0.9993` | `0.324* -> 0.518/0.555` | `+0.5* -> +5.1 dB` |

*\*Phase 1 Pareto comparison row is Phase 1's `40+40` budget (`~43.5–50.0 s` vs. Phase 2 `20+60` `~51.1–54.3 s`).*

### Cross-Cochlea Auralization Comparison (Phase 1 `perc010` -> Phase 2 `perc030`)

| Clip | Condition | STOI (P1 -> P2) | HASPI (P1 -> P2) | HASQI / nl (P1 -> P2) | NAP corr (P1 -> P2) | Env / SAI corr (P2) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `binaural_test` | `ohc05_noisy` | `0.8502 -> 0.8549` | `0.5519 -> 0.5970` | `0.096 -> 0.041` | `0.6041 -> 0.6675` | `0.9373 / 0.9068` |
| `binaural_test` | `sloping_noisy` | `0.9178 -> 0.9233` | `0.9658 -> 0.9938` | `0.249 -> 0.323` | `0.8378 -> 0.8721` | `0.9545 / 0.9741` |
| `binaural_test` | `ohc05_shaca` | `0.8063 -> 0.8100` | `0.9895 -> 0.9580` | `0.167 -> 0.142` | `0.6507 -> 0.7242` | `0.9094 / 0.9404` |
| `binaural_test` | `sloping_shaca` | `0.8273 -> 0.8830` | `0.9085 -> 0.8826` | `0.233 -> 0.295` | `0.8727 -> 0.9071` | `0.9684 / 0.9854` |
| `long_test` | `ohc05_noisy` | `0.5598 -> 0.6904` | `0.6252 -> 0.9377` | `0.094 -> 0.137` | `0.6487 -> 0.7205` | `0.9029 / 0.8714` |
| `long_test` | `sloping_noisy` | `0.6597 -> 0.7282` | `0.6338 -> 0.9522` | `0.157 -> 0.241` | `0.9394 -> 0.9598` | `0.9874 / 0.9908` |
| `long_test` | `ohc05_shaca` | `0.7350 -> 0.7843` | `0.8039 -> 0.9884` | `0.203 -> 0.216` | `0.6957 -> 0.7554` | `0.9176 / 0.9322` |
| `long_test` | `sloping_shaca` | `0.7206 -> 0.7943` | `0.9800 -> 0.9859` | `0.325 -> 0.420` | `0.8982 -> 0.9473` | `0.9828 / 0.9893` |

---

## Remaining Known Gaps (Hand-off to Phase 3)

1. **Deep AGC saturation (`>= 80 dB SPL`) on broadband/transient clips**: `binaural_test` at `80 dB SPL` improved from `-2.60 -> -0.13 dB` `SI-SDR` (`HASPI 0.630 -> 0.886`, `HASQI 0.133 -> 0.232`), but iterative L-BFGS still climbs slowly from the linear-regime warm start (`ws/ref = 0.22 -> est/ref ~ 0.59` in `180` steps).
2. **`> 4 kHz` carrier synthesis & `HASQI nl` ceiling**: Because CARFAC's IHC `lpf_coeff` (~2 kHz) removes carrier phase above ~4 kHz, gradient optimization recovers the `4.2–7.5 kHz` temporal envelope (`HASQI nl = 0.753` / `0.727` at `50 dB`) rather than exact voiced high-frequency harmonic phase (`4.5–7.5 kHz` waveform SNR stays near `-0.9` to `-2.7 dB`).
3. **Runtime (~51 s/clip at `20+60`, ~111 s/clip at `45+135`)**: Iterative autodiff through CARFAC dominates runtime.

---

## Phase 3: Dual-Domain Neural Residual Refiner & CARFAC Cycle Consistency

### Architecture ([`neural.py`](../src/carfac_inversion/neural.py)) & Training ([`train_neural_refiner.py`](../src/carfac_inversion/scripts/train_neural_refiner.py))
- **`NapResidualRefiner` (`1,297,790` parameters)**:
  - **Canonical cochleotopic alignment**: Piecewise-linear log-CF interpolation (`channel_projection_matrix`) resamples arbitrary cochlea channel counts (`65 ch` at 16 kHz, `77 ch` at 32 kHz) onto `64` ascending log-spaced CFs (`40–7200 Hz`).
  - **Scale-invariant amplitude handling + physical SPL conditioning**: Normalizes `x_ws` by `ws_rms_safe = max(ws_rms, 1e-6)` while conditioning every FiLM ConvNeXt block on `log10(ws_rms_safe)` plus 8-macro-band `log1p`-compressed target & residual (`target_nap - ws_nap`) statistics and log-RMS ratios.
  - **Dual-domain synthesis**:
    - Frame-rate (`n_fft=256`, `hop=64`) complex STFT + factored sample-to-frame NAP encoder driving a 6-block FiLM-conditioned dilated ConvNeXt backbone (`dim=160`, `dilations=(1,2,4,8,2,1)`).
    - Zero-initialized synthesis heads predicting global SPL log-gain (`delta_g in [-2.0, +2.5]`), complex ratio mask + additive complex STFT residual, and a sample-aligned lightweight 1D U-Net (`unet_base_dim=24`) with a silence gate (zero-input invariance verified at `0.0`; step-0 near-identity `max |out - x_ws| = 2.33e-9`).
  - **Integration**: Exposed via [`InversionConfig.neural_checkpoint`](../src/carfac_inversion/inversion.py), CLI `--neural_checkpoint` in [`invert_nap.py`](../src/carfac_inversion/scripts/invert_nap.py) and [`benchmark_suite.py`](../src/carfac_inversion/scripts/benchmark_suite.py), supporting both pure single-pass feedforward (`0+0`) and hybrid short L-BFGS polish (`0+15`).

### Speaker-Disjoint Dataset & Two-Stage Training Curriculum
- **Dataset** (`LibriSpeech dev-clean` split strictly by `speaker_id`, zero speaker overlap across splits or against `carfac/test_data`):
  - **Train**: `32` speakers (`384` utterances -> `766` active `1.0 s` windows, stratified across `43–86 dB SPL` with `25%` `5/10/15 dB` Gaussian noisy round-trip mixes).
  - **Validation (held-out)**: `4` unseen speakers (`64` x `1.0 s` segments across `50/65/80 dB` + `snr10_65dB`).
  - **Test (held-out)**: `4` unseen speakers (`[7850, 7976, 8297, 8842]`) plus canonical `carfac/test_data/{binaural_test.wav, long_test.wav}`.
- **Two-Stage Training (`task-740` on Apple M5 Pro CPU, checkpoint `4.85 MB`)**:
  - **Step 0 (`dsp_warm_start` val baseline)**: `val_loss = 31.1176`, `val_sisdr = -3.41 dB`, `stft = 112.67`, `hf_spec = 5.2580`, `log_rms = 1.4830`.
  - **Stage 1 (`1400` steps, `batch=8`, `854.4 s`)**: Multi-resolution STFT (`2048, 1024, 512, 128`), `3.8–7.8 kHz` spectral log-mag MSE, `tgt_rms`-normalized `wave_l1`, `log_rms`, `-SI-SDR`, and `<20 Hz` drift: reached **`val_loss = 0.6287`**, **`val_sisdr = +12.06 dB`**, `stft = 18.95`, `hf_spec = 0.9200`, `log_rms = 0.0373`.
  - **Stage 2 (`350` steps, `batch=4`, `183.3 s`)**: Added variance-normalized differentiable CARFAC NAP + `>3.5 kHz` envelope cycle consistency (`carfac_weight=15.0`): reduced held-out **`carfac_nap` MSE `0.0950 -> 0.0711`** while holding **`val_sisdr = +12.01 dB`**.
  - **Loss Scaling Lesson**: Initial `40+10` smoke check revealed `hp_acoustic`'s default `sqrt(win_len/2)` STFT scaling produced `stft = 112.7`, drowning out `SI-SDR` and Stage 2 `carfac_nap` (`0.27`); re-balancing to `0.15 * stft`, `1.5 * hf_spec`, `3.0 * wave_l1`, `2.5 * log_rms`, `-0.35 * sisdr`, and `15.0 * carfac_nap` enabled rapid convergence on both waveform SPL and HF spectral shape.

### Held-Out LibriSpeech `test` Evaluation (4 Unseen Speakers `[7850, 7976, 8297, 8842]`, `16 kHz`, `1.0 s`, `sii.CARFAC_RMS_ONE_DB_SPL ≈ 107.01 dB`)

| Condition | Method | Runtime | SI-SDR | STOI | HASPI | HASQI / nl | NAP corr | 4.5–7.5 kHz SNR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_50dB` | Phase 2 `0+0` | `3.36 s` | `4.10 dB` | `0.8394` | `0.9998` | `0.538 / 0.644` | `0.9496` | `0.0 dB` |
| `clean_50dB` | Phase 2 `20+60` | `53.79 s` | `16.34 dB` | `0.9822` | `0.9999` | `0.789 / 0.822` | `0.9996` | `-4.8 dB` |
| `clean_50dB` | Phase 3 `S1 0+0` | `3.07 s` | `19.14 dB` | `0.9882` | `0.9999` | `0.819 / 0.862` | `0.9933` | `-1.3 dB` |
| `clean_50dB` | Phase 3 `S2 0+0` | `3.08 s` | `19.39 dB` | `0.9881` | `0.9999` | `0.815 / 0.858` | `0.9946` | `-1.6 dB` |
| `clean_50dB` | Phase 3 `S2 0+15` | `15.89 s` | **`19.44 dB`** | **`0.9943`** | **`1.0000`** | **`0.860 / 0.898`** | `0.9996` | `-1.6 dB` |
| `clean_65dB` | Phase 2 `0+0` | `2.17 s` | `-2.64 dB` | `0.8119` | `0.9983` | `0.404 / 0.520` | `0.8278` | `0.0 dB` |
| `clean_65dB` | Phase 2 `20+60` | `51.53 s` | `10.69 dB` | `0.9346` | `0.9988` | `0.535 / 0.569` | `0.9982` | `-4.7 dB` |
| `clean_65dB` | Phase 3 `S1 0+0` | `3.20 s` | `17.77 dB` | `0.9868` | `0.9998` | `0.712 / 0.752` | `0.9803` | **`-0.9 dB`** |
| `clean_65dB` | Phase 3 `S2 0+0` | `3.21 s` | `17.80 dB` | `0.9887` | `0.9996` | `0.721 / 0.761` | `0.9844` | `-1.0 dB` |
| `clean_65dB` | Phase 3 `S2 0+15` | `14.68 s` | **`18.88 dB`** | **`0.9935`** | **`0.9999`** | **`0.785 / 0.822`** | `0.9986` | `-1.0 dB` |
| `clean_80dB` | Phase 2 `0+0` | `2.30 s` | `-4.58 dB` | `0.7132` | `0.7036` | `0.168 / 0.215` | `0.6875` | `-0.8 dB` |
| `clean_80dB` | Phase 2 `20+60` | `51.22 s` | `1.72 dB` | `0.8294` | `0.8682` | `0.273 / 0.331` | `0.9731` | `-3.6 dB` |
| `clean_80dB` | Phase 3 `S1 0+0` | `3.34 s` | `13.94 dB` | `0.9580` | `0.9994` | `0.583 / 0.617` | `0.9184` | **`-1.1 dB`** |
| `clean_80dB` | Phase 3 `S2 0+0` | `3.33 s` | `14.08 dB` | `0.9564` | `0.9995` | `0.598 / 0.632` | `0.9375` | `-1.2 dB` |
| `clean_80dB` | Phase 3 `S2 0+15` | `13.59 s` | **`14.42 dB`** | **`0.9629`** | **`0.9997`** | **`0.662 / 0.697`** | `0.9879` | **`-1.1 dB`** |
| `snr10_65dB` | Phase 2 `20+60` | `51.72 s` | `8.97 dB` | `0.9450` | `1.0000` | `0.644 / 0.695` | `0.9984` | `-0.1 dB` |
| `snr10_65dB` | Phase 3 `S2 0+0` | `3.28 s` | `10.33 dB` | `0.9675` | `0.9999` | `0.695 / 0.729` | `0.9778` | `-1.7 dB` |
| `snr10_65dB` | Phase 3 `S2 0+15` | `14.58 s` | **`10.73 dB`** | **`0.9844`** | **`1.0000`** | **`0.725 / 0.755`** | `0.9985` | `-1.7 dB` |

### Canonical Out-of-Domain `carfac/test_data` Comparison (`107.01 dB` Calibrated `9da7cbf`: P2 `45+135` -> P3 `S2 0+0` -> P3 `S2 0+15`)

| Clip | Condition | P2 `45+135` (`~110 s`) -> P3 `S2 0+0` (`~4.1 s`) -> P3 `S2 0+15` (`~14–26 s`) |
| :--- | :--- | :--- |
| `binaural_test` | `clean_50dB` | `SI-SDR 16.41 -> 12.49 -> 16.80 dB` \| `STOI 0.9848 -> 0.9803 -> 0.9864` \| `HASQI 0.816 -> 0.701 -> 0.739` \| `4k5 -2.0 -> -1.5 dB` |
| `binaural_test` | `clean_65dB` | `SI-SDR 9.26 -> 8.18 -> 9.44 dB` \| `STOI 0.9204 -> 0.9803 -> 0.9843` \| `HASQI 0.541 -> 0.509 -> 0.519` (`nl 0.572 -> 0.558`) |
| `binaural_test` | `clean_80dB` | `ws/ref 0.247 -> p3/ref 0.860` \| `STOI 0.8471 -> 0.9024 -> 0.9049` \| `HASPI 0.9989 -> 0.9991 -> 0.9986` \| `HASQI 0.358 -> 0.324 -> 0.325` |
| `binaural_test` | `snr10_65dB` | `SI-SDR 8.17 -> 6.24 -> 7.23 dB` \| `STOI 0.9092 -> 0.9671 -> 0.9740` \| `HASQI 0.433 -> 0.471 -> 0.500` (`nl 0.462 -> 0.533`) |
| `long_test` | `clean_50dB` | `SI-SDR 11.54 -> 15.71 -> 14.68 dB` \| `STOI 0.9977 -> 0.9828 -> 0.9908` \| `HASPI 1.0000 -> 1.0000` \| `HASQI 0.751 -> 0.616 -> 0.608` |
| `long_test` | `clean_65dB` | `SI-SDR 10.47 -> 13.91 -> 13.59 dB` \| `STOI 0.9677 -> 0.9770 -> 0.9809` \| `HASPI 0.9989 -> 0.9996 -> 0.9999` |
| `long_test` | `clean_80dB` | `SI-SDR 8.53 -> 11.89 -> 12.11 dB` \| `STOI 0.9734 -> 0.9377 -> 0.9548` \| `HASPI 0.9993 -> 0.9984 -> 0.9996` \| `4k5 -4.5 -> -4.4 dB` |
| `long_test` | `snr10_65dB` | `SI-SDR 6.82 -> 9.42 -> 9.49 dB` \| `STOI 0.9951 -> 0.9443 -> 0.9817` \| `HASPI 1.0000 -> 0.9997 -> 0.9998` \| `HASQI 0.660 -> 0.567 -> 0.596` |

*Refreshed Phase 2 `20+60` / cross-cochlea `perc030` (`9da7cbf`): `binaural_test` `pareto 20+60` `5.58 dB / 0.9086`, `sloping_noisy` `NAP 0.8724 / Env 0.9571 / SAI 0.9710 / HASPI 0.9841 / STOI 0.9351`; `long_test` `pareto 20+60` `6.90 dB / 0.9753`, `sloping_noisy` `NAP 0.9622 / Env 0.9874 / SAI 0.9901 / HASPI 0.8904 / STOI 0.7441`.*

### Calibration Alignment & CARFAC Parameter-Tuning Audit (`f571baa`, `9da7cbf`)
- **Peak-vs-RMS `3.01 dB` Calibration Alignment**: Standardized `benchmark_suite.py` and `invert_nap.py` to `sii.CARFAC_RMS_ONE_DB_SPL` (`104.0 + 20*log10(sqrt(2)) ≈ 107.01 dB SPL` at `RMS = 1.0`, matching `hp-acoustic` `transforms.py` and `perceptual.py`), correcting a `+3.01 dB` over-drive in nominal `50/65/80 dB` benchmark inputs.
- **Per-Ear `ohc_scale` vs. Per-Channel Ablation (Per-Channel Rejected)**: Measured closed-loop CARFAC effective undamping `(1 - agc_memory) * nlf(v)` on speech (`0.689–0.998`) and tested per-channel warm-start `ohc_scale`; rejected per-channel damping because warping adjacent-ERB group delays degraded Phase 3 `0+0` (`binaural_test`: `12.62 -> 9.79 dB` at `50 dB`, `8.46 -> 4.74 dB` at `65 dB`, `+1.14 -> -0.62 dB` at `80 dB`; `long_test 80 dB`: `13.28 -> 10.22 dB`). Replaced global cross-ear averaging with per-ear `ohc_scale` broadcasting (`[0.567, 1.000]` vs. `0.691` global on asymmetric `80 / 40 dB` binaural speech) while preserving identical mono behavior.
- **Dynamic Gain Bounds & High-Q / `48 kHz` IR Tails**: Extended `_calibrate_scalar_gain` brackets dynamically down to `1e-7` (fixing an Option C `20 dB` Pass-1 clamp error from `5.3x` down to `1.01x`), lowered Adam `rms0` and L-BFGS `u_scale_val` floors to `1e-7`, and scaled open-loop BM `ir_len` to `~0.15 s` up to `8192` samples (`171 ms` at `48 kHz`).

### Key Conclusions & Remaining Boundaries
1. **Runtime & Quality Pareto**: Single-pass Phase 3 (`0+0`) runs in **`~3.1–3.3 s/clip`** (held-out LibriSpeech) with **`19.39 / 17.80 / 14.08 dB` SI-SDR** across `50 / 65 / 80 dB`, beating Phase 2 `20+60` (`51–54 s/clip`) and `45+135` (`~110 s/clip`) by ~`16–36x` speedup; hybrid `0+15` (`~14–16 s/clip`) reaches `19.44 / 18.88 / 14.42 dB` (`HASQI 0.860 / 0.785 / 0.662`) and `nap_corr = 0.988–0.9996`.
2. **Scope for cross-cochlea auralization (`mode="perceptual"`)**: Because `Phase 3` was trained strictly on **same-cochlea (`healthy`)** targets, feeding it an **impaired** cochlea NAP (`ohc05_noisy` / `sloping_noisy`) pulls predictions toward healthy speech (`nap_corr = 0.33–0.49` at `0+0`) rather than staying aligned on the impaired cochlear manifold (`Phase 2 perc030`: `nap_corr = 0.67–0.96`). Cross-cochlea auralization should continue using Phase 2 `mode="perceptual"` until a dedicated impaired-target Phase 3b refiner is trained.
3. **Broadband `80 dB` transient phase (`binaural_test 80 dB`)**: Amplitude scaling (`ws/ref 0.247 -> p3/ref 0.860`) and intelligibility/quality (`HASPI 0.9991`, `STOI 0.8471 -> 0.9049`) improved cleanly, while time-domain waveform correlation remained `0.746` due to `4.5–7.5 kHz` fricative phase invariance.
4. **Test Coverage (`31 / 31` passed)**: [`inversion_test.py`](../src/carfac_inversion/inversion_test.py) (`17`), [`neural_test.py`](../src/carfac_inversion/neural_test.py) (`5`), [`benchmark_suite_test.py`](../src/carfac_inversion/scripts/benchmark_suite_test.py) (`5`), and [`invert_nap_test.py`](../src/carfac_inversion/scripts/invert_nap_test.py) (`4`).
