# CARFAC Inversion — Engineering Progress & Roadmap

## High-Level Goal
Build a world-class, state-of-the-art CARFAC neural activity pattern (NAP)-to-audio inversion and hearing-loss auralization system in JAX over a 2–3 month horizon.

---

## Strategic Roadmap

### Month 1: Generative SOTA & Data Scaling
- **Phase 4: HF Phase & Comb Carrier Synthesis**
  - Synthesize missing >4 kHz fine-timing harmonic/comb carrier structure from lower harmonic phase-locking.
  - Implement Flow Matching / Adversarial Refiner architectures to eliminate remaining fricative phase and tonal roughness artifacts.
  - Scale training pipeline across full LibriSpeech datasets.

### Month 2: Clinical Hearing Impairment, Auralization & Inversion Manifolds
- **Phase 5: Clinical Presets & Perceptual Optimization**
  - Integrate ISO 7029 and Bisgaard audiogram clinical profiles into CARFAC impairment modeling.
  - Fit inverse hearing-aid prescriptions directly onto cochlear firing manifolds.
  - End-to-end HASQI/HASPI perceptual metric optimization.

### Month 3: Real-Time Causal Streaming & Benchmark Hegemony
- **Phase 6: Low-Latency Streaming & Publication**
  - Real-time causal frame streaming with sub-20 ms end-to-end algorithmic latency.
  - Publication-ready benchmark suite against competing neural vocoders and vocoder-free inversion methods.
  - Interactive demonstration station and release artifacts.

---

## Completed Milestones (Phases 4, 5, & 5b)

- **16-TPU MLCR Runtime (`mlcr_tpu_launcher.py`)**: Verified `vlp:4x4` (16 TPU devices, `Mesh('data': 16)`) and updated launcher defaults to `--base_slice=vlp:4x4` with `--max_clips`, `--phase4_checkpoint`, `--sai_checkpoint`, and `--healthy_warmup_steps`.
- **Phase 4 (`tc100_gan_v1`, `CochleaDirectInverter` + 1D MPD/MRD GAN)**:
  - Scaled `cochlea_hybrid` on `librispeech_train_clean_100_32k` + 400-step Stage-2 1D GAN fine-tuning (`bc6c0ac`), reaching **`+24.53 dB` SI-SDR / `0.954` HASQI / `0.071` `nap_rt_rel` / `0.048` `sai_rt_rel` / `0.032` `log_stft`** on `dev-clean/test` (`3.6x` lower `nap_rt_rel`, `4.3x` lower `sai_rt_rel`, and `+13.20 dB` higher SI-SDR than `arcfac` `tcn`), and **`+21.62 dB` SI-SDR / `0.929` HASQI** zero-shot on `train-clean-100/test`.
  - On `1.0 s @ 32 kHz` real speech (`65 dB SPL`), outperforms iterative `arcfac_match_nap` across every metric (`nap_rt_rel` `0.065` vs `0.368–0.400`, `sai_rt_rel` `0.056` vs `0.253–0.305`, `log_stft` `0.106` vs `0.531–2.538`, `SI-SDR` `+17.82 dB` vs `-10.02..-7.75 dB`, `HASQI` `0.912` vs `0.064–0.159`, `flat:0.5 nap_kept` `0.170` vs `0.338–0.407`) at `~340x` faster warm inference (`0.40 s` vs `~137 s`).
- **Phase 5 (`cochlea_clinical_32k_v1`, Clinical Audiology & Inverse Prescriptions)**:
  - Added `Bisgaard` (`N1–N4`, `S1–S3`) and `ISO 7029` audiograms + differentiable JAX `HASQI`/`HASPI` losses (`clinical.py`, `perceptual_losses.py`, `train_phase5.py`, `benchmark_phase5.py`).
  - Achieved positive CARFAC NAP healing (**`+7.9%` to `+41.7%`**) across all 8 profiles and `50/65/80 dB SPL` (vs `NAL-R` `-21.1%` to `+1.1%`) and lifted severe `bisgaard:N4` `HASPI` (`0.297 -> 0.863` vs `NAL-R` `0.765`).
- **Phase 5b (`sai_vocos_32k_v1`, Direct Generative `SAI → Audio`)**:
  - Built `SaiDirectInverter` (`5.26M` params: high-CF unresolved harmonic-summed SACF pitch + DDSP `48`-harmonic bank + COLA `n_fft=1024, hop=128` STFT-filtered white noise + chunked modular phase integration) and Stage 1 + 1D MPD/MRD GAN training (`sai_inverter.py`, `train_sai.py`).
  - On unseen `dev-clean/test` (`7976-105575-0000`, `2.0 s @ 32 kHz`), outperforms `AFHI/arcfac` SAI baselines on **`HASQI` (`0.500` vs `0.344` `randphase` / `0.272` `noise` / `0.177` `slaney`)**, **`log_stft` (`0.181` vs `0.468 / 2.238 / 1.802`)**, **`STOI` (`0.912`)**, **`HASPI` (`0.860`)**, `flat:0.5` direction (`sai_dir = 1.006`, `sai_kept = 0.055`), **`sai_rt_rel` vs `arcfac_slaney` (`0.361` vs `0.856`)**, and **`>10,000x` faster warm inference (`0.08 s` vs `798–856 s`)**.
  - Generalizes across `5`-clip `dev-clean/test` (`HASQI = 0.472`, `log_stft = 0.164`, `sai_rt_rel = 0.390`), `train-clean-100/test` (`HASQI = 0.415`, `sai_rt_rel = 0.364`, `flat:0.5 sai_kept = 0.033`, `sai_dir = 1.004`), and `30–95 dB SPL` (`HASQI = 0.576 -> 0.322`, `sai_rt_rel = 0.453 -> 0.330`).
- **Phase 4 `v3` & `18.0 ms` Causal Strobe-SAI (`tc100_gan_v3`, `tc100_strobe_sai_v3`)**:
  - Added causal resting-silence AC gate (`zero_resting_gate`, suppressing resting silence floor to `-199.2 dB SPL`), basal traveling-wave dispersion alignment option (`causal_dispersion_align`), and multi-scale mel/low-frequency complex/log-RMS envelope + `--relative_nap` polish (`cochlea_hybrid_32k_tc100_gan_v3`), preserving **`+24.28 dB` SI-SDR / `0.950` HASQI / `0.072` `nap_rt_rel` / `0.052` `sai_rt_rel` / `0.032` `log_stft`** across `5` unseen `dev-clean/test` clips (`flat:0.5 nap_kept = 0.206`, `nap_dir = 1.017`, `sai_dir = 1.020`).
  - Added strictly causal 2-trigger strobe-SAI decoding (`causal_strobe_sai_to_nap`, `invert_strobe_sai_causal`: `10 ms` OLA hop + `8 ms` causal decoder = **`18.0 ms` algorithmic latency**) and `--strobe_sai_input` fine-tuned decoder (`cochlea_hybrid_32k_tc100_strobe_sai_v3`, `-174.7 dB SPL` resting silence, **`0.160` `log_stft` / `0.348` `HASQI` / `0.606` `sai_rt_rel`** on `5`-clip `dev-clean/test`).
- **AFHI Engineering Conventions Alignment**: Added `.afhi/` (`CONVENTIONS.md`, `python.md`, `research.md`, `index.md`, `REVISION`), `AGENTS.md`, and `CLAUDE.md` aligned with `Australian-Future-Hearing-Initiative/afhi-conventions` (`6e7b24f`).

---

## Environment & Reference Summary

### Environment Setup
- **Target Repository**: `/usr/local/google/home/shaikath/Projects/carfac-inversion`
- **Virtual Environment**: `.venv` synchronized via `uv` using Python 3.12.16 (`cpython-3.12.16-linux-x86_64-gnu`).
- **Dependencies Status**: All packages installed and verified (`carfac[jax]` @ `4cdb95e`, `hp-acoustic` @ `7d913ac`, `jax-haspi-hasqi` @ `f6601d4`, `jax==0.10.2`, `jaxlib==0.10.2`, `optax==0.2.9`, `grain`, `pystoi`, `pytest==9.2.0`, `ruff==0.17.5`).
- **Test Suite Verification**: **81 / 81 tests passing** (`ruff format --check .`, `ruff check .`, `pyrefly check`, and `pytest -q` all clean).

### Historical Reference Metrics (Phases 0–3b)
- **Phase 0–3b Test Suite**: 34 tests defined across `inversion_test.py` (17), `neural_test.py` (7), `benchmark_suite_test.py` (6), and `invert_nap_test.py` (4).
- **Phase 3 (Exact Mode) Headline Results** (4 held-out LibriSpeech speakers):
  - 50 dB clean: `19.39 dB` SI-SDR (`0+0` DSP+Neural, 3.08 s) / `19.44 dB` (`0+15` Hybrid, 15.89 s).
  - 65 dB clean: `17.80 dB` SI-SDR (`0+0`, 3.21 s) / `18.88 dB` (`0+15`, 14.68 s).
  - 80 dB clean: `14.08 dB` SI-SDR (`0+0`, 3.33 s) / `14.42 dB` (`0+15`, 13.59 s).
  - 65 dB, 10 dB SNR: `10.33 dB` SI-SDR (`0+0`, 3.28 s) / `10.73 dB` (`0+15`, 14.58 s).
- **Phase 3b (Perceptual Cross-Cochlea Mode) Headline Results** (4 held-out LibriSpeech speakers × 4 impairment profiles, 16 noisy cases):
  - Phase 2 L-BFGS (`20+60`, 56.78 s): `NAP corr 0.8760 | Env corr 0.9622 | SAI corr 0.9430 | STOI 0.8400 | HASPI 0.8991 | HASQI 0.165`.
  - Phase 3b Neural (`0+0`, 3.39 s): `NAP corr 0.8641 | Env corr 0.9336 | SAI corr 0.9205 | STOI 0.8420 | HASPI 0.8797 | HASQI 0.141`.
  - Phase 3b Hybrid (`0+15`, 16.49 s): `NAP corr 0.8755 | Env corr 0.9613 | SAI corr 0.9425 | STOI 0.8321 | HASPI 0.8856 | HASQI 0.160` (3.4x faster than Phase 2 `20+60`).

---

## Next Action
- **Phase 6**: Low-latency causal streaming (`<20 ms`) & scaling `SaiDirectInverter` on `16`-TPU `vlp:4x4` slices for longer horizons to close the remaining pixel-`sai_rt_rel` gap to iterative per-clip waveform optimization (`0.114`) while preserving `0.08 s` feedforward speed and superior `HASQI`/`log_stft`.
