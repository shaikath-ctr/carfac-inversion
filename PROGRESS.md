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

## Baseline Status (Day 1)

### Environment Setup
- **Target Repository**: `/usr/local/google/home/shaikath/Projects/carfac-inversion`
- **Virtual Environment**: `.venv` synchronized via `uv` using Python 3.12.16 (`cpython-3.12.16-linux-x86_64-gnu`).
- **Dependencies Status**: All 103 packages installed and verified (`carfac[jax]` @ `4cdb95e`, `hp-acoustic` @ `7d913ac`, `jax-haspi-hasqi` @ `f6601d4`, `jax==0.10.2`, `jaxlib==0.10.2`, `optax==0.2.9`, `grain`, `pystoi`, `pytest==9.2.0`, `ruff==0.17.5`).
- **Test Suite Verification**: **34 / 34 tests passing** verified across `inversion_test.py` (17), `neural_test.py` (7), `benchmark_suite_test.py` (6), and `invert_nap_test.py` (4).

### Existing Benchmark Reference Metrics (Phases 0–3b)
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

## Immediate Next Action
- **Phase 4 Plan**: Data pipeline scaling & Generative HF excitation design (HF Phase & Comb Carrier Synthesis).
