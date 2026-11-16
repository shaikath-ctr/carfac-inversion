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

`carfac-inversion` solves the inverse `NAP -> audio` problem in two stages:

1. **Physics-Based CARFAC DSP Warm Start (`dsp_warm_start`)**:
   - Measures open-loop CARFAC basilar-membrane impulse responses through any
     configured conductive and middle-ear front-end filters at an AGC-damped
     operating Q (`ohc_scale`).
   - Excludes apical channels below `min_warm_start_cf_hz` (default `250 Hz`)
     during linear warm-start inversion so quadratic `ohc_nlf` combination-tone
     distortion is not amplified into sub-bass rumble.
   - Compensates discrete-time IHC + synapse `lpf_coeff` phase lag and rolloff
     below `hf_crossover_hz` (`4200 Hz`), extracts high-pass-cleaned envelopes
     modulated onto bandlimited noise above `hf_crossover_hz`, and calibrates
     per-channel RMS SPL plus 1-shot spectral tilt equalization.
2. **Scale-Invariant JAX Optimization (`invert_nap`)**:
   - Projects iterates onto the audible band `[cutoff_low_hz, cutoff_high_hz]`
     in rFFT space to prevent sub-audible drift and Nyquist ringing.
   - Optimizes via cosine-decayed Adam followed by RMS-normalized L-BFGS in two
     modes:
     - `mode="exact"`: Sample-level NAP MSE plus a >3.5 kHz envelope anchor for
       same-cochlea round-trip reconstruction.
     - `mode="perceptual"`: Multi-scale NAP envelopes (2 ms and 8 ms) +
       Stabilized Auditory Image (SAI) matching for cross-cochlea hearing-loss
       auralization (inverting an impaired cochlea's NAP through a healthy
       cochlea even when OHC loss alters traveling-wave group delays).

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
