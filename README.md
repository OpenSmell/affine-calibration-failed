# Affine Calibration — A Documented Failure

This repository records an approach that did not work.

## What we tried

Assume two e-nose devices (or the same device at different times) differ by
a per-sensor gain and offset in log space. Learn that transform from a small
set of calibration measurements and apply it to all future readings.

## What happened

Tested on the UCI Gas Sensor Array Drift Dataset (16 MOX sensors, 6 pure
gases, 36 months of drift data). Different time batches were treated as
different "devices." An affine transform was learned from 60 calibration
points (10 per gas) using Ridge regression.

- Classification accuracy before calibration: **47%**
- Classification accuracy after calibration: **33%**

The correction degraded performance.

## Why it failed

Sensor drift is not a simple gain-and-offset process. It involves non-linear
changes in sensitivity profiles, cross-sensitivity patterns between sensors,
and baseline resistance shifts that interact with temperature, humidity, and
prior exposure history. A linear correction cannot capture these effects.

## What we learned

Calibration must be learned from data, not assumed as a mathematical form.
This failure motivated the development of the learned latent space approach
that ultimately succeeded.

## What replaced it

- **Session-invariance proof (81.8% accuracy):** `opensmell/session-invariance`
- **Universal encoder (R² = 0.892):** `opensmell/universal-encoder`
- **Full project:** `opensmell/opensmell`

## Reproducing

```bash
pip install -r requirements.txt
python affine_calibration.py
```

The script will reproduce the 47% → 33% accuracy drop on the UCI dataset.

## License

MIT
