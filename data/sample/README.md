# Sample Data

This directory contains a small smoke-test sample for verifying the BioSignal-FM
pipeline works without downloading the full NinaPro, PhysioNet, or Brain-BIDS
datasets.

## Files

- `smoke_test.npz` — A 1 MB NPZ file containing:
  - `signal`: shape `(8, 16, 400)` — 8 EMG windows, 16 channels, 400 samples (2s @ 200 Hz)
  - `modality`: `"emg"`
  - `sampling_rate_hz`: 200
  - `subject_ids`: shape `(8,)` — subject IDs 0-3
  - `labels`: shape `(8,)` — class labels 0-3
  - `label_names`: `("rest", "thumb_flex", "index_flex", "fist")`

## Usage

```python
import numpy as np
from pathlib import Path

data = np.load(Path("data/sample/smoke_test.npz"), allow_pickle=True)
signal = data["signal"]  # (8, 16, 400)
subject_ids = data["subject_ids"]
labels = data["labels"]

# Verify shape
assert signal.shape == (8, 16, 400)
```

## Regenerating

To regenerate the smoke test data with a different seed or larger size:

```bash
python scripts/make_sample_data.py --output data/sample/smoke_test.npz --seed 42
```
