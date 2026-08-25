# Benchmarks

This directory contains standardized evaluation protocols for BioSignal-FM.

## Available Benchmarks

### EMG Gesture Recognition (8-class)

- **Dataset:** NinaPro DB5 (10 subjects, 16 channels, 2 kHz)
- **Protocol:** Leave-One-Subject-Out (LOSO)
- **Metric:** Accuracy + Macro F1
- **Target:** ≥ 85% accuracy (base model)

### ECG Arrhythmia Detection (5-class AAMI)

- **Dataset:** PhysioNet MIT-BIH (48 records, 2 leads, 360 Hz)
- **Protocol:** Leave-One-Record-Out
- **Metric:** F1 (per-class) + Macro F1
- **Target:** ≥ 0.90 F1

### EEG Motor Imagery (4-class)

- **Dataset:** PhysioNet EEGMMID (109 subjects, 64 channels, 160 Hz)
- **Protocol:** Leave-One-Subject-Out
- **Metric:** Accuracy + Macro F1
- **Target:** ≥ 70% accuracy

### fNIRS Mental Workload (2-class)

- **Dataset:** Brain-BIDS fNIRS (per-study, 8-32 channels, 10 Hz)
- **Protocol:** Leave-One-Subject-Out
- **Metric:** Accuracy
- **Target:** ≥ 75% accuracy

### Cross-Modal Transfer (Zero-Shot)

- **Setup:** Pretrain on EMG, freeze encoder, fit linear probe on EEG with ≤100 labeled samples
- **Baseline:** EEG-only pretraining under same data budget
- **Metric:** Accuracy improvement (Hedges' g ≥ 0.5)
- **Target:** Statistically significant improvement

## Running Benchmarks

The commands below run today, against the current CLI. **`bsfm evaluate`
runs the full Friedman/Nemenyi/Wilcoxon-Holm-Šídák statistical pipeline on
synthetic per-fold LOSO data** — this is the right way to verify the
statistics machinery end-to-end (its own `--help` text says exactly this),
but it is not yet a per-real-dataset benchmark runner:

```bash
bsfm evaluate --checkpoint model.pt --modality emg --n-classes 8 \
    --n-channels 16 --signal-length 400 --protocol loso
```

`--protocol` accepts `loso` or `lodo`. There is currently no `--dataset`
flag and no cross-modal-transfer flag — running the four per-dataset
benchmarks and the cross-modal transfer benchmark defined above against
*real* data means loading them with the corresponding loader
(`NinaProDB5Loader`, `MITBIHLoader`, `EEGMMIDLoader`, `FnirsLoader` — all in
`biosignal_fm.data`) and driving `FineTuner`/`friedman_nemenyi_test`
directly from a short Python script, the same way `scripts/run_full_study.py`
does for the synthetic comparison. Real per-dataset loaders with full
WFDB/EDF support and a single CLI entry point are the v0.2.0 roadmap item
(see the main README's Roadmap section) — the benchmark *definitions* above
are accurate targets; the automated multi-dataset runner is not built yet.

## Statistical Reporting

All benchmark results include:

- Per-fold metrics (accuracy, F1, etc.)
- Mean ± std across folds
- 95% BCa bootstrap confidence intervals
- Friedman test (if comparing multiple methods)
- Nemenyi post-hoc (if Friedman rejects null)
- Wilcoxon signed-rank with Holm-Šídák correction (for pairwise comparison)
- Hedges' g effect size

See `biosignal_fm.evaluation.statistics` for the full statistical rigor suite.
