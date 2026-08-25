# FDA Good Machine Learning Practice (GMLP)

BioSignal-FM is **not** FDA-cleared Software as a Medical Device (SaMD).
However, the design follows FDA GMLP principles to anticipate future
clinical translation.

## GMLP Principles Applied

### 1. Multidisciplinary Expertise

The 6 design lenses (Systems, Biomedical, Entrepreneur, Artist, AI/ML, DL
Researcher) ensure multidisciplinary review of every architectural decision.

### 2. Good Software Engineering

- Modular, typed, documented code
- 81% test coverage with 172 passing tests
- CI/CD on Python 3.10/3.11/3.12
- Pre-commit hooks (ruff, mypy, format)

### 3. Clinical Study Endpoints

Benchmark targets are clinically meaningful:
- EMG gesture recognition ≥ 85% (prosthetic control usability threshold)
- ECG arrhythmia F1 ≥ 0.90 (clinical diagnostic accuracy)
- EEG motor imagery ≥ 70% (BCI communication threshold)

### 4. Independent Review

- Hostile audit checklist (37 tests) verifies compliance with the master spec
- Statistical rigor suite enforces honest reporting

### 5. Representativeness

- Public datasets from diverse populations (NinaPro: 10 subjects;
  EEGMMID: 109 subjects; MIT-BIH: 48 records)
- Subject-aware cross-validation prevents overfitting to specific subjects

### 6. Monitoring

- RunManifest tracks every training run with SHA-256 + env fingerprint
- MLflow tracking integration for production deployment
- ONNX export with REAL numerical parity verification

## Future Clinical Translation

For clinical use, the following additional steps would be required:

1. IEC 62304 software safety classification (likely Class B)
2. FDA 510(k) submission with substantial equivalence evidence
3. Clinical validation study (multi-site, prospective)
4. Post-market surveillance plan
5. Quality Management System (ISO 13485)

BioSignal-FM v0.1.x is research-grade only.
