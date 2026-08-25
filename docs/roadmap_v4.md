# BioSignal-FM V4 Roadmap

## Purpose

This roadmap converts V4's declared limitations into measurable work. Dates are planning horizons, not promises of scientific or regulatory results. A maturity or public-performance claim can advance only after its specified evidence is complete.

> A synthetic experiment does not become a benchmark, and an encoder does not become a validated foundation model, merely by completing software work.

## Priorities and acceptance evidence

| Priority | Target horizon from V4 approval | Executable objective | Acceptance evidence |
|---|---:|---|---|
| P0 | Completed | Establish release traceability and real Git capture in `RunManifest`. | Clean Git repository; committed release; tested `git_head` and clean/dirty state. |
| P0 | 0–4 weeks | Onboard the first licensed real dataset for every core modality. | Dataset manifest, license/version/access record, real-loader local integration test, and no silent synthetic fallback. |
| P0 | 0–4 weeks | Complete public-release hardening. | Dependency audit clean, pinned Actions, required secret configuration, external deployment checklist, and release review report. |
| P1 | 4–8 weeks | Run one small, reproducible single-modality baseline study per core modality. | Locked protocol, participant/record-level results, recorded baselines, run manifests, and confidence intervals. |
| P1 | 6–10 weeks | Extend CI without making extras mandatory for the core. | Core installs verified on Ubuntu/macOS/Windows and supported Python versions; optional-extra smoke jobs isolated. |
| P2 | 8–12 weeks | Select a future ECoG/iEEG dataset using documented criteria. | Dataset-selection memo, license/access review, electrode metadata audit, local adapter test, and data manifest. |
| P2 | 10–16 weeks | Evaluate a real multimodal study. | Documented synchronization or justified fusion design, preregistered unimodal comparator, and missing-modality analysis. |
| P3 | Continuous | Build release provenance and maintenance automation. | Trusted publisher, protected release workflow, attestations/signatures, dependency update review, and security review cadence. |

## P0 — Real-data onboarding

Raw biosignal data must not be committed to the repository. Each dataset must have a manifest in a project-controlled metadata location and pass the [Data Governance](data_governance.md) onboarding gate.

| Core modality | Initial candidate | Required work | Acceptance criteria |
|---|---|---|---|
| EMG | NinaPro DB5 or another explicitly licensed alternative | Confirm lawful access, record version/license, document channels/participants/sessions, and run the EMG loader on a real local file. | `DataOrigin.REAL`, source version, license, checksum or lawful equivalent, and an optional local integration test. |
| EEG | PhysioNet EEG Motor Movement/Imagery or another licensed alternative | Record license/version, document EDF montage/events, and run `EEGMMIDLoader` on a real file. | Real EEG sample with file provenance, modality, sampling rate, and a test that prevents synthetic fallback when files are available. |
| ECG | PhysioNet MIT-BIH Arrhythmia Database or another licensed alternative | Record license/version, document WFDB records/annotations, and run `MITBIHLoader` on a real record. | Real ECG sample with record context; test of expected channels and 360 Hz where MIT-BIH is used. |

No public benchmark result may be published without a dataset hash/version, preserved split, locked preprocessing configuration, `real` provenance, and a complete `RunManifest`.

## P1 — Reproducible protocols and CI

For every study, define the participant or record as the analysis unit, splits, training-only normalization, metrics, baselines, seeds, and stopping rule. LOSO and LODO definitions remain unchanged; this roadmap does not alter them.

| Deliverable | Entry criterion | Exit criterion |
|---|---|---|
| Single-modality benchmark | Licensed documented real dataset | Per-fold results, participant/record aggregation, confidence intervals, and recorded baselines. |
| Cross-participant study | Valid single-modality baseline | Explicit leakage prevention and session/domain-shift analysis. |
| Multimodal fusion study | Synchronized data or justified fusion design | Preregistered unimodal comparator and missing-modality analysis. |
| Foundation-model claim review | Multiple documented studies | Evidence of transfer/generalization across tasks and datasets plus independent claim review. |

The CI expansion plan is: core install plus `verify_core_install.py` across Ubuntu, macOS, and Windows on Python 3.10–3.12; one reference Ubuntu scientific/model job; isolated MNE/WFDB/h5py extra-import jobs; and strict documentation build. No optional reader or model package should become a core requirement.

## P2 — Experimental ECoG/iEEG path

ECoG work begins with dataset selection, not performance claims. The selection memo must cover license and redistribution constraints, participant/privacy context, electrode locations and metadata, sampling and event/task description, true analysis-unit size, adapter-readability, and provenance fields that do not disclose sensitive information.

The experimental exit criterion is a local real-file adapter integration test, metadata audit, and dataset manifest. It does not promote ECoG to core status or establish a benchmark.

## P3 — Release and operational maturity

Every external deployment needs an environment-specific review of API-key storage and rotation, network ingress, proxy limits, model-file permissions, CI secrets, non-root container behavior, and actual dependency licenses. ONNX, FastAPI, and Streamlit are optional capabilities, not evidence of real-time suitability or compliance.

The roadmap is reviewed at every minor release and whenever a new dataset or modality is added. A roadmap item closes only with links to dated test output, manifests, or reports. [Known Limitations](known_limitations_v4.md) remains the claim-boundary reference; [V4 Architecture](architecture_v4.md) remains the architecture reference.
