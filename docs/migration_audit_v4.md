# BioSignal-FM V4 Migration Closure Audit

**Historical closure date:** 22 August 2026  
**Scope:** Closure of specified migration gaps after the v3.3-to-V4 implementation.  
**Status:** Engineering migration closed. Real-data and scientific-evidence limits remain explicit and are not superseded by this audit.

> This record describes executed engineering checks. It does not establish scientific performance, real-data benchmark results, clinical readiness, or regulatory clearance.

## Migration decision

V4 migrated the platform within one source tree rather than rebuilding it in parallel. The `core` package defines canonical contracts; `modalities` defines maturity and edge adapters; compatible loaders, preprocessing, models, and evaluation tools remain behind those boundaries. EMG, EEG, and ECG are core modalities; ECoG/iEEG is experimental; fNIRS is an optional legacy extension.

| Migration area | V4 decision | Closure evidence |
|---|---|---|
| Signal and provenance contracts | Retained and restructured in `core` | V4 contract tests and structured synthetic provenance. |
| Modality registry and adapters | Added in `modalities` | EMG/EEG/ECG/ECoG/fNIRS registry and clean-core installation test. |
| Preprocessing and models | Preserved and adapted | Adapters/services connect them to contracts without data readers in the core. |
| Demo data | Preserved with mandatory labeling | `synthetic` and `benchmark_eligible: false` propagate through contracts, UI, and CLI. |
| Documentation | Repositioned to evidence-bound research language | Architecture, limitations, roadmap, and tests distinguish software from performance evidence. |
| Reproducibility | Strengthened | Real Git repository and live `RunManifest` clean/dirty-state test. |

## Requested closure items

| Item | Completed action | Historical measured evidence | Closure date |
|---:|---|---|---|
| 1 | Ran `ruff check biosignal_fm tests scripts`; corrected one real I001 import-order finding in `scripts/verify_core_install.py`. | Clean Ruff check in the closure suite. | 22 August 2026 |
| 2 | Added independent EEG and ECG end-to-end tests: loader with labeled synthetic fallback, modality preprocessing, encoder, prediction, ONNX export, and numerical parity. | Targeted tests: 2 passed; full suite after addition: 325 passed. | 22 August 2026 |
| 3 | Added [V4 Roadmap](roadmap_v4.md) and linked it from README and MkDocs. | Strict documentation build succeeded. | 22 August 2026 |
| 4 | Initialized Git and added a live clean/dirty `RunManifest` test. | Historical first clean commit `169d1fdc8c2378bb6f568426137409741a9743a0`; full suite after item: 326 passed. | 22 August 2026 |

## Historical verification snapshot

The previous closure suite recorded 326 passing tests, 80.41% coverage, zero Ruff findings, format compliance, zero mypy errors across 68 source files, a strict documentation build, and a clean isolated core installation. Those values are historical only. The current pre-publication verification report supersedes them with results measured after release-hardening changes.

## Intentional remaining boundaries

V4 does not bundle licensed real data, real benchmark results, or validated multimodal-transfer evidence. It does not change LOSO/LODO definitions or modality definitions. The [V4 Roadmap](roadmap_v4.md) tracks evidence-building work, while [Known Limitations](known_limitations_v4.md) defines claim boundaries.

## Migration acceptance

The migration is accepted for **software architecture and research-platform engineering**: contracts, registry, core-modality integration paths, documentation, reproducibility, clean core installation, and static verification are covered. This acceptance must not be used to present synthetic output as a benchmark or the platform as a clinical tool or validated foundation model.
