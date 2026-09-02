# BioSignal-FM V4.0.2 Audited Release Manifest

## Release identity

| Field | Value |
|---|---|
| Package | `biosignal-fm` |
| Version | `4.0.2` |
| Release type | Audited research-software patch release |
| Git tag | `v4.0.2`; the tag identifies the exact audited commit. |
| Source language | English |
| Project license | Apache License 2.0 |
| Citation | `CITATION.cff`; no DOI is asserted until a real release DOI is issued. |

## Verification record

The tagged source tree must be validated in a clean committed environment immediately before packaging. The V4.0.2 release record contains the following measured gates.

| Gate | Result |
|---|---|
| Static analysis | `ruff check .` passed. |
| Automated tests | 349 passed; 3 external-governed-data integration tests skipped when their data paths were not supplied. |
| Coverage | 80.90% against the project threshold of 75%. |
| Documentation | Strict MkDocs build passed. |
| Core installation | Fresh core smoke passed; version and registry were asserted. |
| CLI | `bsfm info` passed and displayed the modality maturity matrix. |
| Dashboard | Headless Streamlit launch reached a loopback readiness URL. |
| Packaging | Built `biosignal_fm-4.0.2-py3-none-any.whl`; SHA-256 `a622f83e375125b0bf0a3cfd522934b8e4f3dd87e6120903a703042630fa4df0`. |
| Git state | Source archive is created from the clean `v4.0.2` tag. |

The detailed specification comparison, corrections, and claim boundaries are in [`docs/specification_reaudit.md`](docs/specification_reaudit.md). The initial real-data smoke evidence is in the modality phase records and the cross-modality evidence record; it is not a substitute for an adequately powered benchmark study.

## Included material

The source archive contains code, tests, English documentation, GitHub workflows, container/configuration assets, release metadata, and notices. It excludes raw biosignals, participant-level or beat-level predictions, credentials, local virtual environments, caches, generated coverage, and build artifacts. Governed raw data remains external to the repository and only sanitized summaries/manifests may be distributed separately.

## Publication boundary

BioSignal-FM V4.0.2 is **research software**. It does not claim a validated foundation model, clinical efficacy, diagnostic performance, regulatory clearance, universal cross-subject generalization, real-data benchmark superiority, multimodal fusion gain, or missing-modality robustness. Those claims require separately documented datasets, protocols, splits, analyses, comparisons, and reproducible evidence.

## Open scientific gates

The following requirements remain open by design: broad real-data benchmark studies; subject/record-level inference on adequate cohorts; transfer evidence for a foundation-model claim; a synchronized shared-cohort EEG–EMG adapter and alignment audit; unimodal/fusion/missing-modality ablations; clinical validation; and regulatory assessment. These are tracked as research roadmap gates rather than represented as release defects or completed performance claims.
