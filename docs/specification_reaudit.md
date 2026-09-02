# Master Specification Re-Audit — V4.0.2

**Audit date:** 26 August 2026  
**Scope:** Fresh comparison of the delivered package against the supplied master specification, followed by targeted correction and clean-environment validation.

## Audit conclusion

The platform is **release-ready as research software** after correction of the identified public-metadata, licensing, documentation, and synthetic-evidence gaps. It is not release-ready for any stronger scientific or clinical claim. The remaining empirical gates are documented rather than concealed.

| Specification area | Audit outcome | Evidence or correction |
|---|---|---|
| One canonical platform and shared contracts | Pass | Canonical `Signal`, registry, adapters, preprocessing, encoders, task heads, provenance, and UI/CLI are retained under one package. |
| EMG, EEG, and ECG core tracks | Pass at adapter/protocol-smoke level | Governed real-data source register, provenance-aware loaders, opt-in integration tests, and non-claiming held-out-unit smoke artifacts exist for all three. |
| ECoG/iEEG maturity boundary | Pass | Registry and public metadata identify ECoG/iEEG as experimental; no benchmark or mature-support claim is made. |
| Configuration and experiment identity | Pass | Typed configuration validation and deterministic `experiment_id` controls are tested. |
| Modality-specific processing and explicit provenance | Pass | Canonical processing history, time/missingness preservation, source/version/license records, and output hashes are implemented. |
| Synthetic-data integrity | Corrected | The previously implicit optional fNIRS fallback now requires `allow_synthetic_fallback=True`, matching EMG, EEG, and ECG behavior. Related regression tests were updated. |
| Evaluation and statistical boundaries | Pass with empirical limitation | Subject/record-aware utilities and manifest fields exist. Minimal real-data runs remain smoke paths and explicitly avoid window-level inference or benchmark claims. |
| Multimodal fusion and missing-modality support | Engineering pass; empirical gate open | Explicit fusion and presence context are implemented. No shared synchronized cohort has yet completed the required alignment, unimodal-vs-fusion, or missing-modality ablation study. |
| Dashboard and CLI | Pass | `bsfm info` completed; a headless Streamlit dashboard launch reached local readiness at `127.0.0.1`. |
| Packaging, CI, documentation, and release controls | Corrected and passed | Required docs, CI jobs, wheel build, strict docs build, clean-install check, citation metadata, LICENSE/NOTICE, and a refreshed release manifest are included. |
| Citation and public claim boundaries | Corrected | The citation record no longer calls the software a foundation model, no longer contains a placeholder DOI, and describes the actual research-platform scope. |
| Third-party and dataset licensing | Corrected | `NOTICE` now agrees with the governed EMG, EEG, and ECG source-register licenses and explicitly states that raw datasets are not redistributed. |
| README workflow and links | Corrected | Removed malformed public demo/unsupported DOI presentation and replaced stale release references with current master, modality-phase, multimodal-readiness, and release documents. |

## Corrected findings

| ID | Finding in delivered package | Remediation |
|---|---|---|
| RA-01 | `CITATION.cff` used an invalid CFF version, claimed a unified foundation model, and contained a placeholder DOI. | Replaced with valid CFF 1.2.0 metadata and an honest research-platform citation without a placeholder DOI. |
| RA-02 | README contained a malformed live-demo link, presented an unsupported DOI badge, and linked to stale release documentation. | Removed unsupported/malformed public references and linked the canonical current documentation set. |
| RA-03 | The fNIRS optional loader could silently enter synthetic mode when real data was absent. | Added explicit development-only fallback opt-in, native-rate validation, and regression coverage. |
| RA-04 | `NOTICE` listed stale or incorrect active-dataset license statements. | Reconciled NinaPro DB5, EEGMMID, and MIT-BIH entries with the governed source register and source URIs. |
| RA-05 | `RELEASE_MANIFEST.md` reported obsolete test counts, coverage, release state, and references. | Replaced with current audited evidence and release constraints. |

## Clean validation evidence

A clean committed re-audit candidate passed the following checks.

| Gate | Result |
|---|---|
| Static analysis | `ruff check .` passed. |
| Automated tests | 349 passed; 3 opt-in governed-data integration tests skipped when external data paths were absent. |
| Coverage policy | 80.90% against a 75% minimum. |
| Documentation | `mkdocs build --strict` passed. |
| Core installation | `scripts/verify_core_install.py` passed; reported V4.0.2 and the EMG/EEG/ECG/ECoG/fNIRS registry. |
| CLI | `bsfm info` completed and reported the expected modality maturity matrix. |
| Dashboard | Headless Streamlit launch reached local URL readiness on loopback. |
| Packaging | Wheel build passed for `biosignal_fm-4.0.2-py3-none-any.whl`; SHA-256 `a622f83e375125b0bf0a3cfd522934b8e4f3dd87e6120903a703042630fa4df0`. |

## Remaining evidence gates — not defects to conceal

> The following are deliberate **open research gates**, not completed capabilities: broad benchmark validation, subject-level inferential statistics on adequate cohorts, transfer evidence for a foundation-model claim, a synchronized shared-cohort EMG+EEG fusion comparison, missing-modality ablations, clinical validation, and regulatory readiness.

The package remains appropriately described as a **research platform / representation-learning platform**. The existing EMG, EEG, and ECG real-data outputs are retained as reproducible adapter/protocol smoke records only. The multimodal readiness record identifies a synchronized-data candidate but forbids fusion claims until an alignment-aware shared-cohort study passes its protocol gates.[1]

## References

[1]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
