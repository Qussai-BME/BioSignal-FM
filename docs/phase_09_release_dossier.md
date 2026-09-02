# Phase 9 — Release Dossier

This initial V4.0.1 release dossier is retained as execution history. It is superseded for distribution by the **V4.0.2 specification re-audit** in [`specification_reaudit.md`](specification_reaudit.md) and the top-level `RELEASE_MANIFEST.md`. The V4.0.2 candidate was validated in a clean committed repository and packaged only after its fresh-install verification passed.

| Validation gate | Evidence | Result |
|---|---|---|
| Static analysis | Full repository `ruff check .` | Passed |
| Automated tests | `pytest -q` with the repository coverage policy | 349 passed, 3 opt-in real-data tests skipped, 80.90% coverage against a 75% minimum |
| Documentation site | `mkdocs build --strict` | Passed |
| Core installation | Fresh virtual-environment install followed by `scripts/verify_core_install.py` | Passed; version `4.0.2`; registry reported EMG, EEG, ECG, ECoG, and fNIRS |
| CLI/dashboard | `bsfm info` and headless local Streamlit startup | Passed |
| Packaging | `pip wheel . --no-deps` | Passed; built `biosignal_fm-4.0.2-py3-none-any.whl` |
| Wheel identity | SHA-256 of the audited wheel | `a622f83e375125b0bf0a3cfd522934b8e4f3dd87e6120903a703042630fa4df0` |
| Working-tree test | Clean committed copy used for release validation | Passed |

The three real-data integration tests are intentionally opt-in. They are skipped in ordinary CI when the caller has not configured an external, governed data path. Each can be executed only after independently retrieving the registered source data, preserving the project’s rule that protected or bulky raw signals are not committed to the repository.

## Release contents

The release candidate includes corrected and provenance-aware real-data loaders for NinaPro DB5 EMG, PhysioNet EEGMMID EEG, and PhysioNet MIT-BIH ECG; explicit synthetic-development fallback; annotation-aware real-data smoke paths; sanitised artifact-bundle verification; the master specification; source register; modality contract documentation; phase records; and an explicit multimodal readiness gate.[1]

## Residual constraints

The package is an engineering and research-platform release, not a clinical, regulated, benchmark-certified, or foundation-model release. The real-data protocol runs cover only minimal source subsets and are intentionally labeled as smoke tests. The candidate synchronized EMG+EEG source has not yet completed its file-level adapter and alignment gate, so no fusion performance or missing-modality robustness claim is included.[1]

## References

[1]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
