# Phase 7 — Evaluation, Artifact, and Provenance Evidence

Phase 7 converted the modality-specific smoke outputs into inspectable real-data evidence bundles. The focus was traceability and release safety, not metric optimization. Each bundle contains a `RunManifest`, sanitized metrics, a count-only prediction summary, and an independent verification record stored outside the repository.

| Modality | Dataset | Protocol ID | Experiment identity | Bundle verification |
|---|---|---|---|---|
| EMG | NinaPro DB5, Zenodo v1 | `ninapro-db5-two-subject-held-out-smoke-v1` | `exp_69d8e887107a55358c4f87a875c289120f1938de612285c6b000de3d0b43ee9c` | Passed |
| EEG | PhysioNet EEGMMID v1.0.0 | `eegmmid-r04-two-subject-held-out-smoke-v1` | `exp_c2edd9f6b540f33d59762a0d63865018c1125ce6ed47b2e50b1ad545f8b2931d` | Passed |
| ECG | PhysioNet MIT-BIH v1.0.0 | `mitbih-two-record-held-out-smoke-v1` | `exp_87a1b7ff9136cc1b0fc4dad8c26165f6eb87ffbb4f07409b6a5745810c92db78` | Passed |

The bundle verifier checks that the manifest identifies a real source with dataset version and license, declares a protocol ID, split, metrics, and non-window-level analysis unit, contains a deterministic experiment ID, and validates the SHA-256 hashes registered for the metrics and prediction-summary artifacts. It also rejects a bundle unless the prediction summary explicitly confirms that raw participant-level or beat-level predictions were not exported.

| Control | Implementation evidence |
|---|---|
| Raw-data exclusion | Repository ignore rules exclude `real_data/` and `artifacts/real_data/`; all retrieved signals remain in an isolated local workspace. |
| Scientific manifest completeness | Each run calls `validate_research_readiness()` before output publication. |
| Artifact integrity | Metrics and prediction-summary SHA-256 values are registered in the manifest and independently rechecked. |
| Claim boundary | Every metrics artifact declares that it is a real-data adapter/protocol smoke test, not a benchmark or inferential result. |
| Privacy minimization | Only aggregate class counts are exported. Raw window/beat predictions are explicitly absent. |
| Split transparency | EMG/EEG use held-out participant paths; ECG uses a held-out record path. |

## Interpretation boundary

Passing the evidence verifier means that the stored smoke bundle is internally consistent, provenance-aware, and release-reviewable. It does **not** establish scientific efficacy, clinical utility, robustness, generalization, fairness, or a foundation-model claim. Meaningful comparison requires fuller cohorts, locked protocols, appropriate baselines, participant/record-level uncertainty, and preregistered or otherwise auditable analysis choices.[1]

## References

[1]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
