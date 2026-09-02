# Phase 8 — Multimodal Fusion and Missing-Modality Readiness

BioSignal-FM’s framework supports explicit fusion and carries immutable `present_modalities` context through representation and task execution. This platform-level capability is **not equivalent** to a scientifically established EMG+EEG fusion or missing-modality result. The real-data modality studies in phases 4–6 use independent datasets and must not be fused.[1]

## Candidate synchronized sources

| Candidate | Synchronization and coverage | License | Readiness decision |
|---|---|---|---|
| PhysioNet Multimodal Gait Dataset v1.0.0 | 19-channel EEG at 300 Hz, surface EMG from 12 lower-limb muscles, IMU and force plates from healthy young adults walking at three controlled speeds. The record states that modalities were aligned by a trigger-based protocol. | CC BY 4.0 | **Provisionally suitable for a future fusion-adapter investigation.** Requires a file-level audit of timestamps, trigger/event fields, missing-file patterns, participant identifiers, and a locked gait task protocol before loading or benchmarking. |
| Mendeley 8-channel EMG, EEG Upper Limb Gesture Data v1 | 11 participants, 8-channel Myo EMG at 200 Hz and 8-channel OpenBCI EEG at 250 Hz, collected with the same task protocol. The dataset description says devices were acquired simultaneously on independent computers with manual offline data-collection start/stop. | CC BY 4.0 | **Not approved for sample-level fusion without further evidence.** The public description does not establish an adequate timestamp/event/device-offset model for the framework’s alignment requirement. |

## Current readiness matrix

| Requirement | Framework status | Real evidence status | Gate outcome |
|---|---|---|---|
| Explicit pre-head fusion | Implemented and unit-tested. | No synchronized real EMG+EEG adapter run. | Engineering-ready; scientific gate not passed. |
| Modality availability context | Implemented with immutable `present_modalities`. | No controlled ablation on a shared real cohort. | Engineering-ready; robustness gate not passed. |
| Time alignment | Canonical signals can carry time semantics. | Independent EMG and EEG phase sources cannot be aligned. | Blocked for current data. |
| Synchronization-aware source | Candidate identified through the PhysioNet gait record. | File-level synchronization audit not yet executed. | Candidate only. |
| Unimodal versus fusion comparison | Interface allows it. | No locked shared task, split, or baseline table. | Not passed. |
| Missing-modality evaluation | Presence metadata prevents implicit zero-filling. | No declared dropping/imputation policy or real-data ablation. | Not passed. |

## Claim boundary

No multimodal performance, missing-modality robustness, clinical, or foundation-model claim is made. A future fusion phase must first implement an adapter for a single synchronized cohort, preserve its timestamp/trigger provenance, declare an alignment method and tolerance, lock participant-level splits, compare each unimodal model with an interpretable fusion baseline, and predefine missing-modality ablations. Failing that gate, the appropriate behavior is to leave fusion as an explicit framework capability rather than fabricate empirical support.[1]

## References

[1]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
[2]: https://physionet.org/content/multimodal-gait-dataset/1.0.0/ "PhysioNet Multimodal Gait Dataset v1.0.0"
[3]: https://data.mendeley.com/datasets/m6t78vngbt/1 "Mendeley EMG–EEG Upper Limb Gesture Dataset v1"
