# Legacy Architecture Reference

This page is retained for link compatibility. [BioSignal-FM V4 Architecture](architecture_v4.md) is the authoritative reference for current contracts, boundaries, and the modality registry.

## What changed in V4

| Topic | Current position |
|---|---|
| Core | Dependency-light `SignalMetadata`, `SignalProvenance`, `Signal`, and `SignalBatch` contracts. |
| Modalities | EMG, EEG, and ECG are core; ECoG/iEEG is experimental; fNIRS is an optional legacy extension. |
| Adapters | MNE/WFDB objects and legacy samples are converted at the edge and do not enter the core. |
| Modeling | V4 distinguishes encoder, representation, explicit fusion, and task head. |
| Fusion | Optional fusion happens before the task head on a multimodal path. |
| Evidence | Synthetic data is labeled and is not benchmark or clinical evidence. |
| Applications | CLI, UI, and API are clients of research services and core contracts. |

## Important interpretation

The compatibility name `FoundationModel` and the presence of self-supervised-learning components do not establish a validated foundation model. A smoke test, an exported ONNX file, or a Streamlit demonstration does not establish real-data performance.

Use [Migration Notes](migration_notes_v4.md) for v3.3 compatibility, [Scientific Integrity Audit](scientific_integrity_audit_v4.md) for evidence controls, and [Known Limitations](known_limitations_v4.md) before using the platform in a study or publication.
