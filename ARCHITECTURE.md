# BioSignal-FM — Architecture Reference

**Authoritative architecture release:** V4  
**Document status:** Concise entry point to [V4 Architecture](docs/architecture_v4.md)  
**Project license:** Apache License 2.0

## Purpose

This document is the short entry point for BioSignal-FM architecture. The detailed authority is [V4 Architecture](docs/architecture_v4.md); compatibility guidance is in [Migration Notes](docs/migration_notes_v4.md).

BioSignal-FM is a modular multimodal biosignal research platform. The repository alone does not establish a trained and validated foundation model, state-of-the-art performance, clinical readiness, or regulatory clearance.

## Approved architecture

```text
Dataset source or reader at the edge
        ↓
Registered modality adapter
        ↓
Canonical Signal with SignalMetadata and SignalProvenance
        ↓
Modality-specific preprocessing
        ↓
Encoder → Representation → Optional Fusion → Task Head
        ↓
Protocol-aware evaluation + RunManifest
```

| Area | V4 decision |
|---|---|
| Core modalities | EMG, EEG, and ECG. |
| Experimental modality | ECoG/iEEG through an adapter and registry entry; no benchmark claim. |
| Optional extension | fNIRS is legacy-compatible and does not define the core. |
| Core | Independent of MNE, WFDB, PyTorch, UI, and HTTP libraries. |
| Synthetic data | Allowed for smoke tests and demonstrations; explicitly benchmark-ineligible. |
| Multimodal fusion | Optional, but applied before the task head in the default research flow. |
| Evaluation | LOSO/LODO, metrics, and statistics remain tools whose validity depends on the study protocol. |
| Documentation | Must describe available evidence, not future goals or unverified commercial/clinical plans. |

## Related governance documents

- [Detailed V4 Architecture](docs/architecture_v4.md)
- [Migration Closure Audit](docs/migration_audit_v4.md)
- [Migration Notes](docs/migration_notes_v4.md)
- [Scientific Integrity Audit](docs/scientific_integrity_audit_v4.md)
- [Known Limitations](docs/known_limitations_v4.md)
- [Dependency and License Inventory](docs/dependency_license_inventory_v4.md)
- [Data Governance](docs/data_governance.md)
- [Pre-Publication Risk Register](docs/prepublication_risk_register_2026.md)

Any older document that describes V3.3 as a unified foundation model or promises benchmark, commercial, or clinical outcomes is historical reference only and does not supersede this reference or the approved migration specification.
