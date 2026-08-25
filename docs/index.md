# BioSignal-FM V4 Documentation

BioSignal-FM is a modular multimodal biosignal research platform. It supports reproducible paths from a signal source through preprocessing, representation learning, optional fusion, task heads, and protocol-aware evaluation while retaining data provenance and protocol context in a run manifest.

> **Evidence boundary:** This documentation does not describe the repository as a validated foundation model, medical device, or performance benchmark. Those claims require real data, a documented protocol, reproducible results, and appropriate independent evidence.

## Start here

| Document | Use |
|---|---|
| [Project README](https://github.com/qussaiadlbi/biosignal-fm#readme) | Installation, extras, CLI, secure local service mode, and synthetic-data policy. |
| [V4 Architecture](architecture_v4.md) | Contracts, modality registry, dependency boundaries, and fusion flow. |
| [Migration Notes](migration_notes_v4.md) | Move from v3.3 to V4 while preserving supported compatibility paths. |
| [Migration Closure Audit](migration_audit_v4.md) | Engineering decisions and historical closure evidence. |
| [Scientific Integrity Audit](scientific_integrity_audit_v4.md) | Data provenance, study constraints, and permitted claim language. |
| [Data Governance](data_governance.md) | Dataset licenses, credentialed-data handling, and experiment onboarding gate. |
| [Pre-Publication Risk Register](prepublication_risk_register_2026.md) | Release findings, severity, and remediation decisions. |
| [External Evidence Register](references_prepublication_review_2026.md) | Current research, regulatory, data, and supply-chain sources. |

## Modalities

| Modality | V4 status | Note |
|---|---|---|
| EMG | Core | Preserves the mature v3.3 path through a formal adapter. |
| EEG | Core | Integrates MNE/BIDS-compatible concepts at the edge. |
| ECG | Core | Has a distinct preprocessing and representation path. |
| ECoG/iEEG | Experimental | Adapter and extension contract only; no benchmark result. |
| fNIRS | Optional legacy | Retained for compatibility and does not define the V4 core. |

## Research path

```text
Dataset reader → modality adapter → canonical Signal → modality preprocessing
→ encoder → representation → optional fusion → task head → protocol-aware evaluation → RunManifest
```

Use `bsfm inspect` to view the registry without loading data. Use `bsfm pretrain --synthetic-demo` only when you intentionally need a clearly labeled synthetic technical smoke path.

## Further references

- [Quickstart](quickstart.md)
- [API Reference](api_reference.md)
- [Deployment](deployment.md)
- [Research Chain](research_chain.md)
- [Informational Regulatory Appendix](compliance/regulatory_appendix.md)
- [Known Limitations](known_limitations_v4.md)
- [V4 Roadmap](roadmap_v4.md)

## External resources

- [GitHub repository](https://github.com/qussaiadlbi/biosignal-fm)
- [Issues](https://github.com/qussaiadlbi/biosignal-fm/issues)
