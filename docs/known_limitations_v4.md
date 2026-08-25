# Known Limitations — BioSignal-FM V4

BioSignal-FM V4 is a research platform with tested software contracts. It is not a completed real-data benchmark suite, a universal biosignal model, or a clinical product.

| Area | Current limitation | Evidence required before a stronger claim |
|---|---|---|
| Real datasets | The release validates contracts and synthetic smoke paths; it does not bundle a licensed real-data benchmark suite. | Dataset manifests, licenses, fixed splits, seeds, and reproducible artifacts per core modality. |
| Foundation-model status | The architecture and components do not establish a broadly validated foundation model. | Documented corpus scale, pretraining protocol, ablations, transfer studies, external validation, and public evaluation artifacts. |
| Clinical use | No diagnostic, treatment, triage, or patient-management intended use is established. | Defined intended use, risk management, clinical evidence, and applicable regulatory work outside this repository. |
| ECoG/iEEG | The adapter is experimental and has no V4 benchmark result. | A selected licensed dataset, electrode/metadata policy, protocol, and real-data evaluation. |
| Missing modalities | Multi-modality support does not prove robustness to absent inputs. | Dedicated missing-modality experiments and a documented behavior contract. |
| Cross-dataset generalization | No external-validation result is included. | Predefined source/target datasets, harmonization, leakage controls, and held-out evaluation. |
| Statistical inference | Metrics and statistical helpers do not define a study protocol. Windows are not independent participants. | Explicit unit of analysis, fitting scope, aggregation, multiple-testing plan, and protocol review. |
| Deployment security | The package does not provide TLS, rate limits, request-size enforcement, tenant isolation, or managed identity. | A hardened operator-managed reverse proxy or gateway and deployment review. |
| Data governance | The package cannot determine whether an external dataset or derived model may be redistributed. | Dataset-specific license, access, ethics, privacy, and model-release review. |
| Performance | Quantization and latency are hardware-dependent. | Target-hardware measurements with a reproducible environment description. |
| Supply chain | CI includes audit controls but the repository has no published release provenance attestation or signing workflow yet. | Protected release workflow, trusted publisher, and attested/signed artifacts. |

These limitations are release constraints, not hidden defects. The prioritized next steps and acceptance evidence are recorded in the [V4 Roadmap](roadmap_v4.md).
