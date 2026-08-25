# BioSignal-FM V4 Research Chain

BioSignal-FM V4 connects future research questions to testable software boundaries. This page is not a comprehensive literature review or a results announcement; no question becomes a result without real data, a protocol, and reproducible evaluation.

## From question to architecture

| Future question | What V4 provides now | What V4 does not establish |
|---|---|---|
| Can multimodal representation learning improve downstream tasks? | Canonical signal contracts, a modality registry, and replaceable encoders. | Representation quality, transfer, or superiority over a baseline. |
| Does the system generalize across participants or datasets? | LOSO/LODO tools plus provenance and protocol in `RunManifest`. | Actual generalization without real-data execution. |
| Does EMG+EEG or EEG+ECG fusion help? | `ResearchPipeline` enforces fusion before the task head. | Fusion benefit or resilience to a missing modality in a real study. |
| Can the platform extend to ECoG/iEEG? | Experimental adapter/registry with electrode metadata support where supplied by the reader. | ECoG benchmark performance or clinical validity. |

## Correct execution chain

```text
Testable question
  → preregistered protocol
  → real data with license and provenance
  → V4 modality adapter and registry
  → preprocessing fit only on training folds
  → encoder, representation, optional fusion
  → task head and prespecified baselines
  → evaluation at the correct statistical unit
  → RunManifest and rerunnable artifacts
  → honest report of results and limitations
```

This sequence prevents treating windows as independent participants, presenting demo data as a benchmark, hiding late task-head fusion, or converting a code class name into a scientific claim.

## External research systems

BioSignal-FM may consume representations or models from a separate research system through an explicit adapter or contract. Such systems remain external; V4 does not copy their interfaces, training architectures, or deployment stacks into this source tree.

## Evidence progression

| Stage | Required evidence before progression |
|---|---|
| Smoke path | Labeled synthetic data and contract test; no performance claim. |
| Initial real-data experiment | Identified dataset, license, provenance, preprocessing, and recorded protocol. |
| Systematic comparison | Prespecified baselines, valid splits, metrics, confidence intervals, and limitations. |
| Transferable-representation claim | Multiple tasks and participants/datasets with documented transfer evidence. |
| Foundation-model claim | Large documented pretraining data, diverse downstream use, generalization evidence, and reviewable results. |

## Companion documents

- [Research Paper Draft](research/paper_draft.md)
- [Preregistration / Protocol Plan](research/preregistration.md)
- [V4 Architecture](architecture_v4.md)
- [Scientific Integrity Audit](scientific_integrity_audit_v4.md)
- [Known Limitations](known_limitations_v4.md)

V4 supplies the architecture and controls. A scientific answer still requires data, protocol, actual execution, and appropriate review.
