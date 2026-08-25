# Research Draft Template: BioSignal-FM V4

**Status:** Research-plan and evaluation template; not a results paper or validated-foundation-model claim.  
**Release:** V4

## Proposed abstract

BioSignal-FM V4 is a modular multimodal biosignal research platform that standardizes signal contracts, data provenance, modality registration, and testable research services. It supports core EMG, EEG, and ECG paths, an experimental ECoG/iEEG adapter, and a legacy-compatible fNIRS extension. The architecture separates source data and adapters, modality-specific preprocessing, encoder, representation, optional fusion, task head, and protocol-aware evaluation.

This template reports no real benchmark result and does not describe the platform as a validated foundation model. Bundled synthetic paths serve software verification and technical demonstration only; they cannot support claims of performance, generalization, zero-shot transfer, or clinical utility.

## Future research questions

| ID | Testable question | Required before an answer |
|---|---|---|
| RQ1 | Can a multimodal encoder learn useful representations for multiple tasks? | Licensed real data, documented training, and strong baselines. |
| RQ2 | Does generalization improve across participants or datasets? | Defined LOSO/LODO protocol, correct analysis unit, and leakage prevention. |
| RQ3 | Does EMG+EEG or EEG+ECG fusion improve on unimodal paths? | Real synchronized modalities, defined fusion, and missing-modality analysis. |
| RQ4 | Can ECoG/iEEG be supported? | Selected real dataset, electrode description, license, and independent protocol. |

## Architectural methodology

```text
Data reader → modality adapter → canonical Signal
→ modality-specific preprocessing → encoder → representation
→ optional fusion → task head → protocol-aware evaluation → RunManifest
```

Every experiment should record data source/version/license where known, preprocessing, seed, Git state, configuration hash, protocol, metrics, and practical output hashes. MNE and WFDB remain adapter-layer dependencies, not core dependencies.

## Proposed evaluation design

| Area | Requirement |
|---|---|
| Data | Real datasets recorded in separate manifests with license, version, and processing path. |
| Split | LOSO, LODO, or another split declared before execution; no target-participant exposure beyond the protocol. |
| Processing | Fitted statistics restricted to training data inside each fold. |
| Metrics | Task-appropriate prespecified metrics such as accuracy, macro-F1, calibration, and confidence intervals. |
| Statistics | Participant/dataset-level units for small-sample inference; do not count windows as independent participants. |
| Baselines | Prespecify implementation, version, and settings before inspecting results. |
| Reporting | Separate real benchmark outputs from synthetic diagnostics and technical demonstrations. |

## Synthetic data

The project creates deterministic synthetic samples to test loaders, preprocessing, and interfaces. Every V4 output from this path carries `synthetic` provenance and `benchmark_eligible: false`. Do not include accuracy, rank, or p-values from that data in a scientific abstract or benchmark comparison, even when the test is reproducible.

## Editorial language

| Permitted | Unsupported until evidence is complete |
|---|---|
| “Modular multimodal biosignal research platform” | “Validated foundation model” |
| “Trainable encoder or representation model” | “General transferable representation” |
| “Labeled synthetic technical demonstration” | “Benchmark result” or “superiority” |
| “Experimental ECoG path” | “Benchmarked ECoG support” |
| “Informational regulatory guidance” | “Regulatory readiness or clearance” |

## Requirements before converting this template into a results paper

1. Select real datasets and document licenses, provenance, and versions.
2. Lock protocol, baselines, preprocessing parameters, and stopping criteria.
3. Execute reproducible training and evaluation with complete manifests.
4. Analyze performance differences with correct statistical units and appropriate intervals.
5. Report negative results, limitations, and uncertainty as clearly as positive findings.
6. Conduct internal or independent claim review before external publication.

[V4 Architecture](../architecture_v4.md), [Scientific Integrity Audit](../scientific_integrity_audit_v4.md), and [Known Limitations](../known_limitations_v4.md) supersede this draft where they differ.
