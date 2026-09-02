# Roadmap

This roadmap follows the master specification’s evidence-first sequence. A later phase must not be represented as complete merely because an interface exists; each gate needs scientific, engineering, product, and release evidence.[1]

| Phase | Objective | Gate evidence |
|---:|---|---|
| 0 | Repository and architecture bootstrap. | Package structure, configuration, tests, documentation, CI, licensing, and dependency record. |
| 1 | Core contracts. | Import, schema, registry, validation, and provenance tests pass. |
| 2 | Data adapters. | Independently validated EMG, EEG, ECG, and experimental ECoG adapter paths with provenance and licensing. |
| 3 | Preprocessing contracts. | Explicit, configurable, versioned modality-specific transforms with deterministic fixtures. |
| 4 | Encoders and task heads. | Minimal working shape/contract tests; no premature model-family expansion. |
| 5 | Evaluation and provenance. | Subject-level protocol checks, metric correctness, artifact manifests, and reproducibility rerun. |
| 6 | Multimodal fusion. | Interpretable EMG+EEG baseline, unimodal/fusion/ablation comparison, and documented availability semantics. |
| 7 | Dashboard. | Stable scientific contracts visualized through dataset, protocol, run, metric, artifact, and provenance views. |
| 8 | Packaging and release. | Clean installation, CLI/UI checks, semantic version, changelog, license notices, and release dossier. |
| 9 | Benchmark demonstration. | Real-data EMG and EEG studies; multimodal or missing-modality study only if scientifically ready. |

## Immediate priorities

The next priority is not a larger model. It is a licensed, reproducible real-data validation path for each declared core modality, followed by automatic propagation of dataset/protocol metadata into trainer manifests. Fusion must wait for a compatible synchronized dataset and a pre-defined subject-level evaluation plan.[1]

## References

[1]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
