# Multimodal Learning

BioSignal-FM permits multimodal learning only through an explicit representation-fusion step. The canonical order is `Signal → modality preprocessing → encoder → representation → optional fusion → task head`. A batch with more than one representation raises an error unless a `FusionStrategy` is provided.[1]

| Capability | Current platform state | Evidence boundary |
|---|---|---|
| Unimodal representation path | Implemented | Contract and pipeline tests verify one modality at a time. |
| Explicit pre-head fusion | Implemented | A caller supplies a fusion strategy; task heads do not silently merge raw modality features. |
| Presence context | Implemented | Every encoded and task-input representation carries immutable `present_modalities`. |
| EMG+EEG scientific fusion baseline | Not yet benchmarked | Requires synchronized real data, a locked protocol, and subject-level comparison. |
| Missing-modality robustness | Not established | Presence context is necessary but does not constitute a tested robust-inference method. |
| Zero-filled substitution | Prohibited as an implicit default | Any imputation/ablation policy must be declared and evaluated. |

## Fusion design

Candidate strategies include early, intermediate, late, gated, attention-based, and missing-modality-aware approaches. The first scientific baseline should be interpretable, modest, and compared against each component modality. The framework must not implement every strategy prematurely or infer robustness merely because it accepts different modality sets.[2]

## Availability semantics

The `present_modalities` field tells a fusion strategy and task head which modalities were actually observed. It distinguishes, for example, EMG-only, EEG-only, and EMG+EEG input without relying on hidden shape conventions or silently replacing an absent signal with zeros.[1]

## References

[1]: https://github.com/qussaiadlbi/biosignal-fm/blob/main/biosignal_fm/services/research.py "ResearchPipeline and Representation"
[2]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
