# Encoders

Encoders transform validated, modality-specific model input into reusable `Representation` objects. The architecture treats encoders as replaceable components; the stable asset is the contract between the canonical signal, representation, fusion layer, and task head.[1]

| Interface element | Requirement |
|---|---|
| Input | A validated canonical `Signal` after any declared modality preprocessing. |
| Output | A non-empty immutable one-dimensional `Representation.values` vector. |
| Modality | The encoder declares or emits the source modality where applicable. |
| Context | Representation metadata is immutable and carries explicit `present_modalities` availability context. |
| Training state | Model initialization, checkpoint identity, configuration, and training evidence belong in the run manifest. |
| Versioning | A study must record a stable `model_id` and code commit for every real-data result. |

## Representation-learning scope

The platform supports supervised representations, self-supervised objectives, transfer, and embedding extraction. These capabilities do not constitute evidence of a universal or clinically validated foundation model. Such a claim would require substantial pretraining, reusable representations across multiple downstream tasks, documented transfer evidence, and rigorous benchmark comparisons.[1]

## Implementation boundary

`SignalEncoder` is a small protocol consumed by `ResearchPipeline`. Framework-specific models, including PyTorch models, are application-layer implementations and must not become dependencies of the canonical core contract.[2]

## References

[1]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
[2]: https://github.com/qussaiadlbi/biosignal-fm/blob/main/biosignal_fm/services/research.py "ResearchPipeline encoder protocol"
