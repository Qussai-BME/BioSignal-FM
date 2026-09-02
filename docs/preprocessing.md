# Preprocessing

Preprocessing is explicit, configuration-driven, modality-specific, and recorded as provenance. BioSignal-FM does not provide a generic `preprocess(signal)` that silently applies EMG assumptions to all data.[1]

## Processing layers

| Layer | Purpose | Boundary |
|---|---|---|
| Raw acquisition data | Preserves the source signal and source metadata. | Must not be overwritten by a downstream transform. |
| Acquisition normalization | Resolves source format, channel mapping, units, and sample-rate representation. | Performed by adapters with explicit provenance. |
| Modality preprocessing | Applies configured filtering, resampling, and normalization. | Must be fitted only on the protocol-appropriate training scope. |
| Model input | Represents the final validated tensors/windows presented to an encoder. | Must retain an auditable link to preprocessing history. |

`PreprocessingConfig` validates positive sample rates, ordered positive passbands, notch settings, filter order, valid window overlap, and related scientific settings at construction time. `PreprocessingPipeline` supports modality-specific filtering, resampling, and subject-aware normalization.[2]

## Canonical signal path

After fitting on training data, `PreprocessingPipeline.transform_signal()` accepts a canonical `Signal`, applies the configured transform, changes the sampling rate, creates a configuration hash, appends a versioned `SignalProcessingStep`, and marks the result `preprocessed`. If a source has a missingness mask, the method requires a study-specific resampling policy instead of silently interpolating booleans. When source timestamps are present, it generates an explicit output time basis over the original interval.

> A preprocessing change is part of the experiment definition. It must create a different configuration hash and therefore a different experiment identity.

## Protocol controls

Researchers must document filtering, resampling, normalization fitting scope, epoch/window policy, artifact handling, and calibration policy in the study protocol. The test participant or held-out fold must never influence fitted preprocessing statistics.[1]

## References

[1]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
[2]: https://github.com/qussaiadlbi/biosignal-fm/blob/main/biosignal_fm/preprocessing/pipeline.py "PreprocessingPipeline"
