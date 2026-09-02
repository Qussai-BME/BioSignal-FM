# Signal Contract

The canonical signal contract is the boundary between dataset-specific readers and reusable BioSignal-FM research services. It is implemented without MNE, WFDB, PyTorch, FastAPI, or UI dependencies, so modality adapters must convert native data before it enters the research pipeline.[1]

| Type | Responsibility | Required scientific safeguards |
|---|---|---|
| `SignalMetadata` | Declares modality, sampling rate, channel names, units, subject/session/recording/task context, and processing status. | A positive sampling rate, unique channels, and non-empty processing status are validated at construction. |
| `SignalProvenance` | Records origin, dataset/version/license, adapter identity, fallback reason, and immutable details. | Synthetic origin is explicit and cannot be silently treated as benchmark evidence. |
| `SignalProcessingStep` | Records one named, versioned transformation with a configuration hash and parameters. | Processing history is immutable and append-only through `with_processing_step()`. |
| `Signal` | Carries immutable `(channels, samples)` data, metadata, timestamps, events, and an optional missingness mask. | Channels must match metadata; timestamps are strictly increasing; data and masks are copied read-only. |
| `SignalBatch` | Groups non-empty canonical signals for a pipeline operation. | Exposes modalities and synthetic-data presence without relying on filenames. |

## Transformations

`Signal.with_data()` preserves time and missingness semantics only when they remain valid. A shape-changing operation on a signal with timestamps or a missingness mask must supply explicit replacements; otherwise it raises an error. A caller can also supply a `SignalProcessingStep` and a new processing status, which appends immutable provenance rather than silently modifying a signal.

> A canonical transformation must never retain stale timestamps, masks, or preprocessing metadata after changing the sample axis.

## Adapter boundary

Dataset adapters own conversion of external formats to `Signal`. They must declare dataset identity, version, license, channel mapping, source sample rate, labels, subject/session mapping, and provenance. Core consumers must use contract fields rather than filename conventions.[2]

## References

[1]: architecture_v4.md "BioSignal-FM V4 Architecture"
[2]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
