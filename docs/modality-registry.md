# Modality Registry

BioSignal-FM uses a single in-process modality registry to keep reusable infrastructure separate from modality-specific scientific behavior. Each `ModalityPlugin` declares an identifier, maturity, adapter factory, preprocessing factory, optional encoder factory, supported tasks, visualization capabilities, data references, optional dependencies, and explanatory notes.[1]

| Registry operation | Behavior |
|---|---|
| Register | Adds a uniquely identified modality plugin and rejects accidental replacement by default. |
| Resolve | Returns a plugin by normalized modality identifier. |
| Validate | Rejects canonical signals whose modalities are not registered. |
| List | Returns stable registered identifiers or plugin records, optionally filtered by maturity. |
| Export | Produces a JSON-friendly capability summary for reports, CLI inspection, and UI use. |

## Current matrix

| Modality | Maturity | Architectural declaration | Public boundary |
|---|---|---|---|
| EMG | Core | Adapter, preprocessing factory, classification, regression, and representation-learning tasks. | Requires a locked real-data protocol for performance claims. |
| EEG | Core | Adapter, MNE-facing edge integration, preprocessing factory, classification, and representation learning. | MNE/BIDS remain edge concerns; no BCI-generalization claim is implied. |
| ECG | Core | Independent adapter/preprocessing path plus classification, rhythm analysis, and representation learning. | ECG is not an EMG alias and carries no diagnostic claim. |
| ECoG/iEEG | Experimental | Adapter and representation-learning extension path. | No benchmark or mature-support claim. |
| fNIRS | Legacy optional | Compatibility extension. | Not part of the V4 core definition. |

## Adding a modality

A new modality must be registered through a plugin rather than through a separate application. Its contribution must include an adapter, required metadata and provenance expectations, modality-specific preprocessing configuration, declared encoder/task capabilities, contract tests, dataset/license handling, and honest maturity documentation. A plugin does not grant a benchmark or foundation-model claim by itself.[2]

## References

[1]: https://github.com/qussaiadlbi/biosignal-fm/blob/main/biosignal_fm/modalities/registry.py "Modality registry implementation"
[2]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
