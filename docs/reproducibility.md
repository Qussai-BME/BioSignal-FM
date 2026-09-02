# Reproducibility

Reproducibility requires a rerunnable scientific definition, not only a saved model file. BioSignal-FM provides global seed controls, immutable configuration serialization, environment fingerprints, Git-state capture, artifact hashing, protocol capture, dataset provenance, and deterministic experiment identity.[1]

| Required record | Platform mechanism |
|---|---|
| Random seed | `set_global_seed()` and manifest `seed`. |
| Configuration | Frozen dataclasses, YAML serialization, and `config_hash`. |
| Code state | Git head and dirty-state capture. |
| Environment | Package list and runtime context fingerprint. |
| Dataset provenance | Dataset identity, version, license, origin, adapter, and notes. |
| Protocol | Split, metrics, unit of analysis, preprocessing scope/version, and calibration definition. |
| Model identity | Stable `model_id` plus model configuration/checkpoint artifact. |
| Outputs | SHA-256 artifact hashes, registered metrics, predictions, figures, and reports. |

## Protocol discipline

A configuration change that alters scientific behavior must produce a different experiment identity. A held-out test participant, session, or fold cannot be used to fit normalization, tune hyperparameters, choose a checkpoint, or otherwise influence the training decision. Reproducibility therefore includes leakage prevention and documented fitting scope, not just repeatable random numbers.[2]

## Development versus evidence

Synthetic runs are valuable for smoke tests and interface validation. They must remain labeled synthetic and must not be reported as a benchmark, baseline improvement, scientific inference, or clinical result. A real-data study must call `validate_research_readiness()` before it is treated as evidence.[2]

## References

[1]: https://github.com/qussaiadlbi/biosignal-fm/blob/main/biosignal_fm/reproducibility.py "Reproducibility utilities"
[2]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
