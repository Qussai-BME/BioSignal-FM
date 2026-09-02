# Provenance

A provenance manifest is part of the scientific result, not an optional log. `RunManifest` records a run UUID, timestamp, Git head and dirty state, environment fingerprint, configuration and hash, seed, metrics, output hashes, dataset provenance, protocol, runtime context, and optional model identifier.[1]

## Run and experiment identity

| Identifier | Meaning |
|---|---|
| `run_id` | A UUID for one concrete execution. It changes for every run. |
| `experiment_id` | A deterministic hash of the configuration, data provenance, protocol, model identifier, seed, Git head, and environment. Equivalent scientific definitions resolve to the same identity. |
| `config_hash` | A stable hash of the serialized configuration payload. |
| Output hashes | SHA-256 hashes of registered artifacts such as checkpoints, tables, predictions, and manifests. |

## Real-data research-record check

`research_readiness_issues` exposes missing evidence needed for a reproducible real-data result. `validate_research_readiness()` rejects an incomplete record. A complete record requires dataset identity, version, license, origin, model identifier, protocol identifier, split, metrics, unit of analysis, and a preprocessing definition. Synthetic smoke paths may remain useful for development but are not eligible for benchmark claims.[2]

## Artifact policy

Artifacts can include predictions, embeddings, checkpoints, figures, tables, reports, metrics, and manifests. Protected raw data, credentials, and sensitive identifiers must not be committed to the repository or exported as public artifacts.[2]

## References

[1]: https://github.com/qussaiadlbi/biosignal-fm/blob/main/biosignal_fm/reproducibility.py "RunManifest"
[2]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
