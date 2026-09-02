# Security

BioSignal-FM is research infrastructure and must protect data, credentials, model artifacts, and network-facing interfaces. The platform does not claim regulatory compliance certification; the controls here are engineering and data-governance requirements for responsible research use.[1]

| Area | Required control |
|---|---|
| Raw data | Do not commit protected datasets, restricted subject metadata, credentialed access material, or secret URLs. |
| Provenance | Store public-safe dataset identity and licensing evidence; do not expose sensitive participant identifiers in manifests. |
| Configuration | Keep paths, keys, and deployment secrets out of version-controlled experiment configuration. |
| Service binding | Bind locally by default. Network exposure requires a reverse proxy, TLS, request limits, audit logging, and environment-managed API keys. |
| API authentication | Use the configured API-key mechanism for inference or mutation endpoints; do not place secrets in query parameters. |
| Checkpoints | Restrict registration to operator-staged relative paths inside the configured model directory. |
| Artifacts | Sanitize reports, figures, predictions, and manifests before public release. |
| Dependencies | Maintain license and dependency audit records and update known vulnerable or incompatible components. |

## Data-governance boundary

Every real-data source must have a documented source, version, license, access condition, and intended research use before an adapter or experiment is registered. If access fails, the relevant track must be stopped and documented rather than substituted with synthetic data and presented as evidence.[2]

## References

[1]: data_governance.md "Data Governance"
[2]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
