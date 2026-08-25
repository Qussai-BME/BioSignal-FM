# Data Governance and Research Use Policy

## Scope and intended use

BioSignal-FM is distributed as **research software**. It is not intended to diagnose, treat, monitor, or make decisions about an individual patient. The repository may support research workflows involving physiological signals, but users are responsible for lawful access, ethical approval where required, participant protections, and compliance with the terms of each dataset.

> No dataset, model artifact, interface output, or documentation page in this repository establishes clinical validity, regulatory clearance, or permission to use a dataset beyond its license.

## Data classes

| Data class | Repository policy | Required provenance |
|---|---|---|
| Synthetic development data | Allowed for tests, demos, and smoke paths only. Must remain labeled synthetic and benchmark-ineligible. | Generator, seed where applicable, fallback reason, and `benchmark_eligible: false`. |
| Open research data | Not bundled unless redistribution is explicitly permitted. Download and citation instructions must identify the dataset version and license. | Dataset DOI/URL, version, license, access date, processing adapter, and split manifest. |
| Credentialed or restricted data | Never commit, bundle, publish, upload to demos, or route through third-party APIs. Access remains with the authorized individual or institution. | Access basis, dataset version, approval/credential context where appropriate, and a non-sensitive manifest reference. |
| Local participant or institutional data | Never commit to the repository. Process only in an approved environment with an institution-specific governance plan. | Protocol/approval identifier where permissible, consent basis, retention policy, access controls, and de-identification status. |
| Derived models and features | Treat as potentially sensitive when trained on restricted or participant data. Do not publish them until the dataset terms and disclosure risk have been reviewed. | Training-data class, dataset manifest, code revision, configuration hash, evaluation protocol, and release decision. |

## Prohibited actions

The following actions are prohibited in this repository and its public demos:

1. Committing raw biosignal recordings, identifiers, credential files, access tokens, or data-use agreements.
2. Sharing access to credentialed data, including through a hosted inference API, a notebook upload, a public URL, or a model-registration path.
3. Presenting synthetic output as real-data performance, a clinical result, or a benchmark.
4. Training or evaluating on a real dataset without recording its version, license, preprocessing scope, and split protocol.
5. Exporting models trained on restricted data before confirming that the applicable terms permit model release and that privacy risk has been assessed.

## Dataset onboarding gate

A dataset is not ready for a project experiment merely because a loader can read it. Before onboarding, create a dataset manifest outside raw data storage and confirm each item below.

| Gate | Required evidence |
|---|---|
| Legal access | Dataset license or data-use agreement, redistribution status, and required citation. |
| Governance | Ethics/IRB context or explicit public-data basis where applicable. |
| Versioning | Dataset version, source URL/DOI, access date, and file checksums where lawful and practical. |
| Metadata | Modality, sampling rate, channels, units, participant/session/recording identifiers, and label semantics. |
| Privacy | De-identification review, access controls, retention plan, and no public egress path. |
| Protocol | Unit of analysis, train/validation/test split, leakage controls, preprocessing fit scope, metrics, and seed policy. |
| Provenance | `SignalProvenance` and `RunManifest` fields sufficient to reconstruct the experiment without exposing restricted content. |

## Credentialed-data controls

PhysioNet's credentialed-data license requires that access not be shared and that reasonable physical and electronic safeguards be maintained.[1] For credentialed resources, run loaders only in an approved local or institutional environment. Do not mount those datasets into a public container, publish them through Streamlit or FastAPI, place their paths in example commands, or send samples to external services.

Open-access status is dataset-specific. For example, the MIT-BIH Arrhythmia Database page lists an open-data license and version information, but an experiment must still record the version, citation, and any workflow-specific constraints.[2]

## Deployment controls for data-bearing workloads

The public Compose configuration mounts `./data` and `./checkpoints` read-only and binds the API to loopback. This is a baseline hardening pattern, not a compliance boundary. Any network-facing deployment requires TLS, a reverse proxy, rate limits, API-key lifecycle management, access logging reviewed for sensitive content, and a separate data-risk review.

## References

[1]: https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/ "PhysioNet Credentialed Health Data License 1.5.0"
[2]: https://physionet.org/content/mitdb/ "PhysioNet, MIT-BIH Arrhythmia Database"
