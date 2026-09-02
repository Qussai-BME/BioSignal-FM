# Privacy and GDPR Research-Use Position

**Status:** Research-software policy note; not legal advice and not a declaration of GDPR compliance.  
**Applies to:** BioSignal-FM source code, documentation, packaged examples, and local research workflows.

> BioSignal-FM is distributed as research software. The repository does not intentionally bundle raw participant recordings, account telemetry, or user analytics. That fact does **not** establish that every dataset, feature, model, log, or deployment created with the software is non-personal data.

## Scope and responsibilities

Whether GDPR applies depends on the particular data, deployment, parties, purposes, and technical/organizational safeguards. Physiological signals, associated metadata, derived features, identifiers, access logs, and trained artifacts can carry personal-data or re-identification risk depending on context. The person or institution deciding to collect, access, upload, train, deploy, or disclose data remains responsible for determining its lawful basis, roles, safeguards, and applicable obligations.

| Situation | Repository position | Required operator action |
|---|---|---|
| Source code, synthetic fixtures, and documentation | No telemetry is initiated by the package; synthetic examples remain non-benchmark material. | Keep fixtures labeled synthetic and do not attach participant information. |
| Open research data | The repository does not treat public availability as unrestricted processing or redistribution permission. | Record dataset version, license, citation, access date, provenance, and applicable terms. |
| Credentialed or restricted data | Never commit, bundle, upload to public demos, or route to third-party services. | Use only an approved environment with access controls and a non-sensitive manifest reference. |
| Local participant or institutional data | Never commit raw data or identifiers. | Use the institution's ethics/governance process, consent or other lawful basis, retention plan, access controls, and de-identification review. |
| Derived models, features, reports, or logs | Treat as potentially sensitive when trained on restricted or participant data. | Review disclosure risk and license terms before sharing, publishing, or deploying. |

## Engineering controls present in this release

The software is designed to support minimization and traceability rather than to certify compliance. It records code and experiment provenance through manifests, keeps raw data outside the repository by policy, uses synthetic data only for tests/demos, and provides documented boundaries for open, credentialed, restricted, and local data. These are safeguards and workflow aids; they are not a substitute for a data-protection assessment.

## Operator checklist before a data-bearing run

Before processing any non-synthetic dataset, document the dataset/participant source, version, license or access authority, processing purpose, modality and metadata fields, retention/deletion plan, access-control model, disclosure risk for outputs, and protocol/split/provenance record. Do not expose raw data, credentials, sensitive paths, access logs, or data-use agreements in source control, public dashboards, example commands, or support channels.

## Related controls

The detailed repository policy is in [`docs/data_governance.md`](../data_governance.md). It defines dataset onboarding, prohibited actions, credentialed-data controls, and deployment safeguards. A future data-bearing public service requires a separate deployment threat model and privacy/legal review; a local research default is not a public-service compliance boundary.

## Publication boundary

No sentence in this document grants data rights, determines controller/processor status, or makes a clinical, regulatory-clearance, or privacy-compliance claim. Have qualified privacy/legal counsel review any consequential processing, sharing, institutional submission, or deployment decision.
