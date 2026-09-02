# BioSignal-FM — Research-to-Translation Considerations

**Version:** 2.0  
**Date:** 2026-08-26  
**Status:** Research planning note; not a compliance matrix, legal opinion, quality-system record, regulatory submission, or clinical claim.

> BioSignal-FM `4.0.2` is research software. It is not a medical device, not clinically validated, not cleared or approved for diagnosis/treatment/patient management, and not represented as conforming to any regulatory framework by virtue of this repository.

## Current release boundary

This release provides modular biosignal research infrastructure, provenance-aware experimentation, testing, documentation, release records, and local deployment tooling. It does not provide a regulated intended use, a clinical performance claim, a medical-device quality management system, a clinical risk-benefit determination, or evidence sufficient to assign a regulatory risk class.

| Present engineering control | Release value | Explicit non-claim |
|---|---|---|
| Versioned source, contracts, manifests, checksums, and release documentation | Reproducible and auditable research work. | Formal design-history, configuration-management, or regulatory record sufficiency. |
| Unit/integration testing, CI, dependency controls, and security guidance | Software-quality and supply-chain hygiene. | Safety certification, cybersecurity certification, or fitness for patient-facing deployment. |
| Dataset/provenance and research-use policy | More transparent treatment of data rights, evidence class, and study context. | Permission to process data, proof of de-identification, or privacy compliance. |
| Research-only labeling and limitation statements | Honest communication of evidence boundaries. | Clinical validation, diagnostic accuracy, foundation-model validation, or regulatory readiness. |

## Future translation gate

Any organization considering a clinical, medical-device, or otherwise regulated use must create a **separate product program**. Before making an external compliance or safety statement, that program would need to establish and independently review at least the following categories.

| Gate | Evidence needed before a consequential claim |
|---|---|
| Intended use and accountability | Precise indication/use environment, target users/populations, and responsible legal entities. |
| Quality and lifecycle process | Controlled requirements, design history, verification/validation, change control, supplier management, complaint/problem-resolution, and release governance. |
| Safety and risk management | Hazard analysis, risk controls, residual-risk evaluation, and verification of those controls for the actual product. |
| Data and privacy governance | Authorized data basis, governance roles, retention/security controls, disclosure-risk evaluation, and deployment-specific privacy review. |
| Clinical/scientific validation | Protocol-appropriate, authorized evidence for the claimed population, task, comparator, endpoints, and limitations. |
| Cybersecurity and operations | Threat model, secure deployment architecture, access/logging controls, vulnerability process, monitoring, incident response, and update policy. |
| Regulatory pathway | Qualified review of the applicable jurisdiction, product classification, standards, submissions, and post-market obligations. |

## Documentation rule

Future work may use this repository's technical artefacts as inputs, but must not repurpose research benchmarks, synthetic fixtures, or development controls as clinical or regulatory evidence. Every downstream claim must identify the product version, intended use, evidence package, protocol, data authority, and responsible organization.

## References and companion policies

This note is intentionally limited. The source repository’s data practices are in [`docs/data_governance.md`](../data_governance.md), privacy positioning is in [`gdpr.md`](gdpr.md), and the V4 scientific-release boundary is in [`RELEASE_MANIFEST.md`](../../RELEASE_MANIFEST.md). Any legal, regulatory, or clinical decision should be reviewed by appropriately qualified professionals before reliance.
