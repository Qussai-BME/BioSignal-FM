# FDA / GMLP Research-Use Note

**Status:** Informational engineering note; not FDA guidance, legal advice, a medical-device determination, or evidence of conformance.  
**Release scope:** BioSignal-FM `4.0.2` research software.

> BioSignal-FM is not FDA-cleared or approved Software as a Medical Device and is not intended for diagnosis, treatment, patient management, or clinical decision-making. Repository engineering practices do not establish a clinical or regulatory claim.

## Research practices that can support future evidence work

The project uses modular design, versioned configurations, run/artifact provenance, tests, CI, release records, and explicit data-governance limitations. These practices can improve reproducibility and auditability of research. They should not be described as GMLP compliance, medical-device lifecycle evidence, clinical validation, or proof of a performance threshold.

| Current research practice | Appropriate statement | Inappropriate statement |
|---|---|---|
| Subject-aware evaluation and declared split rules | “The protocol reports research performance under its documented split.” | “The model is clinically accurate or usable.” |
| Synthetic test/demo fixtures | “Synthetic fixtures verify code paths and remain benchmark-ineligible.” | “Synthetic outputs validate a decoder, treatment, or safety outcome.” |
| Run manifests, hashes, and environments | “The release improves experiment traceability.” | “The release satisfies regulated record-keeping.” |
| CI, linting, tests, and dependency controls | “The release applies software-quality controls.” | “The system is certified safe, secure, or regulatory-ready.” |

## Requirements before any clinical or regulated claim

A future product effort would need a defined intended use, accountable organization, appropriately governed and authorized data, a controlled quality/risk-management process, human-factors and cybersecurity evidence where applicable, validation for the claimed users and context, and qualified legal/regulatory review. Performance thresholds are not inserted here because an appropriate threshold is claim-, population-, task-, comparator-, and risk-dependent.

## Release statement

Do not cite this page as a clearance, approval, GMLP conformance declaration, SaMD classification, clinical endpoint definition, or regulatory roadmap. Use [`docs/data_governance.md`](../data_governance.md), [`gdpr.md`](gdpr.md), and [`RELEASE_MANIFEST.md`](../../RELEASE_MANIFEST.md) to understand the present research boundary.
