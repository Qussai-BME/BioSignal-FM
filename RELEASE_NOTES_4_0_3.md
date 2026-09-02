# BioSignal-FM 4.0.3 — Professionalization Patch Release

**Release date:** 2026-08-26  
**Release class:** Research-software patch release.

## Scope

Version 4.0.3 replaces the local 4.0.2 working distribution after publication-quality remediation. It improves privacy/regulatory wording, static type correctness at NumPy and sampling-rate boundaries, source formatting, Git traceability, and release reproducibility. It does not introduce new training data, fitted models, benchmarks, fusion results, clinical validation, or regulatory evidence.

| Area | Change | Claim boundary |
|---|---|---|
| Privacy and governance | Rewrote GDPR, EU AI Act, FDA/GMLP, and regulatory appendix pages as research-use positions rather than self-classifications. | No legal advice, compliance declaration, or clinical/regulatory readiness claim. |
| Type safety | Normalized synthetic array returns and added explicit integral sampling-rate validation at the preprocessing boundary. | No change to signal semantics or research protocol. |
| Quality | Applied repository formatting; static analysis and tests were rerun in the delivery environment. | Passing software checks do not establish scientific or clinical performance. |
| Release integrity | Package/version/CITATION identity is 4.0.3 and a clean local Git source revision is recorded in the delivery manifest. | A publisher must still create an official remote, signed tag, and archival DOI if desired. |

## Scientific boundary

BioSignal-FM remains a research-only multimodal biosignal platform. It does not claim a validated universal foundation model, cross-modal generalization, diagnostic performance, clinical validity, regulatory clearance, or benchmark superiority without separate authorized evidence.

## Upgrade note

The update is source-compatible for the documented API. `PreprocessingPipeline.transform_signal()` now rejects a non-integral sampling rate before invoking preprocessing components that require integer resampling factors; callers should supply a rate compatible with their filter/resampling protocol.
