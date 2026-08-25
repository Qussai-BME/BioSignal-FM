# Pre-Publication Risk Register — BioSignal-FM V4

**Assessment date:** 22 August 2026  
**Scope:** Source code, packaging, container configuration, CI workflows, documentation, data governance, scientific claims, and dependency security.  
**Disposition:** This is a research-software release assessment. It is not a clinical-validation, medical-device, regulatory, or performance-certification assessment.

## Severity model

| Severity | Meaning | Release treatment |
|---|---|---|
| Critical | Exploitable condition or false public claim that must block public release. | Fix and independently re-verify before release. |
| High | Material security, runtime, data-governance, or claim-integrity weakness. | Fix before release unless a documented compensating control is independently verified. |
| Medium | Important operational or supply-chain weakness with a practical mitigation. | Fix in this release where localized; otherwise disclose with an owner and target. |
| Low | Hardening, observability, or maintainability improvement. | Document and schedule; do not overstate its absence as a security failure. |

## Findings and release decisions

| ID | Area | Finding | Severity | Release decision |
|---|---|---|---|---|
| SEC-01 | Model registration | The REST registration endpoint accepted an authenticated server-side path without enforcing the registry directory stated in the deployment guidance. | High | Restrict registration to a configured model directory and reject absolute or escaping request paths. |
| SEC-02 | WebSocket authentication | The streaming endpoint placed the API key in a query string, exposing credentials to access logs, browser history, and intermediary telemetry. | High | Move authentication into the first WebSocket message and remove query-string credential support. |
| SEC-03 | Inference input validation | REST and WebSocket inference checked channel count but not the registered sample length or finite numeric values. | High | Enforce both dimensions and reject NaN/Inf before inference. |
| SEC-04 | Container runtime | The Docker image installed the legacy `fm` extra but not the documented FastAPI or Streamlit runtime extras, so public API/UI startup was not guaranteed by the image definition. | High | Install the runtime extras explicitly and add an isolated runtime import smoke test. |
| SEC-05 | Compose secret default | Compose provided a predictable fallback API key (`change-me-in-production`). | High | Require `BSFM_API_KEY` at compose interpolation time; add a safe environment example only. |
| SEC-06 | Dependency audit | `pip-audit` found `cryptography 49.0.0` vulnerable to PYSEC-2026-3552 / CVE-2026-69247; fixed upstream in `cryptography 50.0.0`. The vulnerable package is pulled through MLflow/Google Auth in the legacy `fm` bundle. | High | Constrain `cryptography>=50.0.0` in optional bundles that include MLflow; rerun audit in the clean release environment. |
| SEC-07 | Public security policy | `SECURITY.md` still declared pre-1.0 / `0.1.x` support and claimed that client-supplied checkpoint paths were never loaded. | High | Align the policy with 4.0.0 and the enforced configured-directory model. |
| SCI-01 | Public claims | Container metadata, API metadata, compliance text, and the showcase contained unverified foundation-model or zero-shot language. | High | Replace with an accurate research-platform description and an explicit non-clinical intended-use statement. |
| SCI-02 | Dataset governance | Documentation acknowledged synthetic data but did not provide a release-level policy for credentialed data, dataset manifests, or prohibited data egress. | High | Publish a data-governance policy and reference the external evidence register. |
| OPS-01 | Network binding | CLI and deployment configuration defaulted to `0.0.0.0`, which Bandit reports as a medium-severity exposure pattern. | Medium | Default to loopback, require an explicit public-bind flag, and document reverse-proxy/rate-limit requirements. |
| OPS-02 | CI supply chain | CI uses mutable action tags and has no dependency-vulnerability or secret-scanning workflow. | Medium | Pin existing actions to reviewed SHAs, add dependency audit and secret scanning, and document release provenance. |
| OPS-03 | Error disclosure | The catch-all REST exception response returned the internal exception type. | Medium | Return a stable generic error only; log internal detail server-side. |
| DOC-01 | Language readiness | Public V4 release documents and README were not consistently English. | Medium | Rewrite public V4 documentation in English and validate the strict documentation build. |
| DOC-02 | Traceability | Documentation stated some checks were verified but lacked a unified public evidence register for current literature and release controls. | Medium | Publish the external evidence register and a final audit report with command-level results. |
| LOW-01 | Static analysis noise | Bandit reported 31 low-severity findings, primarily controlled subprocess calls and `assert` use in internal runtime code. | Low | Review individually; retain justified controls and avoid representing a raw Bandit exit code as an exploitable finding. |

## External calibration

The release decisions above are aligned with the external evidence register. Contemporary biosignal-FM literature treats benchmark design, dataset scale, interpretability, reproducibility, and missing modalities as open evaluation problems, not guarantees supplied by an architecture.[1] [2] Official GitHub guidance recommends immutable action pinning and least-privilege workflows.[3] PyPI recommends Trusted Publishing for CI-based uploads when supported.[4] The FDA and European Commission sources reinforce the need to keep clinical and medical intended-use claims outside this research release.[5] [6]

## References

[1]: references_prepublication_review_2026.md#references "External Evidence Register — Biosignal research sources"
[2]: https://arxiv.org/abs/2504.19596 "Jiang et al., multimodal physiological foundation models"
[3]: https://docs.github.com/en/actions/reference/security/secure-use "GitHub Secure Use Reference"
[4]: https://pypi.org/help/#trusted-publishers "PyPI Trusted Publishers"
[5]: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software "FDA Clinical Decision Support Software Guidance"
[6]: https://health.ec.europa.eu/ehealth-digital-health-and-care/artificial-intelligence-healthcare_en "European Commission: Artificial Intelligence in Healthcare"
