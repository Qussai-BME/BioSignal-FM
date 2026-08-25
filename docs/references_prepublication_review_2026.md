# External Evidence Register — Pre-Publication Review (2026)

**Review date:** 22 August 2026  
**Purpose:** This register records the external sources used to calibrate BioSignal-FM V4's public claims, data governance posture, and software-release controls. It does not convert the project into a medical device or establish scientific performance.

## Research and evaluation context

The contemporary biosignal foundation-model literature treats **data scale, modality heterogeneity, reproducible evaluation, benchmark design, interpretability, and data availability** as unresolved challenges rather than assumptions that an architecture alone can satisfy.[1] A 2025 multimodal physiological preprint specifically frames arbitrary missing modalities and cross-dataset generalization as material technical problems, motivating explicit missing-modality evaluation rather than generic multimodal claims.[2] A 2025 ECG foundation-model review similarly characterizes the area as emerging and emphasizes the distinction between representation-learning promise and robust, generalizable clinical evidence.[3]

| Source | Review implication for BioSignal-FM |
|---|---|
| Biosignal FM review (2025) | Preserve the research-platform framing; do not claim a validated universal or foundation model without data, benchmarks, and evaluation artifacts. |
| Multimodal physiological FM work (2025/2026 revision) | Add missing-modality behavior to the public evaluation contract and roadmap; do not infer it from support for multiple modalities. |
| ECG-FM review (2025 revision) | Keep clinical claims out of user-facing copy until a study supplies real data, external validation, and an intended-use case. |
| Large-scale wearable biosignal FM study | Scale and consented longitudinal data are central to published foundation-model evidence; a synthetic smoke path is intentionally insufficient. |

## Data governance and health context

The current FDA Clinical Decision Support Software guidance distinguishes certain non-device CDS functions from device functions and clarifies that device software functions remain subject to applicable digital-health policies.[4] The European Commission describes AI software intended for medical purposes as high risk under the EU AI Act context, with risk management, high-quality data, user information, and human oversight among the listed requirements.[5] This project is therefore documented as **research software only**, without a diagnostic, treatment, clinical-decision, patient-facing, or regulated intended use.

PhysioNet's credentialed-data license prohibits sharing restricted-data access and requires appropriate care for physical and electronic security; the project must never bundle, upload, or route credentialed data through public demos or external services.[6] By contrast, the MIT-BIH Arrhythmia Database page identifies the published open-data license and its record structure; dataset-level terms still need to be checked and recorded per experiment.[7]

## Release and supply-chain context

GitHub states that pinning an Action to a full-length commit SHA is the only currently immutable way to reference an Action, and recommends least-privilege tokens, CodeQL, and workflow review controls.[8] PyPI recommends Trusted Publishing where supported, eliminating the need for upload API tokens in CI.[9] SLSA v1.2 provides a provenance-oriented framework for progressively strengthening source and build integrity.[10] OpenSSF Scorecard is a useful diagnostic rather than a release certification; its individual checks should be addressed explicitly instead of optimizing an aggregate score.[11]

## References

[1]: https://www.techrxiv.org/doi/10.36227/techrxiv.176369849.97173246/v1 "Lee et al. (2025), A Comprehensive Review of Biosignal Foundation Models"
[2]: https://arxiv.org/abs/2504.19596 "Jiang et al. (2025; revised 2026), Towards Robust Multimodal Physiological Foundation Models"
[3]: https://arxiv.org/abs/2410.19877 "Han et al. (2024; revised 2025), A Systematic Review on Foundation Models for Electrocardiogram Analysis"
[4]: https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software "FDA Clinical Decision Support Software Guidance, January 2026"
[5]: https://health.ec.europa.eu/ehealth-digital-health-and-care/artificial-intelligence-healthcare_en "European Commission, Artificial Intelligence in Healthcare"
[6]: https://physionet.org/about/licenses/physionet-credentialed-health-data-license-150/ "PhysioNet Credentialed Health Data License 1.5.0"
[7]: https://physionet.org/content/mitdb/ "PhysioNet, MIT-BIH Arrhythmia Database"
[8]: https://docs.github.com/en/actions/reference/security/secure-use "GitHub Docs, Secure Use Reference"
[9]: https://pypi.org/help/#trusted-publishers "PyPI Help, Trusted Publishers"
[10]: https://slsa.dev/spec/v1.2/ "SLSA Specification v1.2"
[11]: https://github.com/ossf/scorecard "OpenSSF Scorecard"

## MLflow dependency conflict observed during release verification

During this release review, `pip-audit` identified vulnerabilities in the installed MLflow 3.2.0 and PyArrow 21.0.0 packages. Raising the MLflow lower bound to 3.15.0 addressed the known MLflow/PyArrow findings but created an unsatisfiable constraint with `cryptography>=50.0.0`: MLflow 3.15.x requires `cryptography<50`, while the audited Cryptography advisory is fixed in 50.0.0. MLflow's upstream issue reports that the upper-bound relaxation to `cryptography<51` was merged and is intended for the next minor release, but the available 3.15.1 release still carries the restrictive bound.[12] [13]

**Release implication:** MLflow tracking must remain an explicitly deferred optional integration until a released compatible MLflow version is available and passes the project audit. It must not be bundled by the legacy `fm` or `all` release extras, and the final verification report must record this limitation without calling the dependency audit clean.

[12]: https://github.com/mlflow/mlflow/issues/24871 "MLflow issue #24871: release dependency on cryptography<50"
[13]: https://github.com/mlflow/mlflow/issues/24928 "MLflow issue #24928: MLflow pins or restricts cryptography<50"
