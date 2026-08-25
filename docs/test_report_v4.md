# Release Verification Report — BioSignal-FM V4

**Verification date:** 25 August 2026 (supersedes the 22 August 2026 4.0.0 report)  
**Scope:** Independent line-by-line review of the 4.0.0 pre-publication bundle, applied
fixes, and re-verification of the resulting 4.0.1 source release.  
**Decision:** **Passed for release as an engineering and research-software platform.**

> This report was regenerated from a fresh, independently reproduced verification
> run — not copied forward from the 4.0.0 report. See
> [What changed since 4.0.0](#what-changed-since-400) below for exactly what the
> line-by-line review found and fixed, none of which touched application behavior.

> This report verifies code, contracts, integrations, documentation, reproducibility, isolated installation, and dependency posture. It does not establish real-data benchmark performance, generalization, clinical utility, or regulatory clearance.

## Measured final verification

| Verification area | Command or procedure | Measured result |
|---|---|---|
| Unit and integration tests | `python -m pytest -q --disable-warnings --maxfail=1 --cov=biosignal_fm --cov-fail-under=75` | **329 passed**, **0 skipped**, **112 non-blocking warnings**, ~44 s (measured against the committed tree; timing is environment-dependent and not itself a pass/fail gate). |
| Coverage | `pytest-cov` output from the final suite | **80.42%**, above the project threshold of 75%. |
| Ruff lint | `ruff check biosignal_fm tests scripts` | **Passed; zero findings.** |
| Ruff format | `ruff format --check biosignal_fm tests scripts` | **Passed; 92 files formatted.** |
| Static typing | `mypy biosignal_fm scripts` | **Passed; zero errors in 68 source files.** |
| Documentation | `mkdocs build --strict` | **Passed.** The Material theme emitted an upstream MkDocs-2.0 deprecation notice; it did not fail strict validation and is tracked as an ecosystem risk. |
| Git hygiene | `git status --porcelain`, live `RunManifest` test | **Passed** from clean head tagged `v4.0.1` (see `PACKAGE_IDENTITY.txt` — a tag rather than a hardcoded hash, deliberately, per the "What changed since 4.0.0" note below); the live test records both clean and deliberately dirty states, then confirms the tree returns to clean. |
| Isolated core smoke test | Fresh Python 3.12 environment, package installation, then `scripts/verify_core_install.py` | **Passed**; version `4.0.1`, registry `emg,eeg,ecg,ecog,fnirs`, no scientific reader/model extra required by the core check. |
| Isolated release extras | Fresh Python 3.12 environment, `pip install '.[all]'` | **Passed**; all distributed extras resolved without MLflow tracking. |
| Dependency integrity | `pip check` and `pip-audit --progress-spinner off` in the isolated release environment | **Passed; no broken requirements and no known vulnerabilities.** The local editable `biosignal-fm` package is skipped by the public advisory service because it is not published on PyPI. |

## What changed since 4.0.0

An independent line-by-line review of the packaged 4.0.0 bundle (not just the
source tree, but the exact archive that would have been uploaded) found four
release-hygiene defects that the 4.0.0 verification and audit documents did
not catch, none of which affected test results, security posture, or
architecture:

| Finding | Fix |
|---|---|
| `CITATION.cff` declared version `0.1.2` (a pre-V4 version) and a placeholder ORCID (`0000-0000-0000-0000`). | Version corrected to `4.0.1`; the fake ORCID was removed rather than shipped as a false identifier. |
| The packaged archive included `mlflow.db`, `build/`, `biosignal_fm.egg-info/`, a generated `site/` docs build, and `runs/` (local synthetic-data run artifacts) — all already excluded by the project's own `.gitignore`, but present anyway. The MLflow database leaked a developer's absolute local filesystem path. | All removed. The committed tree now matches `.gitignore`. |
| No `requirements.txt` at the repository root. Streamlit Community Cloud's dependency detection does not reliably parse this project's PEP 621 `pyproject.toml` (it assumes a Poetry-format `[tool.poetry]` table). | Added `requirements.txt` (`-e .[fm,api,ui]`, matching the Docker image's extras). |
| `.streamlit/config.toml` hardcoded `server.address` / `browser.serverAddress` to `"localhost"`, which had no effect locally but could conflict with how Streamlit Community Cloud manages its own networking. | Removed; documented the local-dev CLI flag alternative in a comment. |
| The 4.0.0 verification report cited a tested commit (`732672c9…`) that did not match the commit recorded in the packaged archive's own `PACKAGE_IDENTITY.txt` (`f98888e3…`) — the exact artifact shipped had not been re-verified after packaging. | This release is tagged (`v4.0.1`) and the numbers in this report were measured against that exact tagged commit, in the same pass that produced the tag, closing the gap between "tested" and "shipped." |

Full details, including the results table found inside `runs/` — already
correctly labeled as non-benchmark-eligible synthetic data, but not
appropriate to ship in a citable release regardless — are in the CHANGELOG's
`[4.0.1]` entry.

## Closure items retained from migration review

| Item | Status | Evidence |
|---:|---|---|
| 1. Ruff repair | Closed | The one real I001 finding in `scripts/verify_core_install.py` was manually corrected; final lint is clean. |
| 2. Independent EEG and ECG integration | Closed | Each path covers a real loader call with labeled synthetic fallback, modality preprocessing, encoder, prediction, ONNX export, and numerical parity. |
| 3. Executable roadmap | Closed | [V4 Roadmap](roadmap_v4.md) is linked from README and MkDocs and includes real data, Git, ECoG, CI, and release actions. |
| 4. Git and `RunManifest` | Closed | Repository Git capture is exercised against a real clean/dirty working tree. |
| 5. Publication hardening | Closed with declared external deferral | The API restricts model paths and validates finite exact-shape inputs; WebSocket credentials are not accepted in URLs; loopback is the default binding; Compose/CI/public documentation were hardened; MLflow tracking is excluded until its released dependency range supports the Cryptography security baseline. |
| 6. Packaging and citation hygiene | Closed | See [What changed since 4.0.0](#what-changed-since-400): stale `CITATION.cff` metadata, stray dev artifacts, missing `requirements.txt`, and the tested-vs-shipped commit gap are all fixed and re-verified. |

## Functional coverage

| Component | Practical coverage | Result |
|---|---|---|
| `Signal` contracts and provenance | Signal shape, channels, batches, provenance, and synthetic labels. | Passed. |
| Modality registry | EMG/EEG/ECG core, ECoG experimental, fNIRS optional legacy. | Passed. |
| Core isolation | Registry inspection works without heavy scientific processing imports; processing factories are lazy. | Passed. |
| EMG, EEG, and ECG end-to-end paths | Loader/fallback, preprocessing, encoder, task prediction, ONNX export, and inference parity. | Passed. |
| Multimodal order | Representation, explicit optional fusion, then task head. | Passed. |
| Deployment API | Restricted checkpoint staging, input validation, REST and WebSocket authentication behavior, and response hygiene. | Passed. |
| UI deployment playground | Real in-process registration and prediction path under the restricted model directory. | Passed. |
| English release documentation | Source documentation, public showcase language, roadmap, data governance, risk register, and references. | Passed in strict build. |

## Security and supply-chain decision

The review found vulnerable installed versions of MLflow 3.2.0 and PyArrow 21.0.0 in the prior development environment. The latest available MLflow release line still constrained `cryptography<50`, while the required Cryptography remediation begins at 50.0.0. Upstream has merged a relaxation but had not released it at the time of verification.[1] [2]

Accordingly, MLflow tracking is **not distributed by `tracking`, `fm`, or `all` extras** in this release. The core retains local `RunManifest` and local tracking capabilities. Reintroducing MLflow requires a released upstream version compatible with `cryptography>=50.0.0`, a fresh isolated installation, and a clean dependency audit. This is a deliberate security boundary, not a claim that MLflow tracking itself is unsafe in every deployment.

## Intentional remaining boundaries

The release does not include licensed real datasets, dataset manifests for an actual study, published benchmark artifacts, validated cross-dataset transfer evidence, or clinical validation. ECoG/iEEG remains experimental. The [V4 Roadmap](roadmap_v4.md) contains the acceptance gates for changing these positions; [Known Limitations](known_limitations_v4.md) governs public claim language.

## Release decision

BioSignal-FM V4 is approved for publication as a **modular, provenance-aware biosignal research platform**. The source release has a clean Git history, verified distributed extras, no known audited dependency vulnerabilities, passing tests and static checks, and strict English documentation. Publication must preserve the stated data, evidence, operational-security, and regulatory boundaries.

## References

[1]: https://github.com/mlflow/mlflow/issues/24871 "MLflow issue #24871: release dependency on cryptography<50"
[2]: https://github.com/mlflow/mlflow/issues/24928 "MLflow issue #24928: MLflow pins or restricts cryptography<50"
