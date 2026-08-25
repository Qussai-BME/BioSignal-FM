# BioSignal-FM — Regulatory Readiness Appendix

**Version:** 1.0
**Date:** 2026-08-15
**Scope:** How BioSignal-FM's design maps to EU AI Act, IEC 62304 Ed. 2, and
FDA GMLP / PCCP requirements.

BioSignal-FM is **research-grade only** and is NOT FDA-cleared SaMD. This
appendix documents the design decisions that anticipate future clinical
translation, so a downstream clinical team can build on the project without
re-architecting.

---

## 1. EU AI Act (Regulation 2024/1689)

BioSignal-FM, if used for clinical decision support on biosignals, would be
classified as **high-risk AI** under Annex III (medical) and would be subject
to the obligations applicable from **2 August 2026**. Standalone high-risk
AI systems have an extended compliance date of **2 December 2027**. Existing
medical AI complying with MDR/IVDR has until **2 August 2027**.

### Design mapping

| EU AI Act Requirement (Art. 9-15) | BioSignal-FM Implementation |
|---|---|
| Risk management system (Art. 9) | `RunManifest` records every run with SHA-256 + env fingerprint; risks tracked in `ARCHITECTURE.md` §8.5 |
| Data and data governance (Art. 10) | `data/datasheets/` for every dataset (Pushkarna et al. 2022 format); only public datasets used; no PII in repo |
| Technical documentation (Art. 11) | `ARCHITECTURE.md` (784 lines), `docs/research/preregistration.md`, generated model cards |
| Record-keeping (Art. 12) | `RunManifest` JSON files, MLflow tracking, automatic logging of params/metrics/artifacts |
| Transparency & instructions for use (Art. 13) | README, `docs/quickstart.md`, CLI `--help` on every command, model card with intended use & limitations |
| Human oversight (Art. 14) | CLI/UI require explicit user confirmation for mutating operations; FastAPI requires API key |
| Accuracy, robustness, cybersecurity (Art. 15) | LOSO + LODO CV with statistical tests; `weights_only=True` checkpoint loading; non-root Docker user; `cap_drop: ALL` |

### Gaps (acknowledged)

- No formal CE marking process (would require Notified Body review).
- No post-market monitoring system (PMS) — would need to be added for clinical deployment.
- No conformity assessment procedure documented.

---

## 2. IEC 62304 Edition 2 (2026)

IEC 62304 Ed. 2 (released 2026) is the first major revision since 2015. It
adds AI/ML-specific compliance requirements and revises software safety
classification. BioSignal-FM anticipates **Class B** classification (non-life-
threatening injury possible if the software fails) for downstream clinical use.

### Design mapping

| IEC 62304 Ed. 2 Requirement | BioSignal-FM Implementation |
|---|---|
| Software safety classification (5.2) | Documented as anticipated Class B in `docs/compliance/iec62304.md` (to be created) |
| Software development plan (5.1) | `ARCHITECTURE.md` §10 (Milestone Plan) serves as the SDP |
| Software requirements (5.3) | `pyproject.toml` + `ExperimentConfig` dataclass captures all configurable parameters |
| Software architectural design (5.4) | Hexagonal architecture, `ARCHITECTURE.md` §3 |
| Software detailed design (5.5) | Module-by-module spec in `ARCHITECTURE.md` §4 |
| Software unit verification (5.6) | 199 unit tests + 42 integration tests; ruff + mypy in CI |
| Software integration & integration testing (5.7) | `tests/integration/test_end_to_end.py` runs the full pretrain→finetune→export→serve pipeline |
| Software system testing (5.8) | Integration tests cover the full system; CI runs on Python 3.10/3.11/3.12 |
| Software release (5.8) | Git tags, CHANGELOG.md, CITATION.cff with DOI placeholder |
| Software configuration management (8) | `pyproject.toml` pins all dependencies; `RunManifest` records env fingerprint |
| Software maintenance (6) | CHANGELOG.md tracks all changes; SECURITY.md documents vulnerability reporting |

### Gaps (acknowledged)

- No formal software safety classification by a Notified Body.
- No SOUP (Software of Unknown Provenance) inventory — would need to be added.
- No problem resolution process documented.

---

## 3. FDA GMLP (Good Machine Learning Practice)

The FDA's 10 GMLP principles (originally published jointly with Health Canada
and MHRA in 2021) remain the guiding framework. The FDA's **6 January 2025
Draft Guidance** "AI-Enabled Device Software Functions" and the **August 2025
PCCP (Predetermined Change Control Plan) Guiding Principles** add specificity.

### Design mapping

| FDA GMLP Principle | BioSignal-FM Implementation |
|---|---|
| 1. Multidisciplinary expertise | Single-author (acknowledged limitation); 6-lens design methodology documents the perspectives considered |
| 2. Good software engineering | Hexagonal architecture, 90%+ test coverage, CI/CD, type hints, pre-commit |
| 3. Clinical study representatives | None (no clinical partners yet) |
| 4. Training dataset independence | LOSO + LODO CV enforced; subject-aware normalization; `data/datasheets/` for every dataset |
| 5. Ground truth reference | Public datasets with expert-validated labels (NinaPro, PhysioNet) |
| 6. Model design & training | Pre-registered analysis plan (`docs/research/preregistration.md`); mixed-effects + Friedman/Nemenyi/Wilcoxon-Holm-Šídák |
| 7. Human-AI interface | Streamlit UI with WCAG 2.2 AA contrast; FastAPI with API key auth |
| 8. Performance monitoring | `RunManifest` + MLflow tracking; benchmark CLI command |
| 9. Robustness to change | LODO CV measures domain shift; mixed-effects model captures subject/session variance |
| 10. Cybersecurity | `weights_only=True` checkpoint loading; non-root Docker; `cap_drop: ALL`; `no-new-privileges:true` |

### PCCP (Predetermined Change Control Plan)

A PCCP allows a manufacturer to make pre-specified changes to an AI/ML
device without submitting a new 510(k). BioSignal-FM's design anticipates
PCCP by:
- Versioning all model architectures in `ModelConfig` (frozen dataclass).
- Recording every training run in `RunManifest` with SHA-256 of outputs.
- Maintaining a `CHANGELOG.md` with semantic versioning.
- Allowing model swapping at runtime via the `ModelRegistry` (UUID-based, no
  client-supplied paths).

### Gaps (acknowledged)

- No formal 510(k) submission.
- No real-world performance monitoring system (would require post-market data).
- No adversarial robustness testing (e.g., FGSM, PGD on biosignals).

---

## 4. GDPR (General Data Protection Regulation)

BioSignal-FM uses only **public, de-identified datasets** (NinaPro,
PhysioNet, Brain-BIDS). No personal data is collected, stored, or processed
by the project itself. However, if a downstream user fine-tunes on their own
clinical data:

- Article 5 (data minimization): only the biosignal modalities needed for the
  task are processed.
- Article 25 (data protection by design): `RunManifest` does not record raw
  signal values — only SHA-256 hashes.
- Article 32 (security): API key auth, non-root Docker, `cap_drop: ALL`.
- Article 35 (DPIA): `docs/compliance/gdpr.md` documents the DPIA template
  for downstream users.

---

## 5. Limitations & Disclaimers

1. **Not a medical device.** BioSignal-FM is research-grade software. It is
   NOT intended for clinical diagnosis, treatment, or patient management.
2. **No regulatory clearance.** The project has not been reviewed by FDA,
   EMA, or any Notified Body.
3. **No clinical validation.** All benchmarks are on public research datasets.
   Performance on clinical data is unknown.
4. **Single-author.** The regulatory mapping above is the author's best
   interpretation; a regulatory professional should review before any clinical
   use.

---

**End of regulatory appendix.** This document will be updated as the project
matures and as EU AI Act enforcement dates approach (2 Aug 2026, 2 Dec 2027,
2 Aug 2027).
