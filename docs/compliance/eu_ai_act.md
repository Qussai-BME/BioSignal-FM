# EU AI Act Annex IV Compliance

BioSignal-FM is classified as a **limited-risk** AI system under the EU AI Act
(Annex IV technical documentation requirements). This document outlines the
required technical documentation.

## 1. System Description

- **Name:** BioSignal-FM
- **Version:** 0.1.2
- **Author:** Qussai Adlbi
- **Description:** A unified transformer-based foundation model for surface
  biosignals (EMG, ECG, EEG, fNIRS), pretrained via self-supervised learning.
- **Intended use:** Research and educational use in biomedical engineering,
  neural interface development, and biosignal analysis.
- **Not intended for:** Clinical decision-making, medical diagnosis, or any
  use that requires regulatory clearance.

## 2. Data Description

- **Training data:** Public datasets only (NinaPro, PhysioNet, Brain-BIDS)
- **No personal data:** No PII collected or stored
- **Data minimization:** Only biosignal arrays and metadata are loaded
- **Datasheets:** Each dataset has a Pushkarna et al. (2022) datasheet

## 3. Model Architecture

See `docs/architecture.md` for the full architecture description.

## 4. Risk Management

- **Identified risks:** Misuse for clinical decision-making; biased performance
  on underrepresented populations; adversarial inputs.
- **Mitigations:** Clear "research-grade only" labeling; statistical rigor
  suite for honest reporting; input validation on all REST endpoints.

## 5. Post-Market Monitoring

- **Issue tracker:** https://github.com/qussaiadlbi/biosignal-fm/issues
- **Security policy:** SECURITY.md
- **Vulnerability reporting:** qussai.adlbi@proton.me

## 6. Transparency

- **Open source:** Apache 2.0
- **Model cards:** Auto-generated via `biosignal_fm.deployment`
- **Run manifests:** SHA-256 of all outputs, env fingerprint, git HEAD
