# BioSignal-FM — Pre-Registration of Analysis Plan

**Version:** 1.0
**Date:** 2026-08-15
**Author:** Qussai Adlbi
**Status:** Pre-registered (frozen before any experiment results are computed)

This document pre-registers the analysis plan for BioSignal-FM's three core
hypotheses (H1, H2, H3). Pre-registration separates the *a priori* analysis
plan from the *post hoc* interpretation, reducing the risk of p-hacking and
HARKing (Hypothesizing After Results are Known). This is a NeurIPS 2025
Reproducibility Checklist item and an OSF-recognized practice.

---

## 1. Hypotheses (Falsifiable, Pre-Registered)

### H1 (Unification)

**Claim:** A single transformer pretrained jointly on EMG, ECG, EEG, and fNIRS
achieves within 5 percentage points of modality-specific baselines on each
modality's standard benchmark.

**Operationalization:**
- Metric: macro-F1 (LOSO, 10 folds).
- Modality-specific baselines: LDA+TD features (EMG), Pan-Tompkins+SVM (ECG),
  Filter-bank CSP+LDA (EEG), GLM+SVM (fNIRS).
- Datasets: NinaPro DB5 (EMG), MIT-BIH (ECG), EEGMMID (EEG), Brain-BIDS fNIRS.
- Threshold: |F1_unified − F1_baseline| ≤ 0.05 on each modality.

**Falsification:** H1 is falsified if, on any modality, the unified model's
macro-F1 is more than 5 percentage points below the modality-specific baseline
(BCa 95% CI of the difference excludes 0.05).

### H2 (Cross-modal transfer)

**Claim:** Pretraining on EMG (the highest-data modality) and zero-shot
fine-tuning on EEG (the lowest-data modality) outperforms EEG-only pretraining
under ≤100-subject budgets.

**Operationalization:**
- Metric: macro-F1 on EEGMMID 4-class motor imagery, with N=50, 75, 100
  training subjects (3 budgets).
- Effect size: Hedges' g ≥ 0.5 (medium effect) for EMG-pretrained vs
  EEG-only-pretrained.
- Statistical test: paired Wilcoxon signed-rank with Holm-Šídák correction
  (alpha = 0.05, 3 comparisons).
- Reporting: BCa 95% CI on Hedges' g.

**Falsification:** H2 is falsified if Hedges' g < 0.5 at any budget AND the
95% CI excludes 0.5.

### H3 (Calibration reduction)

**Claim:** Fine-tuned BioSignal-FM reaches clinical-grade accuracy (≥90%
gesture recognition) with ≤3 minutes of per-subject calibration data, versus
≥10 minutes for modality-specific baselines.

**Operationalization:**
- Metric: gesture recognition accuracy on NinaPro DB5 (8-class).
- Calibration budgets: 1, 2, 3, 5, 10, 20 minutes.
- Statistical test: two-sample t-test on accuracy at each budget (alpha = 0.05,
  Holm-Šídák across 6 budgets).
- Reporting: BCa 95% CI on accuracy gap (BioSignal-FM − baseline) at 3 minutes.

**Falsification:** H3 is falsified if BioSignal-FM's accuracy at 3 minutes is
<90% (BCa 95% CI upper bound excludes 90%) OR if the baseline's accuracy at
10 minutes exceeds BioSignal-FM's at 3 minutes (paired Wilcoxon p < 0.05).

---

## 2. Analysis Plan (Frozen)

### 2.1 Cross-validation protocol
- LOSO (Leave-One-Subject-Out): primary protocol.
- LODO (Leave-One-Dataset-Out): secondary protocol, for cross-dataset
  generalization.
- Subject-aware normalization: statistics computed on training folds only.

### 2.2 Statistical tests
1. **Friedman test** (alpha = 0.05) across all (dataset, method) pairs.
2. **Nemenyi post-hoc** for pairwise method ranking (critical difference).
3. **Wilcoxon signed-rank** with **Holm-Šídák** step-down correction
   (formula: `corrected_p = 1 - (1 - p)^(m - k + 1)`, NOT Bonferroni-Holm).
4. **BCa bootstrap CIs** (10,000 resamples) for all reported means.
5. **Hedges' g** (small-N corrected Cohen's d) for effect sizes.
6. **Mixed-effects model** (accuracy ~ method + (1|subject) + (1|session)) as
   a complementary analysis reporting ICC, β coefficients, and p-values.

### 2.3 Multiple comparisons correction
- Within-task: Holm-Šídák across all pairwise method comparisons.
- Across-tasks: Holm-Šídák across the 4 modality tasks (not just within one).
- Across-budgets (for H3): Holm-Šídák across the 6 calibration budgets.

### 2.4 Sample size justification
- A-priori power analysis (G*Power or statsmodels): two-sided t-test, alpha = 0.05,
  power = 0.8, effect size d = 0.5 (medium). Required N per group: 64.
- For LOSO with 10 subjects: underpowered for d < 0.8. We will report actual
  achieved power alongside all statistical tests.

### 2.5 Reproducibility
- `set_global_seed(42)` for all experiments.
- RunManifest (SHA-256 + env fingerprint + git HEAD) saved with every run.
- All configs version-controlled under `experiments/configs/`.
- All checkpoints SHA-256-hashed and uploaded to OSF.

### 2.6 Reporting
- All metrics reported as point estimate + BCa 95% CI.
- All hypothesis tests report test statistic, raw p-value, corrected p-value,
  and effect size.
- Negative results reported with the same detail as positive results.

---

## 3. Exclusion Criteria (Pre-Registered)

A subject is excluded from analysis if:
1. >20% of their recorded trials have NaN values that cannot be interpolated.
2. Their per-subject LOSO accuracy is >3 standard deviations from the cohort
   mean (outlier).
3. The recording was flagged by the dataset's own quality-control pipeline.

Exclusions are logged in `experiments/exclusions.csv` with reason codes.

---

## 4. Deviations from Pre-Registration

Any deviation from this plan will be:
1. Documented in `experiments/deviations.md` with date, reason, and signature.
2. Reported in the paper's Methods section under "Deviations from pre-registration."
3. The original (pre-deviation) analysis will also be reported for transparency.

---

## 5. Limitations (Acknowledged Upfront)

1. CPU-only compute limits the size of pretraining (cannot match REVE/LUNA scale).
2. Public datasets only (no clinical data access).
3. Surface biosignals only (no invasive recordings).
4. Single-author project (limited peer review during development).
5. The H2 (EMG→EEG transfer) hypothesis is **unproven** in the literature; this
   project may produce a negative result, which we will report honestly.
6. No clinical validation (research-grade only, not FDA-cleared SaMD).

---

**End of pre-registration.** This document is frozen as of the date above.
Any change requires a versioned revision (`preregistration_v1.1.md`) with a
change log.
