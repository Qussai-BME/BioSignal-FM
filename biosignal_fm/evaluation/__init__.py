"""Cross-validation and statistical rigor suite."""

from __future__ import annotations

from .cross_validation import LeaveOneDatasetOutCV, LeaveOneSubjectOutCV
from .metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    f1_score,
)
from .mixed_effects import MixedEffectsAnalyzer, MixedEffectsResult
from .statistics import (
    bca_bootstrap_ci,
    bonferroni_holm_correction,
    cohens_d,
    friedman_nemenyi_test,
    friedman_test,
    hedges_g,
    holm_sidak_correction,
    nemenyi_posthoc,
    power_analysis_ttest,
    wilcoxon_holm_sidak,
    wilcoxon_signed_rank,
)

__all__ = [
    "LeaveOneSubjectOutCV",
    "LeaveOneDatasetOutCV",
    "cohens_d",
    "hedges_g",
    "bca_bootstrap_ci",
    "friedman_test",
    "friedman_nemenyi_test",
    "nemenyi_posthoc",
    "wilcoxon_signed_rank",
    "wilcoxon_holm_sidak",
    "holm_sidak_correction",
    "bonferroni_holm_correction",
    "power_analysis_ttest",
    "accuracy",
    "f1_score",
    "confusion_matrix",
    "classification_report",
    "MixedEffectsAnalyzer",
    "MixedEffectsResult",
]
