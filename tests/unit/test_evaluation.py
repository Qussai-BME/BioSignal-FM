"""Unit tests for biosignal_fm.evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from biosignal_fm.evaluation import (
    LeaveOneDatasetOutCV,
    LeaveOneSubjectOutCV,
    accuracy,
    bca_bootstrap_ci,
    bonferroni_holm_correction,
    classification_report,
    cohens_d,
    confusion_matrix,
    f1_score,
    friedman_test,
    hedges_g,
    holm_sidak_correction,
    nemenyi_posthoc,
    power_analysis_ttest,
    wilcoxon_holm_sidak,
    wilcoxon_signed_rank,
)


class TestCrossValidation:
    def test_loso_splits(self) -> None:
        cv = LeaveOneSubjectOutCV()
        subjects = np.array([0, 0, 1, 1, 2, 2])
        folds = list(cv.split(subjects))
        assert len(folds) == 3
        # Fold for subject 0: train on 1,2; test on 0
        train_idx, test_idx = folds[0]
        assert set(subjects[train_idx]) == {1, 2}
        assert set(subjects[test_idx]) == {0}

    def test_loso_no_overlap(self) -> None:
        cv = LeaveOneSubjectOutCV()
        subjects = np.array([0, 1, 2, 3])
        for train_idx, test_idx in cv.split(subjects):
            assert not set(train_idx) & set(test_idx)

    def test_lodo_splits(self) -> None:
        cv = LeaveOneDatasetOutCV()
        datasets = np.array(["A", "A", "B", "B", "C", "C"])
        folds = list(cv.split(datasets))
        assert len(folds) == 3


class TestEffectSizes:
    def test_cohens_d_positive(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 1.0, 200)
        b = rng.normal(0.0, 1.0, 200)
        d = cohens_d(a, b)
        assert d > 0.5  # large positive effect

    def test_cohens_d_zero(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        d = cohens_d(a, a)
        assert d == 0.0

    def test_hedges_g_close_to_d_for_large_n(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 1.0, 1000)
        b = rng.normal(0.0, 1.0, 1000)
        d = cohens_d(a, b)
        g = hedges_g(a, b)
        assert abs(d - g) < 0.01  # nearly identical for large N


class TestBCABootstrap:
    def test_ci_contains_mean(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, 100)
        lo, hi = bca_bootstrap_ci(data, np.mean, n_boot=1000, random_state=42)
        assert lo < 5.0 < hi

    def test_ci_narrows_with_more_data(self) -> None:
        rng = np.random.default_rng(42)
        small = rng.normal(5.0, 1.0, 20)
        large = rng.normal(5.0, 1.0, 500)
        lo_s, hi_s = bca_bootstrap_ci(small, np.mean, n_boot=500, random_state=42)
        lo_l, hi_l = bca_bootstrap_ci(large, np.mean, n_boot=500, random_state=42)
        assert (hi_l - lo_l) < (hi_s - lo_s)


class TestFriedman:
    def test_rejects_null(self) -> None:
        # Clear ranking difference
        scores = np.array(
            [
                [0.9, 0.5, 0.4],
                [0.95, 0.55, 0.45],
                [0.92, 0.52, 0.42],
                [0.88, 0.48, 0.38],
                [0.93, 0.53, 0.43],
            ]
        )
        result = friedman_test(scores, alpha=0.05)
        assert result["reject_null"] is True
        assert result["p_value"] < 0.05
        # Method 0 (best) should have rank 1
        assert result["average_ranks"][0] == 1.0

    def test_no_significant_difference(self) -> None:
        # All methods very similar
        scores = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.51, 0.5, 0.51],
            ]
        )
        result = friedman_test(scores, alpha=0.05)
        # With near-identical scores, ranks are essentially random
        assert "p_value" in result

    def test_nemenyi_returns_cd(self) -> None:
        scores = np.array(
            [
                [0.9, 0.5, 0.4],
                [0.95, 0.55, 0.45],
                [0.92, 0.52, 0.42],
            ]
        )
        result = nemenyi_posthoc(scores, alpha=0.05)
        assert "critical_difference" in result
        assert result["critical_difference"] > 0


class TestWilcoxon:
    def test_significant_difference(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(1.0, 1.0, 50)
        b = rng.normal(0.0, 1.0, 50)
        result = wilcoxon_signed_rank(a, b)
        assert result["reject_null"]

    def test_no_difference(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.0, 1.0, 50)
        b = a + rng.normal(0.0, 0.01, 50)  # nearly identical
        result = wilcoxon_signed_rank(a, b)
        assert not result["reject_null"]

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            wilcoxon_signed_rank([1, 2, 3], [1, 2])


class TestHolmSidakCorrection:
    def test_basic_correction(self) -> None:
        pvalues = [0.01, 0.04, 0.03, 0.20]
        result = holm_sidak_correction(pvalues, alpha=0.05)
        assert len(result["rejected"]) == 4
        assert len(result["corrected_pvalues"]) == 4
        # Smallest p (0.01) should be rejected
        assert result["rejected"][0] is True
        # Largest p (0.20) should not be rejected
        assert result["rejected"][3] is False

    def test_formula_is_sidak_not_bonferroni(self) -> None:
        """CRITICAL: Verify the formula is Šídák, not Bonferroni-Holm.

        For a single p-value with m=4 tests, k=1 (smallest):
        - Bonferroni-Holm: p * (m - k + 1) = p * 4
        - Šídák:            1 - (1 - p)^(m - k + 1) = 1 - (1 - p)^4

        For p=0.05:
        - Bonferroni-Holm: 0.05 * 4 = 0.20
        - Šídák:            1 - 0.95^4 = 1 - 0.8145 = 0.1855

        These are distinct values; verify we get Šídák.
        """
        result = holm_sidak_correction([0.05, 0.10, 0.20, 0.30], alpha=1.0)
        # With alpha=1.0, all rejected but we can still inspect corrected p-values
        sidak_corrected = result["corrected_pvalues"][0]
        bonferroni_corrected = 0.05 * 4  # = 0.20
        assert abs(sidak_corrected - bonferroni_corrected) > 0.001  # distinct!
        # Verify it matches Šídák formula
        expected_sidak = 1 - (1 - 0.05) ** 4
        assert abs(sidak_corrected - expected_sidak) < 1e-10

    def test_monotonicity(self) -> None:
        """Corrected p-values should be monotonic (step-down property)."""
        pvalues = [0.001, 0.01, 0.04, 0.05, 0.10]
        result = holm_sidak_correction(pvalues, alpha=1.0)
        corrected = result["corrected_pvalues"]
        # Sort by original p; corrected should be non-decreasing
        indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
        sorted_corrected = [corrected[i] for i, _ in indexed]
        for i in range(1, len(sorted_corrected)):
            assert sorted_corrected[i] >= sorted_corrected[i - 1] - 1e-10

    def test_compare_with_bonferroni(self) -> None:
        """Šídák should be at most slightly more powerful than Bonferroni."""
        pvalues = [0.01, 0.02, 0.03, 0.04]
        sidak = holm_sidak_correction(pvalues, alpha=0.05)
        bonf = bonferroni_holm_correction(pvalues, alpha=0.05)
        # For small p, Šídák and Bonferroni are very close
        for s, b in zip(sidak["corrected_pvalues"], bonf["corrected_pvalues"], strict=True):
            assert abs(s - b) < 0.01


class TestWilcoxonHolmSidak:
    """Regression tests for wilcoxon_holm_sidak's documented list[bool] contract.

    Previously this function returned the full dict from
    holm_sidak_correction() instead of just the "rejected" list, which
    silently broke every caller that iterated over the result expecting
    booleans (they got dict keys instead). Guard against that regressing.
    """

    def test_return_type_is_list_of_bool(self) -> None:
        result = wilcoxon_holm_sidak([0.01, 0.04, 0.03, 0.20], alpha=0.05)
        assert isinstance(result, list)
        assert all(isinstance(x, bool) for x in result)

    def test_matches_holm_sidak_correction_rejected_field(self) -> None:
        pvalues = [0.001, 0.02, 0.03, 0.04, 0.20]
        expected = holm_sidak_correction(pvalues, alpha=0.05)["rejected"]
        assert wilcoxon_holm_sidak(pvalues, alpha=0.05) == expected

    def test_length_matches_input(self) -> None:
        pvalues = [0.01, 0.02, 0.03]
        assert len(wilcoxon_holm_sidak(pvalues)) == len(pvalues)


class TestPowerAnalysis:
    def test_large_effect_needs_fewer_samples(self) -> None:
        small_effect = power_analysis_ttest(effect_size=0.2, power=0.8)
        large_effect = power_analysis_ttest(effect_size=0.8, power=0.8)
        assert large_effect["n_per_group"] < small_effect["n_per_group"]

    def test_higher_power_needs_more_samples(self) -> None:
        low_power = power_analysis_ttest(effect_size=0.5, power=0.7)
        high_power = power_analysis_ttest(effect_size=0.5, power=0.95)
        assert high_power["n_per_group"] > low_power["n_per_group"]


class TestMetrics:
    def test_accuracy(self) -> None:
        assert accuracy([0, 1, 2, 0], [0, 1, 1, 0]) == 0.75

    def test_f1_macro(self) -> None:
        f1 = f1_score([0, 1, 2, 0, 1], [0, 1, 1, 0, 1], average="macro")
        assert 0.0 <= f1 <= 1.0

    def test_confusion_matrix_single_fold(self) -> None:
        cm = confusion_matrix([0, 1, 2, 0], [0, 1, 1, 0], n_classes=3)
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 2  # true=0, pred=0: 2 samples
        assert cm[1, 1] == 1  # true=1, pred=1: 1 sample
        assert cm[2, 1] == 1  # true=2, pred=1: 1 sample

    def test_confusion_matrix_multi_fold_aggregation(self) -> None:
        """CRITICAL: Multiple folds must be aggregated (MyoControl v2.0 audit fix)."""
        # Two folds, each with 3 samples
        y_true_folds = [[0, 1, 2], [0, 1, 2]]
        y_pred_folds = [[0, 1, 1], [1, 1, 2]]
        cm = confusion_matrix(y_true_folds, y_pred_folds, n_classes=3)
        assert cm.sum() == 6  # 6 total samples
        # fold 0: t=0,p=0; t=1,p=1; t=2,p=1
        # fold 1: t=0,p=1; t=1,p=1; t=2,p=2
        # cm[0,0] = 1, cm[0,1] = 1, cm[1,1] = 2, cm[2,1] = 1, cm[2,2] = 1
        assert cm[0, 0] == 1
        assert cm[0, 1] == 1
        assert cm[1, 1] == 2
        assert cm[2, 1] == 1
        assert cm[2, 2] == 1

    def test_classification_report(self) -> None:
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 1]
        report = classification_report(y_true, y_pred)
        assert "per_class" in report
        assert "macro_f1" in report
        assert "accuracy" in report
        assert 0.0 <= report["accuracy"] <= 1.0
