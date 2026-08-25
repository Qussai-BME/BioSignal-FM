"""Statistical rigor suite for biosignal evaluation.

Implements the exact protocol mandated by the master prompt:

1. **Effect sizes:** Cohen's d, Hedges' g (small-N corrected).
2. **Bootstrap CIs:** BCa (bias-corrected and accelerated), NOT naive percentile.
3. **Hypothesis tests:** Friedman + Nemenyi post-hoc + Wilcoxon signed-rank
   with Holm-Šídák step-down correction.
4. **Power analysis:** A-priori sample size justification.

CRITICAL: The Holm-Šídák formula is ``corrected_p = 1 - (1 - p)^(m - k)``,
NOT Bonferroni-Holm ``p * (m - k + 1)``. These are mathematically distinct.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np
from scipy import stats

__all__ = [
    "cohens_d",
    "hedges_g",
    "bca_bootstrap_ci",
    "friedman_test",
    "nemenyi_posthoc",
    "wilcoxon_signed_rank",
    "holm_sidak_correction",
    "bonferroni_holm_correction",
    "power_analysis_ttest",
]


def cohens_d(a: np.ndarray | Sequence[float], b: np.ndarray | Sequence[float]) -> float:
    """Cohen's d effect size (pooled standard deviation).

    Parameters
    ----------
    a, b : array-like
        Two samples.

    Returns
    -------
    float
        Cohen's d. Positive if mean(a) > mean(b).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> a = rng.normal(1.0, 1.0, 100)
    >>> b = rng.normal(0.0, 1.0, 100)
    >>> d = cohens_d(a, b)
    >>> d > 0.5  # large effect
    True
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    pooled_std = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_std)


def hedges_g(a: np.ndarray | Sequence[float], b: np.ndarray | Sequence[float]) -> float:
    """Hedges' g effect size (small-N corrected Cohen's d).

    Applies the correction factor ``J = 1 - 3/(4*df - 1)`` where
    ``df = na + nb - 2``. For large N, g approaches d.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> a = rng.normal(1.0, 1.0, 100)
    >>> b = rng.normal(0.0, 1.0, 100)
    >>> g = hedges_g(a, b)
    >>> g > 0.5
    True
    """
    d = cohens_d(a, b)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    df = len(a) + len(b) - 2
    if df < 2:
        return d
    J = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d * J


def bca_bootstrap_ci(
    data: np.ndarray | Sequence[float],
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    random_state: int | None = None,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    BCa adjusts for both bias (median of bootstrap distribution != point
    estimate) and skewness (acceleration). It is the gold standard for
    bootstrap CIs and should be preferred over naive percentile intervals
    for small or skewed samples.

    Parameters
    ----------
    data : array-like
        Input data (1D).
    statistic : callable
        Function that takes a 1D array and returns a scalar.
        Example: ``lambda x: np.mean(x)``.
    n_boot : int, optional
        Number of bootstrap resamples. Default 10_000.
    alpha : float, optional
        Two-sided alpha. Default 0.05 (returns 95% CI).
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    (lower, upper) : tuple of float
        BCa confidence interval.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> data = rng.normal(5.0, 1.0, 50)
    >>> lo, hi = bca_bootstrap_ci(data, np.mean, n_boot=2000, random_state=42)
    >>> lo < 5.0 < hi
    True
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)
    rng = np.random.default_rng(random_state)

    # Point estimate
    theta_hat = float(statistic(data))

    # Bootstrap resamples
    boot_stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = data[rng.integers(0, n, size=n)]
        boot_stats[i] = float(statistic(sample))

    # Bias correction z0
    # Proportion of boot_stats < theta_hat
    prop = float(np.mean(boot_stats < theta_hat))
    # Avoid 0 or 1 (would give inf z0)
    prop = min(max(prop, 1e-8), 1 - 1e-8)
    z0 = stats.norm.ppf(prop)

    # Acceleration a via jackknife
    jackknife_stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        jackknife_stats[i] = float(statistic(np.delete(data, i)))
    jack_mean = jackknife_stats.mean()
    numerator = np.sum((jack_mean - jackknife_stats) ** 3)
    denominator = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
    a = numerator / denominator if denominator != 0 else 0.0

    # BCa adjusted alphas
    def adjust(alpha_val: float) -> float:
        z_alpha = stats.norm.ppf(alpha_val)
        denom = 1 - a * (z0 + z_alpha)
        if denom == 0:
            return alpha_val
        z_adj = z0 + (z0 + z_alpha) / denom
        return float(stats.norm.cdf(z_adj))

    alpha_lo = adjust(alpha / 2)
    alpha_hi = adjust(1 - alpha / 2)

    # Clamp to [0, 1]
    alpha_lo = min(max(alpha_lo, 0.0), 1.0)
    alpha_hi = min(max(alpha_hi, 0.0), 1.0)

    lower = float(np.quantile(boot_stats, alpha_lo))
    upper = float(np.quantile(boot_stats, alpha_hi))
    return lower, upper


def friedman_test(
    scores: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Friedman test for comparing multiple methods across multiple datasets.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(n_datasets, n_methods)``. Higher = better.
    alpha : float, optional
        Significance level. Default 0.05.

    Returns
    -------
    dict
        {"chi2", "p_value", "df", "reject_null", "average_ranks"}.

    Examples
    --------
    >>> import numpy as np
    >>> scores = np.array([
    ...     [0.8, 0.7, 0.6],  # dataset 1
    ...     [0.9, 0.8, 0.7],  # dataset 2
    ...     [0.7, 0.6, 0.5],  # dataset 3
    ... ])
    >>> result = friedman_test(scores)
    >>> result["reject_null"]
    True
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D (n_datasets, n_methods), got {scores.shape}")

    n_datasets, n_methods = scores.shape
    if n_datasets < 2 or n_methods < 2:
        raise ValueError("Need at least 2 datasets and 2 methods")

    # Rank each row (dataset) — higher score = better rank (1)
    ranks = np.zeros_like(scores)
    for i in range(n_datasets):
        # Argsort descending: best method gets rank 1
        order = np.argsort(-scores[i])
        for rank, method_idx in enumerate(order):
            ranks[i, method_idx] = rank + 1

    average_ranks = ranks.mean(axis=0)
    chi2 = (12 * n_datasets / (n_methods * (n_methods + 1))) * (
        np.sum(average_ranks**2) - (n_methods * (n_methods + 1) ** 2) / 4
    )
    df = n_methods - 1
    p_value = float(stats.chi2.sf(chi2, df))

    return {
        "chi2": float(chi2),
        "p_value": p_value,
        "df": int(df),
        "reject_null": bool(p_value < alpha),
        "average_ranks": average_ranks.tolist(),
        "n_datasets": int(n_datasets),
        "n_methods": int(n_methods),
    }


def nemenyi_posthoc(
    scores: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Nemenyi post-hoc test after Friedman.

    Computes the critical difference (CD) for pairwise method comparison.
    Two methods are significantly different if their average ranks differ
    by more than CD.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(n_datasets, n_methods)``.
    alpha : float, optional
        Significance level. Default 0.05.

    Returns
    -------
    dict
        {"critical_difference", "average_ranks", "n_datasets", "n_methods", "alpha"}
    """
    scores = np.asarray(scores, dtype=np.float64)
    n_datasets, n_methods = scores.shape

    # Average ranks
    ranks = np.zeros_like(scores)
    for i in range(n_datasets):
        order = np.argsort(-scores[i])
        for rank, method_idx in enumerate(order):
            ranks[i, method_idx] = rank + 1
    average_ranks = ranks.mean(axis=0)

    # Nemenyi critical values (for alpha=0.05 and alpha=0.10).
    # Extended from k=2..10 to k=2..15 to cover the BioSignal-FM protocol
    # (up to 15 methods in a comparison). For k>15 we raise ValueError
    # rather than silently falling back to a wrong value.
    q_alpha_table_05 = {
        2: 1.960,
        3: 2.343,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.948,
        8: 3.031,
        9: 3.102,
        10: 3.164,
        11: 3.219,
        12: 3.268,
        13: 3.313,
        14: 3.354,
        15: 3.391,
    }
    q_alpha_table_10 = {
        2: 1.645,
        3: 2.052,
        4: 2.291,
        5: 2.460,
        6: 2.589,
        7: 2.693,
        8: 2.780,
        9: 2.855,
        10: 2.920,
        11: 2.970,
        12: 3.016,
        13: 3.058,
        14: 3.097,
        15: 3.135,
    }
    if abs(alpha - 0.05) < 1e-6:
        table = q_alpha_table_05
    elif abs(alpha - 0.10) < 1e-6:
        table = q_alpha_table_10
    else:
        raise ValueError(
            f"alpha must be 0.05 or 0.10 for Nemenyi post-hoc (got {alpha}). "
            "Use a different post-hoc test for other alphas."
        )
    if n_methods not in table:
        raise ValueError(
            f"Nemenyi q_alpha table only covers k=2..15 methods (got {n_methods}). "
            "Either reduce the number of methods or extend the table."
        )
    q_alpha = table[n_methods]

    cd = q_alpha * math.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets))

    return {
        "critical_difference": float(cd),
        "average_ranks": average_ranks.tolist(),
        "n_datasets": int(n_datasets),
        "n_methods": int(n_methods),
        "alpha": float(alpha),
        "q_alpha": float(q_alpha),
    }


def wilcoxon_signed_rank(
    a: np.ndarray | Sequence[float],
    b: np.ndarray | Sequence[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict:
    """Wilcoxon signed-rank test (paired samples).

    Parameters
    ----------
    a, b : array-like
        Paired samples (e.g. method A vs method B on same datasets).
    alpha : float
        Significance level.
    alternative : str
        "two-sided", "greater", or "less".

    Returns
    -------
    dict
        {"statistic", "p_value", "reject_null"}
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    stat, p = stats.wilcoxon(a, b, alternative=alternative)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "reject_null": bool(p < alpha),
        "n": int(len(a)),
    }


class HolmSidakResult(TypedDict):
    """Return shape of :func:`holm_sidak_correction`."""

    rejected: list[bool]
    corrected_pvalues: list[float]
    alpha: float
    m: int
    formula: str


def holm_sidak_correction(
    pvalues: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> HolmSidakResult:
    """Holm-Šídák step-down correction for multiple comparisons.

    CRITICAL: Uses the Šídák formula ``corrected_p = 1 - (1 - p)^(m - k + 1)``
    (k = 1-indexed rank among sorted p-values, smallest first), NOT the
    Bonferroni-Holm formula ``corrected_p = p * (m - k + 1)``.

    The two formulas are mathematically distinct:

    - Bonferroni: ``p_adj = p * (m - k + 1)`` (Bonferroni inequality)
    - Šídák:      ``p_adj = 1 - (1 - p)^(m - k + 1)`` (exact for independent tests)

    Šídák is slightly more powerful than Bonferroni when tests are independent.

    Parameters
    ----------
    pvalues : list or array
        Raw p-values from m hypothesis tests.
    alpha : float
        Family-wise alpha level.

    Returns
    -------
    dict
        {"rejected": list[bool], "corrected_pvalues": list[float], "alpha"}

    Examples
    --------
    >>> pvalues = [0.01, 0.04, 0.03, 0.20]
    >>> result = holm_sidak_correction(pvalues, alpha=0.05)
    >>> result["rejected"]
    [True, False, False, False]
    """
    pvalues = list(pvalues)
    m = len(pvalues)
    if m == 0:
        return {
            "rejected": [],
            "corrected_pvalues": [],
            "alpha": alpha,
            "m": 0,
            "formula": "1 - (1 - p)^(m - k + 1)",
        }

    # Sort by p-value ascending; keep original indices
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    corrected = [0.0] * m
    rejected = [False] * m

    # Step-down: at step k (1-indexed rank among sorted p-values, smallest
    # first), m - k + 1 hypotheses are still "in play", so the k-th smallest
    # p-value is adjusted as p_adj = 1 - (1 - p)^(m - k + 1). Running-max is
    # then applied across k so adjusted p-values are non-decreasing, which is
    # what makes this a step-down (as opposed to single-step Sidak) procedure.
    # See Shaffer (1995) for the canonical formulation.
    prev_corrected = 0.0
    for k, (orig_idx, p) in enumerate(indexed, start=1):
        # m - k + 1 tests remaining at this step
        exponent = m - k + 1
        # Clamp p to avoid log(0)
        p_clamped = min(max(p, 1e-12), 1 - 1e-12)
        p_adj = 1.0 - (1.0 - p_clamped) ** exponent
        # Enforce monotonicity (Holm step-down property)
        p_adj = max(p_adj, prev_corrected)
        corrected[orig_idx] = p_adj
        rejected[orig_idx] = p_adj < alpha
        prev_corrected = p_adj

    return {
        "rejected": rejected,
        "corrected_pvalues": corrected,
        "alpha": float(alpha),
        "m": int(m),
        "formula": "1 - (1 - p)^(m - k + 1)",
    }


def bonferroni_holm_correction(
    pvalues: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Bonferroni-Holm step-down correction.

    Provided for comparison ONLY. The recommended method is
    :func:`holm_sidak_correction`. Use this only when tests are not
    independent (Šídák assumption violated) or for backward compatibility
    with older analyses.

    Formula: ``corrected_p = p * (m - k + 1)``.
    """
    pvalues = list(pvalues)
    m = len(pvalues)
    if m == 0:
        return {"rejected": [], "corrected_pvalues": [], "alpha": alpha}

    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    corrected = [0.0] * m
    rejected = [False] * m
    prev = 0.0
    for k, (orig_idx, p) in enumerate(indexed, start=1):
        p_adj = p * (m - k + 1)
        p_adj = min(max(p_adj, prev), 1.0)
        corrected[orig_idx] = p_adj
        rejected[orig_idx] = p_adj < alpha
        prev = p_adj
    return {
        "rejected": rejected,
        "corrected_pvalues": corrected,
        "alpha": float(alpha),
        "m": int(m),
        "formula": "p * (m - k + 1)",
    }


def power_analysis_ttest(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    n_groups: int = 2,
) -> dict:
    """A-priori power analysis for two-sample t-test.

    Computes the required sample size to achieve the target power for a
    given effect size.

    Parameters
    ----------
    effect_size : float
        Cohen's d (or Hedges' g).
    alpha : float
        Type I error rate.
    power : float
        Desired statistical power (1 - beta).
    n_groups : int
        Number of groups (2 for two-sample).

    Returns
    -------
    dict
        {"n_per_group", "total_n", "effect_size", "alpha", "power"}

    Examples
    --------
    >>> result = power_analysis_ttest(effect_size=0.5, power=0.8)
    >>> result["n_per_group"] > 30
    True
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")

    # Use statsmodels if available, else fall back to approximation
    try:
        from statsmodels.stats.power import TTestIndPower

        analysis = TTestIndPower()
        n_per_group = float(
            analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                ratio=1.0,
                alternative="two-sided",
            )
        )
        n_per_group = int(math.ceil(n_per_group))
    except ImportError:
        # Approximation: n_per_group ≈ 2 * (z_alpha/2 + z_beta)^2 / d^2
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n_per_group = int(math.ceil(2 * ((z_alpha + z_beta) ** 2) / (effect_size**2)))

    return {
        "n_per_group": n_per_group,
        "total_n": n_per_group * n_groups,
        "effect_size": float(effect_size),
        "alpha": float(alpha),
        "power": float(power),
        "n_groups": int(n_groups),
    }


# ---------------------------------------------------------------------------
# Convenience wrappers (combined pipelines used by the CLI and the UI)
# ---------------------------------------------------------------------------


def friedman_nemenyi_test(
    scores: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Combined Friedman test + Nemenyi post-hoc.

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape ``(n_datasets, n_methods)``. Higher = better.
    alpha : float
        Significance level. Must be 0.05 or 0.10 (Nemenyi table constraint).

    Returns
    -------
    dict
        Merges the keys from :func:`friedman_test` and :func:`nemenyi_posthoc`
        (the latter's keys are prefixed with ``"nemenyi_"`` to avoid collisions).
    """
    fr = friedman_test(scores, alpha=alpha)
    nm = nemenyi_posthoc(scores, alpha=alpha)
    merged = {**fr}
    for k, v in nm.items():
        merged[f"nemenyi_{k}"] = v
    # Also expose unprefixed for convenience (these don't collide with friedman keys).
    merged["critical_difference"] = nm["critical_difference"]
    merged["average_ranks"] = nm["average_ranks"]
    merged["q_alpha"] = nm["q_alpha"]
    return merged


def wilcoxon_holm_sidak(
    pvalues: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> list[bool]:
    """Convenience: Wilcoxon p-values + Holm-Šídák step-down correction.

    This is a thin wrapper around :func:`holm_sidak_correction` so callers
    can use a single named function from the public API.

    Parameters
    ----------
    pvalues : list of float
        Raw (uncorrected) p-values from pairwise Wilcoxon signed-rank tests.
    alpha : float
        Family-wise significance level.

    Returns
    -------
    list of bool
        For each p-value, whether the null is rejected after Holm-Šídák
        correction.

    Examples
    --------
    >>> wilcoxon_holm_sidak([0.01, 0.04, 0.03, 0.20], alpha=0.05)
    [True, False, False, False]
    """
    return holm_sidak_correction(pvalues, alpha=alpha)["rejected"]
