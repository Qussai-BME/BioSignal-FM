"""Mixed-effects statistical models for biosignal evaluation.

The Friedman + Nemenyi + Wilcoxon-Holm-Šídák pipeline is the project's
primary statistical tool, but the 2025-2026 best-practices trend in ML
evaluation is to *also* fit a mixed-effects model with subject and session
random intercepts. This captures the hierarchical variance structure
(subject > session > fold) that non-parametric tests treat as a black box.

This module wraps :mod:`statsmodels.formula.api.mixedlm` and provides a
clean reporting interface. If ``statsmodels`` is not installed, the
constructor raises a clear ``ImportError`` with installation instructions
(rather than silently falling back to an approximation).

Example
-------
>>> import numpy as np
>>> import pandas as pd
>>> from biosignal_fm.evaluation import MixedEffectsAnalyzer
>>> df = pd.DataFrame({
...     "accuracy": np.random.rand(40),
...     "method":    ["fm"] * 20 + ["baseline"] * 20,
...     "subject":   [f"s{i % 5}" for i in range(40)],
...     "session":   [f"sess{i % 2}" for i in range(40)],
... })
>>> an = MixedEffectsAnalyzer(df)
>>> result = an.fit(formula="accuracy ~ method", groups="subject")
>>> # `result.summary` is the statsmodels MixedLM results object.
>>> # `result.to_dict()` returns a JSON-serializable dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

__all__ = ["MixedEffectsAnalyzer", "MixedEffectsResult"]


@dataclass
class MixedEffectsResult:
    """Container for a fitted mixed-effects model.

    Attributes
    ----------
    formula : str
        The fixed-effects formula, e.g. ``"accuracy ~ method"``.
    groups : str
        The grouping variable name, e.g. ``"subject"``.
    n_observations : int
        Total rows in the fitted dataframe.
    n_groups : int
        Number of unique groups.
    coefficients : dict
        Fixed-effects coefficient estimates (name -> value).
    p_values : dict
        Fixed-effects p-values (name -> value).
    icc : float
        Intraclass correlation coefficient = var_random / (var_random + var_residual).
    aic : float
        Akaike information criterion (lower = better).
    bic : float
        Bayesian information criterion (lower = better).
    summary : Any
        The raw statsmodels MixedLM results object (for advanced users).
    """

    formula: str
    groups: str
    n_observations: int
    n_groups: int
    coefficients: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    icc: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    summary: Any = None

    def to_dict(self) -> dict:
        """JSON-serializable summary (excludes the statsmodels object)."""
        return {
            "formula": self.formula,
            "groups": self.groups,
            "n_observations": self.n_observations,
            "n_groups": self.n_groups,
            "coefficients": self.coefficients,
            "p_values": self.p_values,
            "icc": self.icc,
            "aic": self.aic,
            "bic": self.bic,
        }


class MixedEffectsAnalyzer:
    """Fit a linear mixed-effects model with random intercepts.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format dataframe with one row per (subject, session, fold, method)
        and columns for the response variable and the grouping variable.

    Notes
    -----
    This requires ``statsmodels`` to be installed. Install via::

        pip install biosignal-fm[stats]

    The analyzer is intentionally simple: it wraps ``mixedlm`` and reports
    the most-relevant fields. For complex designs (random slopes, crossed
    random effects, GLMM), use ``statsmodels`` directly.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        try:
            import statsmodels.formula.api as smf  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "MixedEffectsAnalyzer requires statsmodels. Install it with: "
                "pip install biosignal-fm[stats]  (or: pip install statsmodels)"
            ) from e
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas.DataFrame, got {type(df).__name__}")
        self.df = df

    def fit(
        self,
        formula: str,
        groups: str,
        re_formula: str = "1",
    ) -> MixedEffectsResult:
        """Fit a mixed-effects model.

        Parameters
        ----------
        formula : str
            Fixed-effects formula in Patsy syntax, e.g. ``"accuracy ~ method"``.
        groups : str
            Column name for the grouping variable (e.g. ``"subject"``).
        re_formula : str
            Random-effects formula. Default ``"1"`` (random intercept only).
            Use ``"1 + session"`` for a random slope on session.

        Returns
        -------
        MixedEffectsResult
        """
        import statsmodels.formula.api as smf

        if groups not in self.df.columns:
            raise KeyError(
                f"groups column {groups!r} not in dataframe columns: {list(self.df.columns)}"
            )
        model = smf.mixedlm(formula, self.df, groups=self.df[groups], re_formula=re_formula)
        fit = model.fit()

        # Extract fixed-effects table
        fe = fit.fe_params
        pvals = fit.pvalues
        coefficients = {str(name): float(val) for name, val in fe.items()}
        p_values = {str(name): float(val) for name, val in pvals.items()}

        # ICC = var_random / (var_random + var_residual)
        try:
            var_random = float(fit.cov_re.iloc[0, 0])
        except (AttributeError, IndexError, ValueError):
            var_random = 0.0
        var_residual = float(fit.scale) if hasattr(fit, "scale") else 0.0
        icc = var_random / (var_random + var_residual) if (var_random + var_residual) > 0 else 0.0

        return MixedEffectsResult(
            formula=formula,
            groups=groups,
            n_observations=int(fit.nobs),
            n_groups=int(self.df[groups].nunique()),
            coefficients=coefficients,
            p_values=p_values,
            icc=float(icc),
            aic=float(fit.aic) if hasattr(fit, "aic") else float("nan"),
            bic=float(fit.bic) if hasattr(fit, "bic") else float("nan"),
            summary=fit,
        )

    def compare_methods(
        self,
        response: str,
        method_col: str,
        subject_col: str,
        baseline_method: str | None = None,
    ) -> dict[str, MixedEffectsResult]:
        """Convenience: fit one model per (method vs baseline) pair.

        Parameters
        ----------
        response : str
            Response column (e.g. ``"accuracy"``).
        method_col : str
            Method column (e.g. ``"method"``).
        subject_col : str
            Subject column (e.g. ``"subject"``).
        baseline_method : str, optional
            Method name to compare against. If None, uses the first unique
            value in ``method_col`` (alphabetical).

        Returns
        -------
        dict
            Mapping from method name to :class:`MixedEffectsResult`.
        """
        if baseline_method is None:
            baseline_method = sorted(self.df[method_col].unique())[0]
        results: dict[str, MixedEffectsResult] = {}
        for method in self.df[method_col].unique():
            if method == baseline_method:
                continue
            sub_df = self.df[self.df[method_col].isin([baseline_method, method])].copy()
            # Use treatment coding with baseline as reference.
            sub_df[method_col] = pd.Categorical(
                sub_df[method_col], categories=[baseline_method, method]
            )
            an = MixedEffectsAnalyzer(sub_df)
            formula = f"{response} ~ C({method_col}, Treatment)"
            results[method] = an.fit(formula=formula, groups=subject_col)
        return results
