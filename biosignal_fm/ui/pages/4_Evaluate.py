"""Evaluate page: cross-modal transfer matrix and statistical tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from biosignal_fm.ui.theme import inject_css, modality_badge, stage_indicator

st.set_page_config(page_title="Evaluate — BioSignal-FM", layout="wide")
inject_css()
st.markdown(stage_indicator("Evaluate"), unsafe_allow_html=True)

st.title("Evaluate")
st.markdown("### Cross-Modal Transfer & Statistics")

st.markdown(
    """
    V4 provides the interfaces needed to study cross-modal transfer and fusion,
    but it does not ship a validated cross-modal experiment. Results should be
    added here only after real-data provenance, a frozen protocol, and a
    reproducible run manifest are available.
    """
)

_legend = " &nbsp; ".join(modality_badge(k) for k in ["emg", "ecg", "eeg", "ecog", "fnirs"])
st.markdown(_legend, unsafe_allow_html=True)
st.info(
    "No validated cross-modal benchmark is bundled with V4. Synthetic demonstrations "
    "below verify statistics code only and must not be interpreted as scientific inference."
)

st.markdown("---")
st.markdown("### Live statistics demo")
st.markdown(
    """
    The Friedman + Nemenyi and Wilcoxon + Holm-Šídák sections below are
    **live and real** — not the illustration above. Clicking the button
    generates synthetic per-fold scores for three methods (same approach
    `biosignal-fm evaluate` uses) and runs the actual statistics functions
    on them, so the numbers you see are genuinely computed, just on
    synthetic rather than experimentally-measured accuracies.
    """
)

if st.button("Run live statistics demo", type="primary"):
    from biosignal_fm.evaluation import (
        friedman_nemenyi_test,
        hedges_g,
        wilcoxon_holm_sidak,
    )
    from scipy import stats as ss

    method_names = ["BioSignal-FM encoder", "CNN1D baseline", "LDA+TD baseline"]
    n_subjects = 8
    rng = np.random.default_rng(42)
    # Rows = subjects/folds, columns = methods (the orientation
    # friedman_nemenyi_test documents and expects).
    scores = np.array(
        [
            [0.72 + 0.02 * (i % 3) + 0.01 * j + rng.normal(0, 0.01) for i in range(3)]
            for j in range(n_subjects)
        ]
    )

    fn = friedman_nemenyi_test(scores, alpha=0.05)
    st.markdown("#### Friedman + Nemenyi Test")
    st.markdown(
        f"**k (methods):** {fn['n_methods']}  |  **n (datasets):** {fn['n_datasets']}  |  "
        f"**Critical difference:** {fn['critical_difference']:.4f}"
    )
    rank_df = pd.DataFrame(
        {"Method": method_names, "Average rank": [f"{r:.2f}" for r in fn["average_ranks"]]}
    )
    st.dataframe(rank_df, width="stretch")

    st.markdown("#### Wilcoxon + Holm-Šídák Correction")
    st.markdown(
        """
        Pairwise significance (method 0 vs each other), corrected with
        Holm-Šídák step-down: `corrected_p = 1 - (1 - p)^(m - k + 1)`
        (**not** the Bonferroni-Holm formula `p * (m - k + 1)`).
        """
    )
    pvalues = []
    for i in range(1, scores.shape[1]):
        try:
            p = float(ss.wilcoxon(scores[:, 0], scores[:, i]).pvalue)
        except ValueError:
            p = 1.0
        pvalues.append(p)
    rejected = wilcoxon_holm_sidak(pvalues, alpha=0.05)
    corr_df = pd.DataFrame(
        {
            "Comparison": [f"{method_names[0]} vs {method_names[i]}" for i in range(1, 3)],
            "Raw p": pvalues,
            "Rejected (alpha=0.05)": rejected,
        }
    )
    st.dataframe(corr_df, width="stretch")

    g = hedges_g(scores[:, 0], scores[:, 1])
    st.markdown(f"**Hedges' g** ({method_names[0]} vs {method_names[1]}): {g:.4f}")
