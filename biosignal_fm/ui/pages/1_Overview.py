"""Overview page: architecture, model card, dataset summary."""

from __future__ import annotations

import streamlit as st
from biosignal_fm.ui.theme import inject_css, modality_badge, stage_indicator

st.set_page_config(page_title="Overview — BioSignal-FM", layout="wide")
inject_css()
st.markdown(stage_indicator("Overview"), unsafe_allow_html=True)

st.title("Overview")
st.markdown("### System Architecture")

modality_row = " &nbsp; ".join(modality_badge(k) for k in ["emg", "ecg", "eeg", "ecog", "fnirs"])
st.markdown(
    f"""
    BioSignal-FM V4 uses explicit contracts and adapters. Each modality —
    {modality_row} — enters through a registered edge adapter and becomes a
    canonical `Signal` with structured provenance. EMG, EEG, and ECG are core;
    ECoG/iEEG is experimental; fNIRS is a legacy-compatible optional extension.

    ### V4 safeguards
    1. **Library-independent core** — signal contracts do not import readers, UI, HTTP, or PyTorch.
    2. **Data provenance** — synthetic data is labeled and cannot be presented as a benchmark.
    3. **Explicit modalities** — capabilities and optional dependencies live in the registry.
    4. **Correct multimodal ordering** — representation fusion occurs before a task head.
    5. **Protocol-aware evaluation** — LOSO/LODO and statistical tools remain separate from benchmark claims.
    """,
    unsafe_allow_html=True,
)

st.markdown("### Implementation inventory")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Base**")
    st.markdown(
        """
        | Field | Value |
        |---|---|
        | Architecture | Transformer encoder, 12 layers, d=512, h=8 |
        | Status | Encoder implementation; no pretrained-model claim |
        | Intended role | Representation learning experiments |
        """
    )
with col2:
    st.markdown("**Distilled** (CPU deployment)")
    st.markdown(
        """
        | Field | Value |
        |---|---|
        | Architecture | Transformer encoder, 6 layers, d=256, h=8 |
        | Status | Compact encoder implementation; no measured deployment claim |
        | Intended role | Resource-constrained research experiments |
        """
    )

st.markdown(
    """
    | Field | Value |
    |---|---|
    | Input | Canonical `Signal` objects or an adapter-compatible source |
    | Output | Representations and task outputs with provenance labels |
    | Evidence boundary | Synthetic demos verify software paths, not scientific benchmarks |
    | License | Apache 2.0 |
    | Author | Qussai Adlbi |
    """
)
