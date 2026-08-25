"""BioSignal-FM Streamlit dashboard — main entry point.

Run with::

    streamlit run biosignal_fm/ui/app.py

This is a thin entry point that sets the page config and renders the
overview page. The 5 sections are organized as separate pages under
``biosignal_fm/ui/pages/`` and Streamlit auto-discovers them.
"""

from __future__ import annotations

import streamlit as st

from biosignal_fm.ui.theme import MODALITIES, inject_css, modality_badge

st.set_page_config(
    page_title="BioSignal-FM",
    page_icon=None,  # no emoji per design guidelines
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# --- Hero: illustrative traces communicate modality diversity only; they are
# not clinical recordings or benchmark evidence. ---
_trace_svg = f"""
<svg viewBox="0 0 800 160" width="100%" height="160" role="img"
     aria-label="Illustrative waveform traces for EMG, ECG, EEG, ECoG, and fNIRS">
  <g stroke-width="2" fill="none" stroke-linecap="round">
    <path d="M0,20 L40,20 L48,4 L56,34 L64,20 L100,20 L800,20"
          stroke="{MODALITIES["emg"]["color"]}" class="bsfm-trace-path"
          stroke-dasharray="6 5" opacity="0.9"/>
    <path d="M0,55 L60,55 L68,55 L74,30 L80,75 L86,45 L92,55 L800,55"
          stroke="{MODALITIES["ecg"]["color"]}" class="bsfm-trace-path"
          stroke-dasharray="6 5" opacity="0.9"/>
    <path d="M0,90 C20,80 40,100 60,90 C80,80 100,100 120,90 C140,80 160,100 180,90 L800,90"
          stroke="{MODALITIES["eeg"]["color"]}" class="bsfm-trace-path"
          stroke-dasharray="6 5" opacity="0.9"/>
    <path d="M0,120 C30,112 60,128 90,120 C120,112 150,128 180,120 L800,120"
          stroke="{MODALITIES["ecog"]["color"]}" class="bsfm-trace-path"
          stroke-dasharray="6 5" opacity="0.9"/>
    <path d="M0,135 C30,131 60,139 90,135 C120,131 150,139 180,135 L800,135"
          stroke="{MODALITIES["fnirs"]["color"]}" class="bsfm-trace-path"
          stroke-dasharray="6 5" opacity="0.9"/>
  </g>
</svg>
"""
st.markdown(_trace_svg, unsafe_allow_html=True)

st.title("BioSignal-FM")
st.caption(
    "Modular multimodal biosignal research platform — core: EMG, EEG, ECG; "
    "experimental: ECoG/iEEG; optional legacy extension: fNIRS."
)

badges = " &nbsp; ".join(modality_badge(k) for k in ["emg", "ecg", "eeg", "ecog", "fnirs"])
st.markdown(f'<div style="margin: 0.5rem 0 1.5rem 0;">{badges}</div>', unsafe_allow_html=True)

st.markdown(
    """
    Use the sidebar to move through the pipeline in order:

    1. **Overview** — this page: system summary and model card
    2. **Pretrain** — launch provenance-labelled SSL pretraining
    3. **Finetune** — fine-tune on a downstream task with explicit LOSO protocol
    4. **Evaluate** — inspect metrics and statistical methods with evidence labels
    5. **Deploy** — ONNX export, quantized benchmarking, REST + WebSocket serving
    """
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Core modalities", value="3")
with col2:
    st.metric(label="Experimental", value="ECoG")
with col3:
    st.metric(label="Optional", value="fNIRS")
with col4:
    st.metric(label="Evidence labels", value="on")
st.caption(
    "Illustrations and synthetic demos verify software paths only. They are not real-data benchmarks "
    "or clinical evidence."
)

st.divider()

st.markdown("### About")
st.markdown(
    """
    <div class="bsfm-card">
    BioSignal-FM is a modular research platform for reproducible signal
    processing, representation learning, cross-subject generalization, and
    downstream task evaluation. It exposes canonical signal contracts and a
    modality registry so datasets, encoders, and research systems can be
    connected without turning the platform into a monolith. It supports future
    foundation-model research; it does not claim a validated foundation model,
    clinical readiness, or benchmark leadership without documented evidence.
    <br><br>
    <b>Author:</b> Qussai Adlbi — Biomedical Engineering
    (Al-Andalus University / Pázmány Péter Catholic University)
    <br>
    <b>License:</b> Apache 2.0
    </div>
    """,
    unsafe_allow_html=True,
)
