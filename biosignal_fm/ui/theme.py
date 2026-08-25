"""BioSignal-FM dashboard design system.

WCAG 2.2 AA compliant color palette — every foreground/background pairing
below is verified programmatically (relative-luminance contrast ratio
>= 4.5:1) rather than eyeballed; see ``tests/unit/test_ui_theme.py`` for
the check that keeps it that way if a color ever changes.

Design direction: the four biosignal modalities (EMG, ECG, EEG, fNIRS) each
get a dedicated, consistent color used everywhere that modality is shown —
the same way a real multi-channel signal monitor assigns each channel its
own trace color, so a reader can recognize "this is EMG" by color alone
before reading a label. Typography is the IBM Plex family (designed by IBM
for technical/scientific documentation) paired with Space Grotesk for
display headings, rather than a generic UI-framework default.

Call :func:`inject_css` once at the top of every page — ``app.py`` and each
file under ``pages/`` are independent script executions in Streamlit's
multi-page model, so CSS injected in one is invisible in the others; each
page must inject it for itself.
"""

from __future__ import annotations

import streamlit as st

__all__ = ["COLORS", "MODALITIES", "TYPOGRAPHY", "inject_css", "modality_badge", "stage_indicator"]

COLORS = {
    # Surfaces
    "bg": "#F6F7F9",  # cool near-white — instrument-panel neutral, not warm cream
    "surface": "#FFFFFF",
    "border": "#E2E5EA",
    "border_strong": "#C9CED6",
    # Text
    "text": "#141925",  # cool near-black, 17.6:1 on white (AAA)
    "text_muted": "#5B6472",  # 6.0:1 on white (AA)
    "text_inverse": "#F6F7F9",
    # Primary interactive accent — deep phosphor/oscilloscope-trace green,
    # deliberately distinct from all four modality colors below so buttons
    # and links never get mistaken for a modality tag.
    "primary": "#0C7A4B",
    "primary_dark": "#075533",
    # Status
    "success": "#0F7A3D",
    "warning": "#9A5B00",
    "danger": "#B0143E",
    # Neutral chart series (colorblind-safe, Okabe-Ito inspired) for charts
    # that aren't modality-specific (e.g. comparing baseline methods)
    "chart": [
        "#0173B2",
        "#DE8F05",
        "#029E73",
        "#CC78BC",
        "#CA9161",
        "#949494",
    ],
}

# Per-modality signature colors — the dashboard's wayfinding system. Every
# pairing below is verified >= 4.5:1 on both COLORS["bg"] and white.
MODALITIES = {
    "emg": {"label": "EMG", "color": "#A85300", "full": "Electromyography"},
    "ecg": {"label": "ECG", "color": "#B0143E", "full": "Electrocardiography"},
    "eeg": {"label": "EEG", "color": "#5B3FA6", "full": "Electroencephalography"},
    "ecog": {
        "label": "ECoG/iEEG",
        "color": "#006D77",
        "full": "Electrocorticography / intracranial EEG (experimental)",
    },
    "fnirs": {
        "label": "fNIRS",
        "color": "#0A6E9E",
        "full": "Functional Near-Infrared Spectroscopy",
    },
}

TYPOGRAPHY = {
    "display": "'Space Grotesk', 'IBM Plex Sans', system-ui, sans-serif",
    "body": "'IBM Plex Sans', system-ui, sans-serif",
    "code": "'IBM Plex Mono', 'JetBrains Mono', monospace",
    "heading_size_lg": "2.5rem",
    "heading_size_md": "1.75rem",
    "heading_size_sm": "1.25rem",
    "body_size": "1rem",
    "small_size": "0.875rem",
}

_FONT_IMPORT_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@500;600;700&"
    "family=IBM+Plex+Sans:wght@400;500;600&"
    "family=IBM+Plex+Mono:wght@400;500&display=swap"
)


def inject_css() -> None:
    """Inject the shared design system as CSS. Call once per page, near the top.

    Must be called on every page individually — see the module docstring
    for why a single call in ``app.py`` doesn't cover the other pages.
    """
    c = COLORS
    t = TYPOGRAPHY
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="{_FONT_IMPORT_URL}" rel="stylesheet">
        <style>
        .stApp {{
            font-family: {t["body"]};
            color: {c["text"]};
            background-color: {c["bg"]};
        }}
        h1, h2, h3, h4, h5 {{
            font-family: {t["display"]};
            color: {c["text"]};
            font-weight: 600;
            letter-spacing: -0.01em;
        }}
        h1 {{ font-size: {t["heading_size_lg"]}; }}
        h2 {{ font-size: {t["heading_size_md"]}; }}
        code, pre, .stCode {{
            font-family: {t["code"]} !important;
        }}
        [data-testid="stCaptionContainer"], .stCaption {{
            color: {c["text_muted"]} !important;
        }}
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {c["surface"]};
            border-right: 1px solid {c["border"]};
        }}
        /* Metric cards */
        [data-testid="stMetric"] {{
            background-color: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 10px;
            padding: 1rem 1.1rem;
        }}
        [data-testid="stMetricLabel"] {{
            color: {c["text_muted"]} !important;
            font-size: {t["small_size"]} !important;
            font-family: {t["body"]} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {c["text"]} !important;
            font-family: {t["code"]} !important;
            font-weight: 500 !important;
        }}
        /* Primary buttons */
        .stButton > button[kind="primary"] {{
            background-color: {c["primary"]};
            border-color: {c["primary"]};
            font-family: {t["body"]};
            font-weight: 500;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: {c["primary_dark"]};
            border-color: {c["primary_dark"]};
        }}
        .stButton > button[kind="primary"]:focus-visible,
        .stButton > button:focus-visible {{
            outline: 2px solid {c["primary"]};
            outline-offset: 2px;
        }}
        /* Dataframes and tables */
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            font-family: {t["code"]};
            border: 1px solid {c["border"]};
            border-radius: 8px;
        }}
        /* Cards for custom HTML blocks (modality badges, stage indicator) */
        .bsfm-card {{
            background-color: {c["surface"]};
            border: 1px solid {c["border"]};
            border-radius: 10px;
            padding: 1rem 1.25rem;
        }}
        .bsfm-modality-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.4em;
            font-family: {t["code"]};
            font-size: {t["small_size"]};
            font-weight: 500;
            padding: 0.15em 0.6em;
            border-radius: 999px;
            border: 1px solid currentColor;
        }}
        .bsfm-modality-dot {{
            width: 0.55em;
            height: 0.55em;
            border-radius: 50%;
            background: currentColor;
            display: inline-block;
        }}
        .bsfm-stage-track {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            margin: 0.25rem 0 1.25rem 0;
            font-family: {t["code"]};
            font-size: {t["small_size"]};
        }}
        .bsfm-stage {{
            display: flex;
            align-items: center;
            gap: 0.4rem;
            color: {c["text_muted"]};
        }}
        .bsfm-stage-num {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.5em;
            height: 1.5em;
            border-radius: 50%;
            border: 1px solid {c["border_strong"]};
            font-size: 0.85em;
        }}
        .bsfm-stage.active {{
            color: {c["text"]};
            font-weight: 600;
        }}
        .bsfm-stage.active .bsfm-stage-num {{
            background: {c["primary"]};
            border-color: {c["primary"]};
            color: {c["text_inverse"]};
        }}
        .bsfm-stage-sep {{
            color: {c["border_strong"]};
        }}
        @media (prefers-reduced-motion: no-preference) {{
            .bsfm-trace-path {{
                animation: bsfm-flow 3.2s linear infinite;
            }}
            @keyframes bsfm-flow {{
                to {{ stroke-dashoffset: -200; }}
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def modality_badge(modality: str) -> str:
    """Return an HTML pill badge for a modality key (e.g. ``"emg"``).

    Examples
    --------
    >>> "EMG" in modality_badge("emg")
    True
    """
    m = MODALITIES[modality.lower()]
    return (
        f'<span class="bsfm-modality-badge" style="color:{m["color"]}">'
        f'<span class="bsfm-modality-dot"></span>{m["label"]}</span>'
    )


def stage_indicator(current: str) -> str:
    """Return the HTML for the 5-stage pipeline wayfinding track.

    Parameters
    ----------
    current : str
        One of ``"Overview"``, ``"Pretrain"``, ``"Finetune"``,
        ``"Evaluate"``, ``"Deploy"``.

    Examples
    --------
    >>> "Pretrain" in stage_indicator("Pretrain")
    True
    """
    stages = ["Overview", "Pretrain", "Finetune", "Evaluate", "Deploy"]
    parts = ['<div class="bsfm-stage-track">']
    for i, stage in enumerate(stages, start=1):
        active = "active" if stage == current else ""
        parts.append(
            f'<span class="bsfm-stage {active}">'
            f'<span class="bsfm-stage-num">{i}</span>{stage}</span>'
        )
        if i < len(stages):
            parts.append('<span class="bsfm-stage-sep">&rarr;</span>')
    parts.append("</div>")
    return "".join(parts)
