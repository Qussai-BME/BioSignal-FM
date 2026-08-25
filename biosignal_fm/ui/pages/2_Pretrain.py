"""Pretrain page: SSL pretraining launcher and loss monitor."""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st
from biosignal_fm.ui.theme import inject_css, stage_indicator

st.set_page_config(page_title="Pretrain — BioSignal-FM", layout="wide")
inject_css()
st.markdown(stage_indicator("Pretrain"), unsafe_allow_html=True)

st.title("Pretrain")
st.markdown("### Self-Supervised Pretraining")

st.markdown(
    """
    Configure SSL pretraining for a representation encoder. The combined loss is::

        L = reconstruction_weight * MSE(masked_patches) + contrastive_weight * NT-Xent(view_a, view_b)

    Span masking (mean span length 8) is used rather than random masking
    to better capture temporal structure of biosignals.
    """
)

with st.sidebar:
    st.markdown("### Pretraining Config")
    d_model = st.number_input("d_model", value=512, step=64)
    n_layers = st.number_input("n_layers", value=12, step=1)
    n_heads = st.number_input("n_heads", value=8, step=1)
    batch_size = st.number_input("batch_size", value=64, step=8)
    lr = st.number_input("learning_rate", value=1e-4, format="%.1e")
    max_steps = st.number_input("max_steps", value=100_000, step=1000)

st.markdown("---")
st.markdown("#### Live demo")
st.warning(
    "SYNTHETIC DEMO ONLY: this verifies an engineering path. It is not real-data "
    "pretraining, a benchmark, or evidence of a foundation model."
)
st.caption(
    "Runs a real (but tiny, step-capped) pretraining loop on synthetic data, "
    "on this session's CPU — not a simulation. Uses a small fixed-size model "
    "regardless of the sidebar config above, so it finishes in seconds; the "
    "sidebar config is for the full run generated below instead."
)

if st.button("Run live demo (15 real steps)", type="primary"):
    import torch
    from biosignal_fm.config import Modality, ModelConfig, TrainingConfig
    from biosignal_fm.models import ContrastiveHead, FoundationModel, SpanMaskedReconstructionHead
    from biosignal_fm.training import SSLPretrainer

    demo_cfg = ModelConfig(
        d_model=64, n_heads=4, n_layers=2, d_ff=128, patch_length=32, patch_stride=16
    )
    train_cfg = TrainingConfig(batch_size=8, learning_rate=1e-4, warmup_steps=2, max_steps=15)
    n_channels_per_modality = {m.value: 16 for m in Modality}
    signal_length = 400
    n_patches = 1 + (signal_length - demo_cfg.patch_length) // demo_cfg.patch_stride

    model = FoundationModel(demo_cfg, n_channels_per_modality)
    ssl_head = SpanMaskedReconstructionHead(
        d_model=demo_cfg.d_model, patch_length=demo_cfg.patch_length, n_channels=16
    )
    contrastive_head = ContrastiveHead(d_model=demo_cfg.d_model)
    trainer = SSLPretrainer(
        model=model,
        ssl_head=ssl_head,
        contrastive_head=contrastive_head,
        config=train_cfg,
        model_config=demo_cfg,
    )

    rng = torch.Generator().manual_seed(42)
    progress = st.progress(0.0, text="Starting...")
    chart_placeholder = st.empty()
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for step in range(train_cfg.max_steps):
        batch = (
            torch.randn(train_cfg.batch_size, 16, signal_length, generator=rng),
            torch.zeros(train_cfg.batch_size, dtype=torch.long),
            torch.randn(train_cfg.batch_size, n_patches, 16, demo_cfg.patch_length, generator=rng),
        )
        metrics = trainer.train_step(batch)
        history.append(metrics)
        progress.progress(
            (step + 1) / train_cfg.max_steps,
            text=f"Step {step + 1}/{train_cfg.max_steps} — loss={metrics['loss']:.4f}",
        )
        # Redraw the chart every few steps rather than every single step —
        # full chart redraws are the expensive part, not the training itself
        # (15 real steps train in ~3-4s; redrawing 15 times was the slow part).
        if step % 3 == 0 or step == train_cfg.max_steps - 1:
            df = pd.DataFrame(history)[["loss", "mse", "contrastive"]]
            df.index.name = "step"
            chart_placeholder.line_chart(df)

    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    st.success(
        f"Done — {train_cfg.max_steps} real training steps in {elapsed:.1f}s "
        f"on a {n_params:,}-parameter model. Final loss: {history[-1]['loss']:.4f} "
        f"(started at {history[0]['loss']:.4f})."
    )

st.markdown("---")
st.markdown("#### Full run")
st.caption(
    "Generates a configuration. A real run requires a real-data adapter and documented provenance; "
    "the displayed CLI command is explicitly a synthetic smoke test."
)

if st.button("Generate config for full run"):
    from pathlib import Path

    import yaml as _yaml
    from biosignal_fm.config import ExperimentConfig, ModelConfig, TrainingConfig

    cfg = ExperimentConfig(
        name="exp001",
        output_dir=Path("runs/exp001"),
        model=ModelConfig(d_model=int(d_model), n_layers=int(n_layers), n_heads=int(n_heads)),
        training=TrainingConfig(
            batch_size=int(batch_size), learning_rate=float(lr), max_steps=int(max_steps)
        ),
    )
    yaml_text = _yaml.safe_dump(cfg.to_dict(), default_flow_style=False, sort_keys=False)
    st.code(yaml_text, language="yaml")
    st.download_button("Download config.yaml", yaml_text, file_name="exp.yaml", mime="text/yaml")
    st.code(
        "bsfm pretrain --config exp.yaml --output-dir runs/exp001 --synthetic-demo",
        language="bash",
    )
