"""Finetune page: fine-tuning with LOSO evaluation."""

from __future__ import annotations

import streamlit as st
from biosignal_fm.ui.theme import inject_css, stage_indicator

st.set_page_config(page_title="Finetune — BioSignal-FM", layout="wide")
inject_css()
st.markdown(stage_indicator("Finetune"), unsafe_allow_html=True)

st.title("Finetune")
st.markdown("### Downstream Fine-Tuning")

st.markdown(
    """
    Fine-tune a pretrained encoder checkpoint on a downstream task
    (e.g. 8-class EMG gesture recognition). The default protocol is
    **Leave-One-Subject-Out (LOSO)** to prevent subject-level leakage.

    Three strategies are supported:

    - **Linear probe** — freeze encoder, train only a linear head
    - **Partial** — unfreeze last K layers
    - **Full** — unfreeze everything
    """
)

strategy = st.radio(
    "Fine-tuning strategy",
    options=["linear", "partial", "full"],
    horizontal=True,
)

if strategy == "partial":
    n_unfrozen = st.slider("Unfrozen layers", min_value=1, max_value=12, value=4)
    st.caption(f"Last {n_unfrozen} layers will be unfrozen.")

st.markdown("---")
st.markdown("#### Live demo")
st.warning(
    "SYNTHETIC DEMO ONLY: LOSO mechanics are exercised on random data and a random "
    "initialization. This is not a benchmark, transfer result, or scientific inference."
)
st.caption(
    "Runs the real LOSO fine-tuning pipeline (same code path as `biosignal-fm "
    "finetune`) on a small **randomly-initialized** model — not a pretrained "
    "checkpoint, since none ships with the package. Accuracy will hover near "
    "chance level (1/n_classes) as a result; that's expected. This proves the "
    "pipeline itself — fresh-per-fold training, real confusion-matrix "
    "aggregation across folds — actually works, not that the model is good."
)

if st.button("Run live LOSO demo", type="primary"):
    import numpy as np
    import pandas as pd
    import torch
    from biosignal_fm.config import Modality, ModelConfig, TrainingConfig
    from biosignal_fm.evaluation import LeaveOneSubjectOutCV, confusion_matrix
    from biosignal_fm.models import FoundationModel, LinearProbe
    from biosignal_fm.training import FineTuner

    n_classes = 8
    n_channels = 16
    signal_length = 64
    n_subjects = 6
    samples_per_subj = 8
    labels_names = ["rest", "thumb", "index", "fist", "pinch", "pron", "supin", "lat"]

    demo_cfg = ModelConfig(
        d_model=32, n_heads=4, n_layers=2, d_ff=64, patch_length=16, patch_stride=8
    )
    train_cfg = TrainingConfig(max_steps=3, batch_size=8, learning_rate=1e-3)

    rng = torch.Generator().manual_seed(42)
    samples: list[tuple[torch.Tensor, int, int, int]] = []
    for subj in range(n_subjects):
        for _ in range(samples_per_subj):
            signal = torch.randn(n_channels, signal_length, generator=rng)
            label = int(torch.randint(0, n_classes, (1,), generator=rng).item())
            samples.append((signal, 0, label, subj))
    subjects_arr = [s[3] for s in samples]

    cv = LeaveOneSubjectOutCV()
    n_ch = {m.value: n_channels for m in Modality}
    fold_results = []
    y_true_folds, y_pred_folds = [], []
    progress = st.progress(0.0, text="Starting LOSO folds...")

    folds = list(cv.split(subjects_arr))
    for i, (train_idx, test_idx) in enumerate(folds):
        # Fresh model per fold -- same fix applied to `biosignal-fm finetune`
        # (reusing a model/optimizer across folds would let each "held-out"
        # subject leak into earlier folds' training, invalidating LOSO).
        model = FoundationModel(demo_cfg, n_ch)
        head = LinearProbe(d_model=demo_cfg.d_model, n_classes=n_classes)
        ft = FineTuner(model, head, strategy=strategy, config=train_cfg)  # type: ignore[arg-type]

        train_batch = (
            torch.stack([samples[j][0] for j in train_idx]),
            torch.tensor([samples[j][1] for j in train_idx], dtype=torch.long),
            torch.tensor([samples[j][2] for j in train_idx], dtype=torch.long),
        )
        for _ in range(train_cfg.max_steps):
            ft.train_step(train_batch)

        test_signals = torch.stack([samples[j][0] for j in test_idx])
        test_mods = torch.tensor([samples[j][1] for j in test_idx], dtype=torch.long)
        test_labels = torch.tensor([samples[j][2] for j in test_idx], dtype=torch.long)
        with torch.no_grad():
            model.eval()
            head.eval()
            cls_token, _ = model(test_signals, test_mods)
            preds = head(cls_token).argmax(dim=-1)
        acc = (preds == test_labels).float().mean().item()

        fold_results.append({"Subject": f"S{subjects_arr[test_idx[0]]:02d}", "Accuracy": acc})
        y_true_folds.append(test_labels.tolist())
        y_pred_folds.append(preds.tolist())
        progress.progress((i + 1) / len(folds), text=f"Fold {i + 1}/{len(folds)}")

    accs = np.array([r["Accuracy"] for r in fold_results])
    st.markdown("### LOSO Results (live — random-init model)")
    st.dataframe(pd.DataFrame(fold_results), width="stretch")
    col1, col2 = st.columns(2)
    col1.metric("Mean Accuracy", f"{accs.mean():.4f}")
    col2.metric("Std", f"{accs.std():.4f}")
    st.caption(f"Chance level for {n_classes} classes ≈ {1 / n_classes:.4f}.")

    st.markdown("### Confusion Matrix (aggregated across all LOSO folds)")
    cm = confusion_matrix(y_true_folds, y_pred_folds, n_classes=n_classes)
    cm_df = pd.DataFrame(cm, index=labels_names, columns=labels_names)
    st.dataframe(cm_df, width="stretch")

st.markdown("---")
st.markdown("#### Full run (real pretrained checkpoint)")
st.code(
    "biosignal-fm finetune \\\n"
    "    --checkpoint path/to/pretrained.pt \\\n"
    "    --modality emg \\\n"
    "    --n-classes 8 \\\n"
    f"    --strategy {strategy} \\\n"
    "    --output-dir runs/finetune/",
    language="bash",
)
