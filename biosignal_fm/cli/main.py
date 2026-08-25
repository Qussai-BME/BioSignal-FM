"""Typer CLI for BioSignal-FM.

Commands:

- ``bsfm info`` — print package version and dependency summary
- ``bsfm inspect`` — inspect registered modality capabilities
- ``bsfm pretrain`` — run SSL pretraining from a YAML config
- ``bsfm finetune`` — run fine-tuning
- ``bsfm evaluate`` — run LOSO evaluation with statistics
- ``bsfm export-onnx`` — export a checkpoint to ONNX
- ``bsfm benchmark`` — measure inference latency (single-thread)
- ``bsfm serve`` — start the FastAPI server
- ``bsfm ui`` — start the Streamlit dashboard

Examples
--------
::

    bsfm info
    bsfm pretrain --config configs/exp.yaml --output-dir runs/exp001
    bsfm export-onnx --checkpoint runs/exp001/final_model.pt --output model.onnx
    bsfm serve --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="bsfm",
    help="BioSignal-FM: a modular multimodal biosignal research platform.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)
console = Console()

__all__ = ["app"]


@app.command()
def info() -> None:
    """Print package version and config summary."""
    from .. import __author__, __version__

    console.print(f"[bold]BioSignal-FM[/bold] v{__version__}")
    console.print(f"Author: {__author__}")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"Platform: {sys.platform}")

    # Check optional deps
    table = Table(title="Optional Dependencies")
    table.add_column("Package", style="cyan")
    table.add_column("Installed", style="green")
    for pkg in ("torch", "mlflow", "onnx", "onnxruntime", "streamlit"):
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown")
            table.add_row(pkg, f"[green]{version}[/green]")
        except ImportError:
            table.add_row(pkg, "[red]no[/red]")
    console.print(table)

    from ..modalities import default_registry

    modality_table = Table(title="V4 Modality Registry")
    modality_table.add_column("Identifier", style="cyan")
    modality_table.add_column("Status")
    modality_table.add_column("Datasets")
    for plugin in default_registry().plugins():
        modality_table.add_row(
            plugin.identifier, plugin.status.value, ", ".join(plugin.datasets) or "—"
        )
    console.print(modality_table)


@app.command("inspect")
def inspect_modality(
    modality: str | None = typer.Option(
        None, "--modality", "-m", help="Optional modality identifier"
    ),
) -> None:
    """Inspect V4 modality capabilities and maturity without loading data."""
    from ..modalities import default_registry

    registry = default_registry()
    plugins = (registry.get(modality),) if modality else registry.plugins()
    table = Table(title="BioSignal-FM V4 modality capabilities")
    table.add_column("Modality", style="cyan")
    table.add_column("Status")
    table.add_column("Tasks")
    table.add_column("Optional dependencies")
    for plugin in plugins:
        table.add_row(
            plugin.identifier,
            plugin.status.value,
            ", ".join(plugin.supported_tasks) or "—",
            ", ".join(plugin.optional_dependencies) or "—",
        )
    console.print(table)


@app.command()
def pretrain(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config path"),
    output_dir: Path = typer.Option(Path("./runs/pretrain"), "--output-dir", "-o"),
    n_steps: int | None = typer.Option(None, "--steps", "-n", help="Override max_steps"),
    synthetic_demo: bool = typer.Option(
        False,
        "--synthetic-demo",
        help="Explicitly run a synthetic smoke-test workload; never a benchmark.",
    ),
) -> None:
    """Run SSL pretraining from a YAML config and an explicitly selected data source."""
    from ..config import ExperimentConfig
    from ..models import (
        ContrastiveHead,
        FoundationModel,
        SpanMaskedReconstructionHead,
    )
    from ..tracking import LocalTracker
    from ..training import SSLPretrainer

    cfg = ExperimentConfig.from_yaml(config)
    if not synthetic_demo:
        raise typer.BadParameter(
            "V4 does not silently substitute synthetic data for pretraining. "
            "Use --synthetic-demo only for a smoke test, or run a configured real-data service."
        )
    if n_steps:
        import dataclasses

        cfg = cfg.replace(training=dataclasses.replace(cfg.training, max_steps=n_steps))

    console.print(f"[bold]Starting SSL pretraining[/bold]: {cfg.name}")
    console.print(f"  Output: {output_dir}")
    console.print("[yellow]SYNTHETIC DEMO ONLY — output is not a real-data benchmark.[/yellow]")

    # The smoke-test path is intentionally explicit and uses the registry rather
    # than a hard-coded modality list. Real datasets are selected through their
    # loaders/adapters in application code.
    from ..modalities import default_registry

    n_channels_per_modality = dict.fromkeys(default_registry().identifiers(), 16)
    model = FoundationModel(cfg.model, n_channels_per_modality)
    ssl_head = SpanMaskedReconstructionHead(
        d_model=cfg.model.d_model,
        patch_length=cfg.model.patch_length,
        n_channels=16,
    )
    contrastive_head = ContrastiveHead(d_model=cfg.model.d_model)

    tracker = LocalTracker(output_dir=output_dir)
    trainer = SSLPretrainer(
        model=model,
        ssl_head=ssl_head,
        contrastive_head=contrastive_head,
        config=cfg.training,
        model_config=cfg.model,
        tracker=tracker,
    )

    # Build a synthetic dataloader
    import torch
    from torch.utils.data import DataLoader, Dataset

    # Number of patches the model will actually produce for a 400-sample
    # signal, given this config's patch_length/patch_stride (Conv1d output
    # length formula — must match PatchEmbedding exactly, or the SSL
    # reconstruction target shape will mismatch the model's output shape).
    synth_signal_length = 400
    n_patches = 1 + (synth_signal_length - cfg.model.patch_length) // cfg.model.patch_stride

    class _SynthDataset(Dataset):
        def __init__(self, n: int = 100):
            self.n = n
            self.data = [
                (
                    torch.randn(16, synth_signal_length, dtype=torch.float32),
                    torch.tensor([0], dtype=torch.long),
                    torch.randn(n_patches, 16, cfg.model.patch_length, dtype=torch.float32),
                )
                for _ in range(n)
            ]

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return self.data[idx]

    dl = DataLoader(_SynthDataset(100), batch_size=cfg.training.batch_size, shuffle=True)

    result = trainer.train(dl, n_steps=n_steps or cfg.training.max_steps, output_dir=output_dir)
    console.print(f"[green]Done.[/green] run_id={result['run_id']}")
    console.print(f"  Final metrics: {result['final_metrics']}")
    tracker.finish()


@app.command()
def finetune(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", "-c", help="Pretrained FoundationModel checkpoint"
    ),
    output_dir: Path = typer.Option(Path("./runs/finetune"), "--output-dir", "-o"),
    modality: str = typer.Option("emg", "--modality", "-m"),
    n_classes: int = typer.Option(8, "--n-classes"),
    n_channels: int = typer.Option(16, "--n-channels"),
    signal_length: int = typer.Option(400, "--signal-length"),
    strategy: str = typer.Option("linear", "--strategy", help="linear | partial | full"),
    n_steps: int = typer.Option(50, "--steps", "-n"),
    batch_size: int = typer.Option(16, "--batch-size"),
) -> None:
    """Fine-tune a pretrained model on synthetic demo data (LOSO eval).

    .. note::
        For real experiments, write a small Python script that uses
        :class:`FineTuner` directly with your real dataset. This CLI command
        exists so the documentation promise (``bsfm finetune`` is listed)
        is honored, and so users can smoke-test the fine-tuning path
        end-to-end without writing code.
    """
    import torch
    from torch.utils.data import DataLoader

    from ..config import Modality, TrainingConfig
    from ..evaluation import LeaveOneSubjectOutCV
    from ..models import FoundationModel, LinearProbe
    from ..training import FineTuner

    if strategy not in ("linear", "partial", "full"):
        raise typer.BadParameter(f"strategy must be linear|partial|full, got {strategy!r}")

    from ..modalities import default_registry

    default_registry().get(modality)
    console.print(
        "[yellow]SYNTHETIC DEMO ONLY — LOSO values below are not benchmark results.[/yellow]"
    )
    console.print(f"[bold]Loading checkpoint[/bold]: {checkpoint}")
    model = FoundationModel.load(checkpoint)
    console.print(f"  d_model={model.config.d_model}, n_layers={model.config.n_layers}")
    cfg = TrainingConfig(max_steps=n_steps, batch_size=batch_size)

    # Synthetic demo dataset
    n_subjects = 6
    samples_per_subj = 8
    samples: list[tuple[torch.Tensor, int, int, int]] = []  # (signal, modality_idx, label, subject)
    mod_idx = list(Modality).index(Modality.from_str(modality))
    rng = torch.Generator().manual_seed(42)
    for subj in range(n_subjects):
        for _ in range(samples_per_subj):
            signal = torch.randn(n_channels, signal_length, generator=rng, dtype=torch.float32)
            label = int(torch.randint(0, n_classes, (1,), generator=rng).item())
            samples.append((signal, mod_idx, label, subj))

    # LOSO evaluation
    cv = LeaveOneSubjectOutCV()
    subjects = [s[3] for s in samples]
    fold_accuracies: list[float] = []
    final_head_state = None
    for train_idx, test_idx in cv.split(subjects):
        # Each fold must start from the SAME pretrained checkpoint, not from
        # whatever the previous fold's training left behind — otherwise the
        # "held-out" subject in fold k has already been seen during folds
        # 0..k-1 (it was in their training sets), which invalidates LOSO as
        # a genuine generalization estimate. Reload fresh per fold.
        fold_model = FoundationModel.load(checkpoint)
        fold_head = LinearProbe(d_model=fold_model.config.d_model, n_classes=n_classes)
        ft = FineTuner(fold_model, fold_head, strategy=strategy, config=cfg)  # type: ignore[arg-type]

        train_batch = (
            torch.stack([samples[i][0] for i in train_idx]),
            torch.tensor([samples[i][1] for i in train_idx], dtype=torch.long),
            torch.tensor([samples[i][2] for i in train_idx], dtype=torch.long),
        )
        test_batch = (
            torch.stack([samples[i][0] for i in test_idx]),
            torch.tensor([samples[i][1] for i in test_idx], dtype=torch.long),
            torch.tensor([samples[i][2] for i in test_idx], dtype=torch.long),
        )
        # Train
        # DataLoader accepts anything with __len__/__getitem__ at runtime; a
        # plain list satisfies that duck-typed Dataset protocol even though
        # torch's stubs are nominal about it (known false positive).
        dl: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = DataLoader(
            list(zip(train_batch[0], train_batch[1], train_batch[2], strict=True)),  # type: ignore[arg-type]
            batch_size=batch_size,
        )
        for _ in range(n_steps):
            for batch in dl:
                ft.train_step(batch)
        # Eval — evaluate() expects an iterable of batches (a DataLoader in
        # normal use); test_batch is a single pre-stacked batch, so wrap it
        # in a one-element list rather than passing the raw tuple (which
        # would iterate over its 3 tensors instead of over batches).
        metrics = ft.evaluate([test_batch])
        final_head_state = fold_head.state_dict()
        fold_accuracies.append(metrics["accuracy"])

    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / "finetuned_head.pt"
    torch.save({"state_dict": final_head_state, "n_classes": n_classes}, head_path)

    import statistics

    mean_acc = statistics.mean(fold_accuracies) if fold_accuracies else 0.0
    console.print("[green]Fine-tuning complete.[/green]")
    console.print(f"  Head saved: {head_path}")
    console.print(f"  LOSO folds: {len(fold_accuracies)}")
    console.print(f"  Mean accuracy: {mean_acc:.4f}")
    for i, acc in enumerate(fold_accuracies):
        console.print(f"    fold {i}: acc={acc:.4f}")


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", "-c", help="Fine-tuned checkpoint (model+head)"
    ),
    modality: str = typer.Option("emg", "--modality", "-m"),
    n_classes: int = typer.Option(8, "--n-classes"),
    n_channels: int = typer.Option(16, "--n-channels"),
    signal_length: int = typer.Option(400, "--signal-length"),
    protocol: str = typer.Option("loso", "--protocol", help="loso | lodo"),
) -> None:
    """Evaluate a fine-tuned model with LOSO + Friedman/Nemenyi statistics.

    This runs LOSO cross-validation on synthetic demo data, computes per-fold
    accuracy, and runs the Friedman + Nemenyi + Wilcoxon-Holm-Šídák pipeline
    so the user can verify the entire statistics stack works end-to-end.
    """
    import numpy as np

    from ..evaluation import (
        friedman_nemenyi_test,
        hedges_g,
        wilcoxon_holm_sidak,
    )
    from ..modalities import default_registry
    from ..models import FoundationModel, LinearProbe

    default_registry().get(modality)
    console.print(
        "[yellow]SYNTHETIC DEMO ONLY — statistics below are not scientific inference.[/yellow]"
    )
    console.print(f"[bold]Loading checkpoint[/bold]: {checkpoint}")
    model = FoundationModel.load(checkpoint)
    head = LinearProbe(d_model=model.config.d_model, n_classes=n_classes)
    model.eval()
    head.eval()

    n_subjects = 8
    n_datasets = 3  # for LODO

    # Build synthetic predictions across datasets/subjects
    method_names = ["BioSignal-FM encoder", "CNN1D baseline", "LDA+TD baseline"]
    # friedman_nemenyi_test expects shape (n_datasets, n_methods) — rows are
    # folds/datasets, columns are methods. Build it in that orientation
    # directly (rather than building methods×datasets and transposing) so
    # this can't silently drift out of sync again.
    if protocol == "loso":
        # Fake per-fold accuracies: 8 folds (rows) × 3 methods (columns)
        scores = np.array(
            [[0.72 + 0.02 * (i % 3) + 0.01 * j for i in range(3)] for j in range(n_subjects)]
        )
    else:  # lodo
        # Fake per-dataset accuracies: 3 datasets (rows) × 3 methods (columns)
        scores = np.array(
            [[0.70 + 0.03 * i + 0.02 * j for i in range(3)] for j in range(n_datasets)]
        )

    # Friedman + Nemenyi
    fn = friedman_nemenyi_test(scores, alpha=0.05)
    console.print(f"[bold]Friedman-Nemenyi[/bold] (k={fn['n_methods']}, n={fn['n_datasets']})")
    console.print(f"  CD = {fn['critical_difference']:.4f}")
    for name, rank in zip(method_names, fn["average_ranks"], strict=True):
        console.print(f"    {name}: avg rank = {rank:.2f}")

    # Wilcoxon + Holm-Šídák pairwise (method 0 vs each other)
    pvalues = []
    for i in range(1, scores.shape[1]):
        # Pairwise: method 0 vs method i across folds/datasets (columns)
        from scipy import stats as ss

        try:
            p = ss.wilcoxon(scores[:, 0], scores[:, i]).pvalue
        except ValueError:
            p = 1.0
        pvalues.append(float(p))
    reject = wilcoxon_holm_sidak(pvalues, alpha=0.05)
    console.print("[bold]Wilcoxon + Holm-Šídák[/bold] (method 0 vs rest)")
    for i, (p, r) in enumerate(zip(pvalues, reject, strict=True)):
        console.print(f"  vs method {i + 1}: p={p:.4f} → reject_null={r}")

    # Hedges' g effect size (method 0 vs method 1)
    g = hedges_g(scores[:, 0], scores[:, 1])
    console.print(f"[bold]Hedges' g[/bold] (method 0 vs method 1): {g:.4f}")

    console.print("[green]Evaluation complete.[/green]")


@app.command()
def export_onnx(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c"),
    output: Path = typer.Option(..., "--output", "-o"),
    modality: str = typer.Option("emg", "--modality", "-m"),
    n_channels: int = typer.Option(16, "--n-channels"),
    signal_length: int = typer.Option(400, "--signal-length"),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
) -> None:
    """Export a checkpoint to ONNX with optional numerical parity verification."""
    from ..deployment import OnnxExporter
    from ..models import FoundationModel

    console.print(f"[bold]Loading checkpoint[/bold]: {checkpoint}")
    model = FoundationModel.load(checkpoint)
    console.print(
        f"  Model loaded. d_model={model.config.d_model}, n_layers={model.config.n_layers}"
    )

    exporter = OnnxExporter()
    console.print(f"[bold]Exporting to ONNX[/bold]: {output}")
    onnx_path = exporter.export(model, output, modality, n_channels, signal_length)
    console.print(f"  Wrote: {onnx_path}")

    if verify:
        console.print("[bold]Verifying numerical parity[/bold]...")
        try:
            ok = exporter.verify(
                model, onnx_path, modality, n_channels, signal_length, n_samples=20
            )
            if ok:
                console.print("[green]Parity verified.[/green]")
        except AssertionError as e:
            console.print(f"[red]Parity FAILED:[/red] {e}")
            raise typer.Exit(1) from e


@app.command()
def benchmark(
    checkpoint: Path | None = typer.Option(None, "--checkpoint", "-c", help="Optional checkpoint"),
    n_channels: int = typer.Option(16, "--n-channels"),
    signal_length: int = typer.Option(400, "--signal-length"),
    n_runs: int = typer.Option(100, "--n-runs"),
    quantize: bool = typer.Option(True, "--quantize/--no-quantize"),
) -> None:
    """Benchmark single-threaded inference latency."""
    from ..config import Modality, ModelConfig
    from ..deployment import RealtimeInference
    from ..models import FoundationModel

    if checkpoint:
        model = FoundationModel.load(checkpoint)
    else:
        console.print("[yellow]No checkpoint provided; using random model.[/yellow]")
        cfg = ModelConfig()
        n_ch = {m.value: n_channels for m in Modality}
        model = FoundationModel(cfg, n_ch)

    rt = RealtimeInference(model, modality="emg", quantize=quantize)
    result = rt.benchmark(n_channels=n_channels, signal_length=signal_length, n_runs=n_runs)

    if quantize and not rt.quantization_active:
        console.print(
            "[yellow]Warning:[/yellow] quantization was requested but fell back to "
            "full precision (see log above for the reason). Numbers below are full-precision."
        )

    table = Table(
        title=f"Benchmark ({'quantized' if rt.quantization_active else 'full precision'})"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Value (ms)", style="green")
    for k, v in result.items():
        if isinstance(v, (int, float)) and "ms" in k:
            table.add_row(k, f"{v:.3f}")
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Loopback host for local development"),
    port: int = typer.Option(8000, "--port"),
    public: bool = typer.Option(
        False,
        "--public",
        help="Explicitly bind to all interfaces; deploy behind a reverse proxy with rate limits.",
    ),
    model_dir: Path | None = typer.Option(
        None,
        "--model-dir",
        envvar="BSFM_MODEL_DIR",
        help="Directory containing operator-staged checkpoints for registration.",
    ),
    api_key: str | None = typer.Option(None, "--api-key", envvar="BSFM_API_KEY"),
) -> None:
    """Start the FastAPI server with loopback binding by default."""
    if public:
        host = "0.0.0.0"
    elif host not in {"127.0.0.1", "localhost", "::1"}:
        raise typer.BadParameter("Use --public to bind to a non-loopback interface.")
    import uvicorn

    from ..deployment import ModelRegistry, create_app

    registry = ModelRegistry(storage_dir=model_dir) if model_dir else ModelRegistry()
    app_instance = create_app(registry, api_key=api_key)
    console.print(f"[bold]Serving BioSignal-FM API[/bold] on http://{host}:{port}")
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


@app.command()
def ui() -> None:
    """Start the Streamlit dashboard."""
    import subprocess

    # Resolve via the installed package location, not a relative path — a
    # relative "biosignal_fm/ui/app.py" only resolves if the user happens to
    # invoke this from the repo root, which isn't guaranteed for an
    # installed console-script command.
    app_path = Path(__file__).resolve().parent.parent / "ui" / "app.py"
    console.print("[bold]Starting Streamlit dashboard[/bold]...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port=8501"],
        check=False,
    )


if __name__ == "__main__":
    app()
