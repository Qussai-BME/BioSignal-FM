"""Fine-tuning loop for downstream tasks."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from ..models import FoundationModel
from ..reproducibility import RunManifest, set_global_seed

__all__ = ["FineTuner"]


@dataclass
class FineTuner:
    """Fine-tune a pretrained FoundationModel on a downstream task.

    Parameters
    ----------
    model : FoundationModel
        Pretrained model.
    task_head : nn.Module
        Task head (e.g. LinearProbe, ClassificationHead).
    strategy : str
        One of "linear" (frozen encoder), "partial" (last K layers), "full".
    config : TrainingConfig
        Training hyperparameters.
    device : str, optional
        "cpu" or "cuda". Default "cpu".

    Examples
    --------
    >>> import torch
    >>> from biosignal_fm.config import ModelConfig, TrainingConfig, Modality
    >>> from biosignal_fm.models import FoundationModel, LinearProbe
    >>> from biosignal_fm.training import FineTuner
    >>> cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
    >>> n_ch = {m.value: 4 for m in Modality}
    >>> model = FoundationModel(cfg, n_ch)
    >>> head = LinearProbe(d_model=32, n_classes=8)
    >>> ft = FineTuner(model, head, strategy="linear", config=TrainingConfig(max_steps=2))
    """

    model: FoundationModel
    task_head: nn.Module
    strategy: Literal["linear", "partial", "full"] = "linear"
    config: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cpu"
    n_unfrozen_layers: int = 6  # used only if strategy == "partial"
    # Backward-compat alias. Old name was misleading: this is the number of
    # layers to UNfreeze (train), not the number to freeze.
    n_frozen_layers: int | None = field(default=None, repr=False)

    _optimizer: torch.optim.Optimizer | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Backward-compat: if a caller passed n_frozen_layers, treat it as
        # n_unfrozen_layers and warn.
        if self.n_frozen_layers is not None:
            import warnings

            warnings.warn(
                "`n_frozen_layers` is deprecated and misleading (it actually meant "
                "the number of layers to UNfreeze). Use `n_unfrozen_layers` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.n_unfrozen_layers = self.n_frozen_layers
        set_global_seed(self.config.seed)
        self.model.to(self.device)
        self.task_head.to(self.device)
        self._apply_freeze_strategy()

        # Only train non-frozen params
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        trainable += [p for p in self.task_head.parameters() if p.requires_grad]

        if self.config.optimizer == "adamw":
            self._optimizer = torch.optim.AdamW(
                trainable, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
        else:
            self._optimizer = torch.optim.Adam(
                trainable, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )

    def _apply_freeze_strategy(self) -> None:
        """Apply freezing according to ``strategy``."""
        if self.strategy == "linear":
            for p in self.model.parameters():
                p.requires_grad_(False)
        elif self.strategy == "partial":
            # Freeze all but the last n_unfrozen_layers layers.
            all_layers = list(self.model.encoder.layers)
            n_layers = len(all_layers)
            n_to_unfreeze = min(self.n_unfrozen_layers, n_layers)
            for layer in all_layers[: n_layers - n_to_unfreeze]:
                for p in layer.parameters():
                    p.requires_grad_(False)
        # else "full": no freezing

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, float]:
        """One training step.

        Parameters
        ----------
        batch : tuple
            (signals, modality_ids, labels)
        """
        if self.strategy == "linear":
            self.model.eval()
        else:
            self.model.train()
        self.task_head.train()

        signals, modality_ids, labels = batch
        signals = signals.to(self.device)
        modality_ids = modality_ids.to(self.device)
        labels = labels.to(self.device).long()

        # _optimizer is always set in __post_init__; asserting here (rather
        # than leaving it as an unchecked Optional) turns any future
        # refactor that breaks that invariant into a clear error instead of
        # a silent AttributeError deep in the training loop.
        assert self._optimizer is not None, "FineTuner._optimizer was not initialized"
        self._optimizer.zero_grad(set_to_none=True)

        # No grad through encoder if linear
        if self.strategy == "linear":
            with torch.no_grad():
                cls_token, _ = self.model(signals, modality_ids)
        else:
            cls_token, _ = self.model(signals, modality_ids)

        logits = self.task_head(cls_token)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad]
            + [p for p in self.task_head.parameters() if p.requires_grad],
            self.config.gradient_clip_norm,
        )
        self._optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()

        return {"loss": float(loss.item()), "accuracy": float(acc)}

    @torch.no_grad()
    def evaluate(
        self, dataloader: DataLoader | Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> dict[str, float]:
        """Evaluate on a dataloader (or any iterable of (signals, modality_ids, labels) batches).

        Returns
        -------
        dict
            {"loss", "accuracy"} averaged across the dataloader.
        """
        self.model.eval()
        self.task_head.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for batch in dataloader:
            signals, modality_ids, labels = batch
            signals = signals.to(self.device)
            modality_ids = modality_ids.to(self.device)
            labels = labels.to(self.device).long()

            cls_token, _ = self.model(signals, modality_ids)
            logits = self.task_head(cls_token)
            loss = F.cross_entropy(logits, labels, reduction="sum")
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).sum().float()

            total_loss += float(loss.item())
            total_acc += float(acc.item())
            n_batches += labels.shape[0]

        n_batches = max(1, n_batches)
        return {
            "loss": total_loss / n_batches,
            "accuracy": total_acc / n_batches,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        n_steps: int | None = None,
        output_dir: Path | str | None = None,
        run_name: str = "finetune",
    ) -> dict[str, Any]:
        """Run fine-tuning.

        Returns
        -------
        dict
            {"run_id", "final_train_metrics", "final_val_metrics", "manifest"}
        """
        n_steps = n_steps or self.config.max_steps
        output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        manifest = RunManifest.create(name=run_name, seed=self.config.seed)

        data_iter = iter(train_loader)
        last_train: dict[str, float] = {}

        for step in range(1, n_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            metrics = self.train_step(batch)
            last_train = metrics

            if step % max(1, self.config.eval_every_steps) == 0:
                msg = f"[ft step {step}/{n_steps}] loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}"
                if val_loader is not None:
                    val_metrics = self.evaluate(val_loader)
                    msg += (
                        f" val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
                    )
                print(msg)

        final_val = self.evaluate(val_loader) if val_loader else {}

        for k, v in last_train.items():
            manifest.add_metric(f"train_{k}", v)
        for k, v in final_val.items():
            manifest.add_metric(f"val_{k}", v)

        if output_dir:
            ckpt_path = output_dir / "finetuned.pt"
            self.model.save(ckpt_path)
            manifest.add_output(ckpt_path)
            manifest_path = output_dir / "manifest.json"
            manifest.save(manifest_path)

        return {
            "run_id": manifest.run_id,
            "final_train_metrics": last_train,
            "final_val_metrics": final_val,
            "manifest": manifest.to_dict(),
        }
