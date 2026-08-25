"""SSL pretraining loop for BioSignal-FM.

Implements:

- Combined MSE + contrastive SSL loss
- Span masking (per-batch, reproducible)
- Mixed precision (AMP) — disabled on CPU-only
- Gradient clipping
- Learning rate warmup + cosine decay
- EMA of weights
- MLflow / local JSON tracking
- RunManifest generation
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..config import ModelConfig, TrainingConfig
from ..models import (
    ContrastiveHead,
    FoundationModel,
    SpanMaskedReconstructionHead,
    span_mask,
)
from ..reproducibility import RunManifest, set_global_seed

__all__ = ["SSLPretrainer"]


@dataclass
class SSLPretrainer:
    """Orchestrates self-supervised pretraining.

    Parameters
    ----------
    model : FoundationModel
        The foundation model to pretrain.
    ssl_head : SpanMaskedReconstructionHead
        Masked reconstruction head.
    contrastive_head : ContrastiveHead
        Contrastive head.
    config : TrainingConfig
        Training hyperparameters.
    model_config : ModelConfig
        Model architecture (for EMA setup).
    tracker : object, optional
        Tracker object exposing ``log_params``, ``log_metrics``, ``log_artifact``.
        If None, no tracking.
    device : str, optional
        "cpu" or "cuda". Default "cpu".

    Examples
    --------
    >>> import torch
    >>> from biosignal_fm.config import ModelConfig, TrainingConfig, Modality
    >>> from biosignal_fm.models import FoundationModel, SpanMaskedReconstructionHead, ContrastiveHead
    >>> from biosignal_fm.training import SSLPretrainer
    >>> cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, patch_length=16, patch_stride=8)
    >>> n_ch = {m.value: 4 for m in Modality}
    >>> model = FoundationModel(cfg, n_ch)
    >>> ssl = SpanMaskedReconstructionHead(d_model=32, patch_length=16, n_channels=4)
    >>> ctr = ContrastiveHead(d_model=32, projection_dim=32)
    >>> trainer = SSLPretrainer(model, ssl, ctr, TrainingConfig(max_steps=2, warmup_steps=0), cfg)
    """

    model: FoundationModel
    ssl_head: SpanMaskedReconstructionHead
    contrastive_head: ContrastiveHead
    config: TrainingConfig
    model_config: ModelConfig
    tracker: Any | None = None
    device: str = "cpu"

    # Internal state
    _optimizer: torch.optim.Optimizer | None = field(default=None, init=False, repr=False)
    _scheduler: Any | None = field(default=None, init=False, repr=False)
    _scaler: torch.amp.GradScaler | None = field(default=None, init=False, repr=False)
    _ema_model: FoundationModel | None = field(default=None, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        set_global_seed(self.config.seed)
        self.model.to(self.device)
        self.ssl_head.to(self.device)
        self.contrastive_head.to(self.device)

        # Optimizer
        params = (
            list(self.model.parameters())
            + list(self.ssl_head.parameters())
            + list(self.contrastive_head.parameters())
        )
        if self.config.optimizer == "adamw":
            self._optimizer = torch.optim.AdamW(
                params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            self._optimizer = torch.optim.Adam(
                params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
        else:
            self._optimizer = torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )

        # LR scheduler
        self._scheduler = self._build_scheduler()

        # AMP scaler (CPU: disabled automatically)
        # Use the new torch.amp API; autocast is enabled per-step.
        self._use_amp = self.config.use_amp and self.device != "cpu" and torch.cuda.is_available()
        if self._use_amp:
            self._scaler = torch.amp.GradScaler("cuda")
        else:
            self._scaler = None

        # EMA
        if self.config.ema_use:
            self._ema_model = self._build_ema()

    def _build_scheduler(self) -> Any:
        """Build LR scheduler."""
        # __post_init__ always assigns _optimizer (all three branches of the
        # if/elif/else above set it) before calling this method, so it's
        # never actually None here — asserting narrows the type for mypy
        # and protects against a future refactor breaking that ordering.
        assert self._optimizer is not None
        total = max(self.config.max_steps, 1)
        warmup = min(self.config.warmup_steps, total)
        if self.config.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.LambdaLR(
                self._optimizer,
                lr_lambda=lambda step: self._cosine_with_warmup(step, warmup, total),
            )
        elif self.config.lr_scheduler == "linear":
            return torch.optim.lr_scheduler.LambdaLR(
                self._optimizer,
                lr_lambda=lambda step: self._linear_with_warmup(step, warmup, total),
            )
        return torch.optim.lr_scheduler.LambdaLR(
            self._optimizer,
            lr_lambda=lambda step: 1.0,
        )

    @staticmethod
    def _cosine_with_warmup(step: int, warmup: int, total: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def _linear_with_warmup(step: int, warmup: int, total: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(0.0, 1.0 - progress)

    def _build_ema(self) -> FoundationModel:
        """Build an EMA copy of the model."""
        import copy

        ema = copy.deepcopy(self.model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    def _update_ema(self) -> None:
        """Update EMA parameters."""
        if self._ema_model is None:
            return
        decay = self.config.ema_decay
        with torch.no_grad():
            for ema_p, p in zip(self._ema_model.parameters(), self.model.parameters(), strict=True):
                ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, float]:
        """One training step.

        Parameters
        ----------
        batch : tuple
            (signals, modality_ids, targets) where:
            - signals: (B, C, T) raw signal
            - modality_ids: (B,) modality indices
            - targets: (B, n_patches, C, patch_length) for reconstruction

        Returns
        -------
        dict
            {"loss", "mse", "contrastive", "lr", "step"}
        """
        self.model.train()
        signals, modality_ids, targets = batch
        signals = signals.to(self.device)
        modality_ids = modality_ids.to(self.device)
        targets = targets.to(self.device)

        # __post_init__ always assigns both before train_step can be called
        # (see _build_scheduler's comment for _optimizer specifically);
        # asserting narrows the type for mypy at every use below.
        assert self._optimizer is not None
        assert self._scheduler is not None

        self._optimizer.zero_grad(set_to_none=True)

        # Forward pass — view A
        with torch.amp.autocast("cuda" if self.device != "cpu" else "cpu", enabled=self._use_amp):
            cls_a, patch_tokens_a = self.model(signals, modality_ids)

            # View B: same signals with mild noise augmentation
            noise = torch.randn_like(signals) * 0.01
            cls_b, _ = self.model(signals + noise, modality_ids)

            # Span mask
            B, n_patches, _ = patch_tokens_a.shape
            mask = span_mask(
                B,
                n_patches,
                mask_ratio=self.model_config.mask_ratio,
                mean_span_length=self.model_config.mean_mask_span_length,
            ).to(self.device)

            # Losses
            mse_loss = self.ssl_head(patch_tokens_a, mask, targets)
            ctr_loss = self.contrastive_head(cls_a, cls_b)
            loss = (
                self.model_config.reconstruction_weight * mse_loss
                + self.model_config.contrastive_weight * ctr_loss
            )

        # Backward
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            self._optimizer.step()

        self._scheduler.step()
        self._update_ema()
        self._step += 1

        return {
            "loss": float(loss.item()),
            "mse": float(mse_loss.item()),
            "contrastive": float(ctr_loss.item()),
            "lr": float(self._optimizer.param_groups[0]["lr"]),
            "step": self._step,
        }

    def train(
        self,
        dataloader: DataLoader,
        n_steps: int | None = None,
        output_dir: Path | str | None = None,
        run_name: str = "ssl_pretrain",
    ) -> dict[str, Any]:
        """Run full SSL pretraining.

        Parameters
        ----------
        dataloader : DataLoader
            Training data loader. Must yield (signals, modality_ids, targets).
        n_steps : int, optional
            Number of steps. Default ``config.max_steps``.
        output_dir : Path or str, optional
            Directory to save checkpoints and manifest. If None, no saving.
        run_name : str
            Name for the RunManifest.

        Returns
        -------
        dict
            Final metrics and run_id.
        """
        n_steps = n_steps or self.config.max_steps
        output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Log params
        if self.tracker:
            self.tracker.log_params(
                {
                    "max_steps": n_steps,
                    "batch_size": self.config.batch_size,
                    "lr": self.config.learning_rate,
                    "optimizer": self.config.optimizer,
                    "scheduler": self.config.lr_scheduler,
                    "warmup_steps": self.config.warmup_steps,
                    "weight_decay": self.config.weight_decay,
                    "gradient_clip_norm": self.config.gradient_clip_norm,
                    "use_amp": self.config.use_amp,
                    "ema_use": self.config.ema_use,
                }
            )

        manifest = RunManifest.create(
            name=run_name,
            config=self.model_config.to_dict() if hasattr(self.model_config, "to_dict") else None,
            seed=self.config.seed,
        )

        data_iter = iter(dataloader)
        last_metrics: dict[str, float] = {}
        t0 = time.time()

        for step in range(1, n_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            metrics = self.train_step(batch)
            last_metrics = metrics

            if self.tracker and step % max(1, self.config.eval_every_steps) == 0:
                self.tracker.log_metrics(metrics, step=step)

            if step % max(1, self.config.eval_every_steps) == 0:
                elapsed = time.time() - t0
                print(
                    f"[step {step}/{n_steps}] loss={metrics['loss']:.4f} "
                    f"mse={metrics['mse']:.4f} ctr={metrics['contrastive']:.4f} "
                    f"lr={metrics['lr']:.2e} ({elapsed:.1f}s)"
                )

            if output_dir and step % max(1, self.config.save_every_steps) == 0:
                ckpt_path = output_dir / f"checkpoint_step{step}.pt"
                self.model.save(ckpt_path)
                manifest.add_output(ckpt_path, alias=f"checkpoint_step{step}")

        # Final save
        if output_dir:
            final_path = output_dir / "final_model.pt"
            self.model.save(final_path)
            manifest.add_output(final_path, alias="final_model")
            for k, v in last_metrics.items():
                manifest.add_metric(k, v)
            manifest_path = output_dir / "manifest.json"
            manifest.save(manifest_path)
            if self.tracker:
                self.tracker.log_artifact(manifest_path)

        return {
            "run_id": manifest.run_id,
            "final_metrics": last_metrics,
            "manifest": manifest.to_dict(),
        }
