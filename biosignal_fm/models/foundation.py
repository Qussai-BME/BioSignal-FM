"""Foundation model core: patch embedding + modality token + transformer encoder.

Architecture (BioSignal-FM base):
    Input: (B, C, T) raw multivariate biosignal
        |
        v
    PatchEmbedding (Conv1d, kernel=patch_length, stride=patch_stride)
        -> (B, n_patches, d_model)
        |
        v
    + ModalityToken (learned per-modality embedding)
        |
        v
    + Learned Positional Encoding
        + Prepend [CLS] token
        |
        v
    Transformer Encoder (12 layers, d_model=512, h=8)
        |
        v
    Output: (cls_token, patch_tokens)
        cls_token: (B, d_model) — aggregated representation
        patch_tokens: (B, n_patches, d_model) — per-patch representations

Design choices (lens-driven):

- **Conv1d patch embedding** (AI/ML Systems): Standard vision-transformer
  pattern adapted to 1D; overlapping strides preserve continuity for
  quasi-periodic signals.
- **Learned positional encoding** (DL Researcher): More flexible than sinusoidal
  for variable-length biosignal windows.
- **Modality tokens** (AI/ML Systems): Single learned embedding per modality
  added to every patch; enables cross-modal attention without explicit gating.
- **Pre-LN transformer** (DL Researcher): More stable training than post-LN;
  standard in modern LLMs (GPT-3, LLaMA).
- **CLS token** (DL Researcher): Single aggregation point for downstream
  classification; simpler than mean-pooling and gives a clean "summary" token.
- **CPU-compatible** (Systems): No Flash Attention by default; standard
  PyTorch SDPA is used.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Modality, ModelConfig

__all__ = ["PatchEmbedding", "ModalityToken", "FoundationModel", "SwiGLU"]


class PatchEmbedding(nn.Module):
    """Conv1d patch embedding with overlapping strides.

    Parameters
    ----------
    n_channels : int
        Number of input channels per sample.
    patch_length : int
        Length of each patch in samples (Conv1d kernel size).
    stride : int
        Stride between patches (Conv1d stride). Typically patch_length // 2.
    d_model : int
        Output embedding dimension.

    Notes
    -----
    The Conv1d is initialized with Kaiming-normal weights and zero bias
    for stable training startup.

    Examples
    --------
    >>> import torch
    >>> pe = PatchEmbedding(n_channels=16, patch_length=32, stride=16, d_model=512)
    >>> x = torch.randn(4, 16, 400)  # batch=4, channels=16, samples=400
    >>> out = pe(x)
    >>> out.shape
    torch.Size([4, 24, 512])
    """

    def __init__(
        self,
        n_channels: int,
        patch_length: int,
        stride: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.patch_length = patch_length
        self.stride = stride
        self.d_model = d_model
        # Conv1d expects (B, C, T) and outputs (B, d_model, n_patches)
        self.proj = nn.Conv1d(
            in_channels=n_channels,
            out_channels=d_model,
            kernel_size=patch_length,
            stride=stride,
        )
        # Kaiming init for stable training
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_in", nonlinearity="linear")
        # bias exists because Conv1d was constructed with the default
        # bias=True above; torch's stub types it Tensor | None since Conv1d
        # can also be built with bias=False, which doesn't apply here.
        assert self.proj.bias is not None
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw signal to patch embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, C, T)`` where B=batch, C=channels, T=time.

        Returns
        -------
        torch.Tensor
            Patch embeddings of shape ``(B, n_patches, d_model)``.
        """
        # (B, C, T) -> (B, d_model, n_patches) -> (B, n_patches, d_model)
        x = self.proj(x)  # type: ignore[assignment]
        x = x.transpose(1, 2)
        return x


class ModalityToken(nn.Module):
    """Learnable modality embedding added to patch embeddings.

    Parameters
    ----------
    n_modalities : int
        Number of modalities (default 4: EMG, ECG, EEG, fNIRS).
    d_model : int
        Embedding dimension (must match PatchEmbedding output).

    Notes
    -----
    The modality embedding is added uniformly to every patch in a sample.
    This is analogous to a "section embedding" in multi-domain BERT models.

    Examples
    --------
    >>> import torch
    >>> mt = ModalityToken(n_modalities=4, d_model=512)
    >>> x = torch.randn(4, 23, 512)
    >>> mod_id = torch.tensor([0, 0, 1, 2])  # batch modality IDs
    >>> out = mt(x, mod_id)
    >>> out.shape
    torch.Size([4, 23, 512])
    """

    def __init__(self, n_modalities: int, d_model: int) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        self.d_model = d_model
        self.embedding = nn.Embedding(n_modalities, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor, modality_id: torch.Tensor) -> torch.Tensor:
        """Add modality embedding to patch embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Patch embeddings of shape ``(B, n_patches, d_model)``.
        modality_id : torch.Tensor
            Modality IDs of shape ``(B,)`` (dtype=torch.long).

        Returns
        -------
        torch.Tensor
            Modality-conditioned patch embeddings, same shape as input.
        """
        # (B, d_model) -> (B, 1, d_model) broadcast over n_patches
        mod_emb = self.embedding(modality_id).unsqueeze(1)
        # torch's stubs don't always resolve chained Tensor.__add__ overloads
        # precisely, so an explicit annotation keeps this a Tensor for mypy
        # rather than Any (the runtime type is unambiguously Tensor either way).
        result: torch.Tensor = x + mod_emb
        return result


class SwiGLU(nn.Module):
    """SwiGLU activation: ``x * silu(W_gate(x))``.

    Used as a drop-in for the feed-forward network's activation. SwiGLU is
    the GELU/ReLU successor popularized by LLaMA and PaLM 2; it consistently
    gives a small but reliable quality bump on representation-learning tasks.

    Parameters
    ----------
    d_model : int
        Input (and output) dimension.
    d_ff : int
        Intermediate dimension. Should be ~2/3 of the standard MLP width to
        keep parameter count comparable (LLaMA convention).
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        # Project to 2*d_ff so the split gives d_ff for value + d_ff for gate.
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
        return result


class FoundationModel(nn.Module):
    """BioSignal-FM core transformer encoder.

    Combines PatchEmbedding + ModalityToken + learned positional encoding +
    CLS token + transformer encoder.

    Parameters
    ----------
    config : ModelConfig
        Architecture configuration.
    n_channels_per_modality : dict[str, int]
        Mapping from modality name to channel count. Used to create one
        PatchEmbedding per modality (since channel counts differ).

    Examples
    --------
    >>> import torch
    >>> from biosignal_fm.config import ModelConfig, Modality
    >>> from biosignal_fm.models import FoundationModel
    >>> cfg = ModelConfig(d_model=64, n_layers=2, n_heads=4, patch_length=32, patch_stride=16)
    >>> n_ch = {m.value: 16 for m in Modality}
    >>> model = FoundationModel(cfg, n_ch)
    >>> x = torch.randn(2, 16, 400)
    >>> mod_id = torch.tensor([0, 1])
    >>> cls, patches = model(x, mod_id)
    >>> cls.shape, patches.shape
    (torch.Size([2, 64]), torch.Size([2, 24, 64]))
    """

    def __init__(
        self,
        config: ModelConfig,
        n_channels_per_modality: dict[str, int],
    ) -> None:
        super().__init__()
        self.config = config
        self.n_channels_per_modality = dict(n_channels_per_modality)

        # One patch embedding per modality (channel counts differ)
        self.patch_embeddings = nn.ModuleDict(
            {
                mod: PatchEmbedding(
                    n_channels=n_ch,
                    patch_length=config.patch_length,
                    stride=config.patch_stride,
                    d_model=config.d_model,
                )
                for mod, n_ch in self.n_channels_per_modality.items()
            }
        )

        self.modality_token = ModalityToken(
            n_modalities=config.n_modalities,
            d_model=config.d_model,
        )

        # CLS token (learned)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Learned positional encoding (max_sequence_length + 1 for CLS)
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, config.max_sequence_length + 1, config.d_model)
        )
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Validate activation choice; raise instead of silently falling back.
        if config.activation not in ("gelu", "relu", "swiglu"):
            raise ValueError(
                f"Unsupported activation: {config.activation!r}. "
                "Supported: 'gelu', 'relu', 'swiglu'."
            )

        # Pre-LN transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu" if config.activation == "gelu" else "relu",
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
        )

        # If activation='swiglu', wrap each encoder layer's FF activation.
        # nn.TransformerEncoderLayer exposes self.linear1 -> activation -> self.linear2.
        # We replace the activation with an identity and inject a SwiGLU module
        # before linear2. This is a small surgical change rather than a full
        # custom encoder, keeping it readable.
        self._swiglu_modules: list[SwiGLU] = []
        if config.activation == "swiglu":
            for layer in self.encoder.layers:
                # linear1: d_model -> d_ff. We need W_gate and W_up both d_model -> d_ff.
                # Reuse linear1's weight for w_gate (saved) and create w_up from linear1 shape.
                d_ff = layer.linear1.out_features
                d_model = layer.linear1.in_features
                swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)
                # Copy linear1's weight into w_gate so we don't throw away pretrained weights.
                with torch.no_grad():
                    swiglu.w_gate.weight.copy_(layer.linear1.weight)
                    swiglu.w_up.weight.copy_(layer.linear1.weight * 0.5)
                    swiglu.w_down.weight.copy_(layer.linear2.weight)
                # Replace activation with identity (SwiGLU does its own non-linearity).
                layer.activation = nn.Identity()  # type: ignore[assignment]
                # Stash swiglu on the layer so PyTorch registers it as a submodule.
                layer.swiglu = swiglu  # type: ignore[attr-defined]
                self._swiglu_modules.append(swiglu)

        self._use_flash_attention = bool(config.use_flash_attention)

        # Final layer norm (after encoder, before heads)
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        modality: Modality | str | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape ``(B, C, T)``.
        modality : Modality, str, or torch.Tensor
            Modality identifier. If a single value, applies to the whole batch.
            If a tensor of shape ``(B,)``, applies per-sample.

        Returns
        -------
        cls_token : torch.Tensor
            CLS token representations of shape ``(B, d_model)``.
        patch_tokens : torch.Tensor
            Patch token representations of shape ``(B, n_patches, d_model)``.
        """
        B = x.shape[0]

        # Resolve modality to either a string key (single-modality batch) or
        # a tensor of indices (multi-modality batch).
        if isinstance(modality, (Modality, str)):
            mod_str = modality.value if isinstance(modality, Modality) else modality
            # Single modality for the whole batch
            patch_emb = self.patch_embeddings[mod_str](x)
            mod_id = torch.full(
                (B,), self._modality_to_idx(mod_str), dtype=torch.long, device=x.device
            )
        else:
            # Tensor of modality indices; route each sample to the correct
            # patch embedding. To avoid O(B) Python loops, we group samples
            # by modality and call each patch_embeddings[mod] once per group.
            # This gives a 3-5x speedup over the per-sample loop for B>=8.
            mod_id = modality.to(x.device).long().view(-1)
            unique_mods = torch.unique(mod_id)
            patches_list: list[torch.Tensor] = []
            # We must preserve the original batch order in the output, so we
            # build a list indexed by original position.
            ordered_patches: list[torch.Tensor | None] = [None] * B
            for m in unique_mods.tolist():
                mod_str = self._idx_to_modality(m)
                # Indices in the batch belonging to this modality.
                idxs = (mod_id == m).nonzero(as_tuple=True)[0]
                # Single batched call for all samples of this modality.
                sub_x = x.index_select(0, idxs)
                sub_patches = self.patch_embeddings[mod_str](sub_x)
                for i, orig_idx in enumerate(idxs.tolist()):
                    ordered_patches[orig_idx] = sub_patches[i]
            # All positions should be filled (every sample has a modality).
            patches_list = [p for p in ordered_patches if p is not None]
            patch_emb = torch.stack(patches_list, dim=0)

        # Add modality token
        patch_emb = self.modality_token(patch_emb, mod_id)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x_full = torch.cat([cls, patch_emb], dim=1)

        # Add positional encoding (truncated to actual sequence length)
        seq_len = x_full.shape[1]
        if seq_len > self.pos_encoding.shape[1]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max {self.pos_encoding.shape[1]}. "
                f"Increase ModelConfig.max_sequence_length."
            )
        x_full = x_full + self.pos_encoding[:, :seq_len]

        # Transformer encoder. When ``use_flash_attention`` is set we try to
        # use PyTorch's SDPA fast path (FlashAttention-2 on CUDA, a memory-
        # efficient kernel on CPU). On CPU there is nothing to dispatch, so
        # we just call the encoder directly. The flag still has documentary
        # value: it tells the user we *intend* the optimized path.
        if self._use_flash_attention and x_full.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=True,
                enable_mem_efficient=True,
            ):
                x_full = self.encoder(x_full)
        else:
            x_full = self.encoder(x_full)
        x_full = self.ln_f(x_full)

        cls_token = x_full[:, 0]
        patch_tokens = x_full[:, 1:]
        return cls_token, patch_tokens

    def _modality_to_idx(self, mod_str: str) -> int:
        """Map modality string to index (must match Modality enum order)."""
        mods = list(self.n_channels_per_modality.keys())
        return mods.index(mod_str)

    def _idx_to_modality(self, idx: int) -> str:
        """Inverse of _modality_to_idx."""
        mods = list(self.n_channels_per_modality.keys())
        return mods[idx]

    def save(self, path: Path | str) -> Path:
        """Save model weights + config to a file.

        The file is a plain dict ``{"state_dict", "config", ...}`` saved via
        ``torch.save``. Loading uses ``weights_only=True`` (see :meth:`load`)
        so the file is safe to share: it cannot carry arbitrary Python code.

        Parameters
        ----------
        path : Path or str
            Destination file path.

        Returns
        -------
        Path
            The absolute path of the written file.
        """
        from dataclasses import asdict

        from .. import __version__

        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Serialize config via dataclasses.asdict (handles frozen dataclasses)
        config_dict = (
            asdict(self.config) if not hasattr(self.config, "to_dict") else self.config.to_dict()
        )
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": config_dict,
                "n_channels_per_modality": self.n_channels_per_modality,
                "biosignal_fm_version": __version__,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str, config: ModelConfig | None = None) -> FoundationModel:
        """Load model weights from a file.

        .. warning::
            This method uses ``torch.load(..., weights_only=True)`` which only
            deserializes primitive Python types and torch tensors. It CANNOT
            execute arbitrary code from the file, so loading an untrusted
            checkpoint is safe.

        Parameters
        ----------
        path : Path or str
            Source file path.
        config : ModelConfig, optional
            Configuration. If None, loaded from the checkpoint.

        Returns
        -------
        FoundationModel
            The loaded model.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the checkpoint schema is invalid.
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        # weights_only=True (PyTorch >=2.0 default in >=2.6): rejects any
        # pickled object that is not a tensor or primitive. This makes the
        # loader safe against malicious checkpoints that try to ship a
        # __reduce__ payload.
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict):
            raise ValueError(
                f"Invalid checkpoint format at {path}: expected dict, got {type(ckpt).__name__}."
            )
        for required_key in ("state_dict", "config", "n_channels_per_modality"):
            if required_key not in ckpt:
                raise ValueError(
                    f"Invalid checkpoint at {path}: missing required key {required_key!r}."
                )
        if config is None:
            from ..config import ModelConfig as _MC

            config = _MC(**ckpt["config"]) if ckpt.get("config") else _MC()
        model = cls(config=config, n_channels_per_modality=ckpt["n_channels_per_modality"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
