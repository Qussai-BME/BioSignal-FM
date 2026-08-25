"""FastAPI serving with API-key auth and UUID-based model registry.

Security:

- All mutating endpoints (POST /predict, POST /models/register, DELETE)
  require an API key in the ``X-API-Key`` header.
- Read endpoints (GET /health, GET /models/{id}/info) are unauthenticated.
- The model registry uses UUIDs and rejects client-supplied file paths
  (no ``pickle.load`` from user-supplied paths).
- API keys are compared with ``secrets.compare_digest`` (constant-time).
"""

from __future__ import annotations

import logging
import os
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Modality
from ..models import FoundationModel
from ..reproducibility import compute_sha256
from .realtime import RealtimeInference

__all__ = ["ModelRegistry", "create_app"]

logger = logging.getLogger(__name__)


@dataclass
class _RegistryEntry:
    """Internal registry entry."""

    model_id: str
    name: str
    modality: Modality
    inference: RealtimeInference
    n_channels: int
    signal_length: int
    sha256: str
    created_at: str


class ModelRegistry:
    """UUID-indexed model registry.

    Models are registered by uploading a checkpoint file via
    :meth:`register_from_path`. The registry computes SHA-256 of the
    checkpoint, assigns a UUID, and stores the loaded model in memory.

    Client-supplied file paths are NEVER used at inference time — the
    registry only loads files that have been explicitly registered.

    Examples
    --------
    >>> from biosignal_fm.deployment import ModelRegistry
    >>> reg = ModelRegistry()
    >>> # entry_id = reg.register_from_path("my_model.pt", name="exp1",
    >>> #                                   modality="emg", n_channels=16,
    >>> #                                   signal_length=400)
    >>> # info = reg.get_info(entry_id)
    """

    def __init__(self, storage_dir: Path | str | None = None) -> None:
        self.storage_dir = (
            Path(storage_dir).expanduser().resolve()
            if storage_dir
            else Path.home() / ".cache" / "biosignal_fm" / "registry"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, _RegistryEntry] = {}

    def register_from_path(
        self,
        path: Path | str,
        name: str,
        modality: Modality | str,
        n_channels: int,
        signal_length: int,
        quantize: bool = True,
    ) -> str:
        """Register a model from a checkpoint file.

        Parameters
        ----------
        path : Path or str
            Path to the model checkpoint (must be a BioSignal-FM FoundationModel
            checkpoint saved via ``FoundationModel.save()``).
        name : str
            Human-readable model name.
        modality : Modality or str
        n_channels : int
        signal_length : int
        quantize : bool

        Returns
        -------
        str
            The assigned model UUID.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        """
        requested_path = Path(path).expanduser()
        if requested_path.is_absolute() or ".." in requested_path.parts:
            raise ValueError("Checkpoint paths must be relative to the configured model directory.")

        path = (self.storage_dir / requested_path).resolve()
        try:
            path.relative_to(self.storage_dir)
        except ValueError as exc:
            raise ValueError("Checkpoint path escapes the configured model directory.") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {requested_path}")

        sha = compute_sha256(path)
        # FoundationModel.load uses torch.load(weights_only=True), which means
        # the checkpoint cannot execute arbitrary Python. This is the safety
        # net even if a malicious file is placed on disk.
        model = FoundationModel.load(path)
        rt = RealtimeInference(model=model, modality=modality, quantize=quantize)

        model_id = str(uuid.uuid4())
        entry = _RegistryEntry(
            model_id=model_id,
            name=name,
            modality=Modality.from_str(modality) if isinstance(modality, str) else modality,
            inference=rt,
            n_channels=n_channels,
            signal_length=signal_length,
            sha256=sha,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._entries[model_id] = entry
        return model_id

    def get(self, model_id: str) -> _RegistryEntry:
        """Get a registry entry by UUID.

        Raises
        ------
        KeyError
            If the model_id is not registered.
        """
        if model_id not in self._entries:
            raise KeyError(f"Model {model_id} not registered")
        return self._entries[model_id]

    def get_info(self, model_id: str) -> dict:
        """Get metadata about a registered model (no inference object)."""
        e = self.get(model_id)
        return {
            "model_id": e.model_id,
            "name": e.name,
            "modality": e.modality.value,
            "n_channels": e.n_channels,
            "signal_length": e.signal_length,
            "sha256": e.sha256,
            "created_at": e.created_at,
        }

    def list_models(self) -> list[dict]:
        """List all registered models."""
        return [self.get_info(mid) for mid in self._entries]

    def remove(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        if model_id in self._entries:
            del self._entries[model_id]
            return True
        return False


# ---- Pydantic request/response models ----


class PredictRequest(BaseModel):
    """Request body for /predict."""

    signal: list[list[float]] = Field(..., description="2D array (channels, samples)")
    model_id: str = Field(..., min_length=1, description="Registered model UUID")


class PredictResponse(BaseModel):
    """Response body for /predict."""

    cls_token: list[list[float]]
    model_id: str
    latency_ms: float
    n_channels: int
    signal_length: int


class RegisterRequest(BaseModel):
    """Request body for /models/register (relative checkpoint path)."""

    path: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Path relative to the configured model directory",
    )
    name: str = Field(..., min_length=1, max_length=128)
    modality: str = Field(..., min_length=1, max_length=32)
    n_channels: Annotated[int, Field(ge=1)]
    signal_length: Annotated[int, Field(ge=1)]
    quantize: bool = True


class RegisterResponse(BaseModel):
    model_id: str
    name: str
    sha256: str


class HealthResponse(BaseModel):
    status: str
    version: str
    n_models: int


def create_app(
    registry: ModelRegistry,
    api_key: str | None = None,
    cors_origins: tuple[str, ...] = ("http://localhost:8501",),
) -> FastAPI:
    """Create the FastAPI application.

    Parameters
    ----------
    registry : ModelRegistry
        Pre-populated model registry.
    api_key : str, optional
        API key for mutating endpoints. If None, reads from ``BSFM_API_KEY``
        environment variable. If neither is set, mutating endpoints are
        disabled (return 503).
    cors_origins : tuple of str
        Allowed CORS origins.

    Returns
    -------
    FastAPI
        Configured application instance.
    """
    if api_key is None:
        api_key = os.environ.get("BSFM_API_KEY")
    app = FastAPI(
        title="BioSignal-FM API",
        description="Research-only REST and WebSocket API for BioSignal-FM signal representations.",
        version=__version__,
        contact={"name": "Qussai Adlbi", "email": "qussai.adlbi@proton.me"},
        license_info={"name": "Apache 2.0", "url": "http://www.apache.org/licenses/LICENSE-2.0"},
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    def verify_api_key(request: Request) -> None:
        """Dependency: verify API key on mutating endpoints."""
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API key not configured on the server. Set BSFM_API_KEY env var.",
            )
        provided = request.headers.get("X-API-Key", "")
        if not secrets.compare_digest(provided, api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key. Provide X-API-Key header.",
            )

    def validate_signal(raw_signal: object, entry: _RegistryEntry) -> np.ndarray:
        """Convert and validate a finite 2D signal against registered model dimensions."""
        try:
            signal = np.asarray(raw_signal, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("signal must be a numeric 2D array (channels, samples)") from exc
        if signal.ndim != 2:
            raise ValueError("signal must be 2D (channels, samples)")
        if signal.shape != (entry.n_channels, entry.signal_length):
            raise ValueError(
                "Expected signal shape "
                f"({entry.n_channels}, {entry.signal_length}), got {tuple(signal.shape)}"
            )
        if not np.isfinite(signal).all():
            raise ValueError("signal must contain only finite numeric values")
        return signal

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Health check (no auth)."""
        return HealthResponse(status="ok", version=__version__, n_models=len(registry._entries))

    @app.get("/models", tags=["models"])
    def list_models() -> list[dict]:
        """List registered models (no auth — read-only metadata)."""
        return registry.list_models()

    @app.get("/models/{model_id}/info", tags=["models"])
    def model_info(model_id: str) -> dict:
        """Get metadata about a specific model (no auth)."""
        try:
            return registry.get_info(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found") from None

    @app.post(
        "/predict",
        response_model=PredictResponse,
        tags=["inference"],
        dependencies=[Depends(verify_api_key)],
    )
    def predict(req: PredictRequest) -> PredictResponse:
        """Run inference (requires API key)."""
        try:
            entry = registry.get(req.model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Model {req.model_id} not found") from None

        try:
            signal = validate_signal(req.signal, entry)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

        import time

        t0 = time.perf_counter()
        cls = entry.inference.predict(signal)
        latency = (time.perf_counter() - t0) * 1000.0

        return PredictResponse(
            cls_token=cls.tolist(),
            model_id=req.model_id,
            latency_ms=latency,
            n_channels=signal.shape[0],
            signal_length=signal.shape[1],
        )

    @app.post(
        "/models/register",
        response_model=RegisterResponse,
        tags=["models"],
        dependencies=[Depends(verify_api_key)],
    )
    def register_model(req: RegisterRequest) -> RegisterResponse:
        """Register a checkpoint staged in the configured model directory.

        The path is relative to ``ModelRegistry.storage_dir``. The endpoint
        never accepts an absolute path or a path that escapes that directory.
        Checkpoints are loaded with ``torch.load(weights_only=True)``; operators
        must still stage only trusted, expected artifacts in the model directory.
        """
        try:
            model_id = registry.register_from_path(
                path=req.path,
                name=req.name,
                modality=req.modality,
                n_channels=req.n_channels,
                signal_length=req.signal_length,
                quantize=req.quantize,
            )
            info = registry.get_info(model_id)
            return RegisterResponse(model_id=model_id, name=info["name"], sha256=info["sha256"])
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except (ValueError, RuntimeError, KeyError) as e:
            # Specific, expected failures → 422.
            raise HTTPException(status_code=422, detail=f"Registration failed: {e}") from e

    @app.delete(
        "/models/{model_id}",
        tags=["models"],
        dependencies=[Depends(verify_api_key)],
    )
    def remove_model(model_id: str) -> dict:
        """Remove a model from the registry (requires API key)."""
        ok = registry.remove(model_id)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return {"removed": model_id}

    @app.websocket("/ws/predict/{model_id}")
    async def predict_ws(websocket: WebSocket, model_id: str) -> None:
        """Streaming inference over a persistent WebSocket connection.

        Protocol
        --------
        Client sends one JSON text message per signal window::

            {"signal": [[...], [...], ...]}   # shape (n_channels, n_samples)

        Server replies with one JSON text message per prediction::

            {"cls_token": [[...]], "latency_ms": 1.23}

        or, for a bad window (connection stays open so the client can just
        send the next one)::

            {"error": "..."}

        Authentication
        --------------
        The first message must be ``{"api_key": "..."}``. Credentials are
        intentionally not accepted in query parameters, which are commonly
        retained by access logs and intermediary telemetry. After successful
        authentication, each subsequent message contains one signal window.
        """
        await websocket.accept()
        if api_key is None:
            await websocket.close(code=4503, reason="API key not configured on the server")
            return
        try:
            auth_payload = await websocket.receive_json()
            provided = auth_payload.get("api_key", "") if isinstance(auth_payload, dict) else ""
        except (TypeError, ValueError):
            provided = ""
        if not isinstance(provided, str) or not secrets.compare_digest(provided, api_key):
            await websocket.close(code=4401, reason="Invalid or missing API key")
            return

        try:
            entry = registry.get(model_id)
        except KeyError:
            await websocket.close(code=4404, reason=f"Model {model_id} not found")
            return

        import time

        try:
            while True:
                payload = await websocket.receive_json()
                try:
                    raw_signal = payload["signal"] if isinstance(payload, dict) else None
                    signal = validate_signal(raw_signal, entry)
                except (KeyError, ValueError, TypeError) as e:
                    await websocket.send_json({"error": str(e)})
                    continue

                t0 = time.perf_counter()
                cls = entry.inference.predict(signal)
                latency = (time.perf_counter() - t0) * 1000.0
                await websocket.send_json({"cls_token": cls.tolist(), "latency_ms": latency})
        except WebSocketDisconnect:
            pass

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Log an unexpected failure without disclosing implementation detail to clients."""
        logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app
