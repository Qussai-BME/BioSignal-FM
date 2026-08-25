"""Unit tests for biosignal_fm.deployment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from biosignal_fm.config import Modality, ModelConfig
from biosignal_fm.deployment import ModelRegistry, OnnxExporter, RealtimeInference, create_app
from biosignal_fm.models import FoundationModel
from fastapi import FastAPI


def _make_small_model(n_channels: int = 4) -> FoundationModel:
    cfg = ModelConfig(
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        patch_length=16,
        patch_stride=8,
        max_sequence_length=64,
    )
    n_ch = {m.value: n_channels for m in Modality}
    return FoundationModel(cfg, n_ch)


class TestOnnxExporter:
    def test_export_and_verify(self, tmp_path: Path) -> None:
        """End-to-end: export + REAL numerical parity verification."""
        model = _make_small_model(n_channels=4)
        exporter = OnnxExporter()
        onnx_path = exporter.export(
            model=model,
            output_path=tmp_path / "model.onnx",
            modality="emg",
            n_channels=4,
            signal_length=64,
        )
        assert onnx_path.exists()
        # SHA-256 sidecar
        sidecar = onnx_path.with_suffix(".onnx.sha256")
        assert sidecar.exists()

        # REAL numerical parity check
        ok = exporter.verify(
            pytorch_model=model,
            onnx_path=onnx_path,
            modality="emg",
            n_channels=4,
            signal_length=64,
            n_samples=20,
            atol=1e-5,
        )
        assert ok is True

    def test_verify_missing_file(self, tmp_path: Path) -> None:
        model = _make_small_model()
        exporter = OnnxExporter()
        with pytest.raises(FileNotFoundError):
            exporter.verify(model, tmp_path / "nonexistent.onnx", "emg", 4, 64)


class TestRealtimeInference:
    def test_predict_2d(self) -> None:
        model = _make_small_model(n_channels=4)
        rt = RealtimeInference(model, modality="emg", quantize=False)
        signal = np.random.randn(4, 64).astype(np.float32)
        cls = rt.predict(signal)
        assert cls.shape == (1, 32)

    def test_predict_3d_batch(self) -> None:
        model = _make_small_model(n_channels=4)
        rt = RealtimeInference(model, modality="emg", quantize=False)
        signals = np.random.randn(3, 4, 64).astype(np.float32)
        cls = rt.predict_batch(signals)
        assert cls.shape == (3, 32)

    def test_benchmark(self) -> None:
        model = _make_small_model(n_channels=4)
        rt = RealtimeInference(model, modality="emg", quantize=False)
        result = rt.benchmark(
            n_channels=4, signal_length=64, n_runs=10, warmup=2, force_single_thread=True
        )
        assert "mean_ms" in result
        assert "p50_ms" in result
        assert "p99_ms" in result
        assert result["single_thread"] is True
        assert result["n_runs"] == 10

    def test_quantized(self) -> None:
        """Quantized model should still produce valid output shape."""
        model = _make_small_model(n_channels=4)
        rt = RealtimeInference(model, modality="emg", quantize=True)
        signal = np.random.randn(4, 64).astype(np.float32)
        cls = rt.predict(signal)
        assert cls.shape == (1, 32)

    def test_quantization_actually_activates(self) -> None:
        """Regression test: quantize_dynamic used to always raise inside
        TransformerEncoderLayer's fast-path weight inspection and silently
        fall back to full precision, so quantize=True never actually did
        anything (previous test above didn't catch this — the fallback
        also produces a valid-shaped output). Assert the real state, not
        just the output shape.
        """
        model = _make_small_model(n_channels=4)
        rt = RealtimeInference(model, modality="emg", quantize=True)
        assert rt.quantization_active is True

    def test_quantization_reported_in_benchmark(self) -> None:
        model = _make_small_model(n_channels=4)
        rt = RealtimeInference(model, modality="emg", quantize=True)
        result = rt.benchmark(n_channels=4, signal_length=64, n_runs=5, warmup=2)
        assert result["quantization_requested"] is True
        assert result["quantization_active"] is True

    def test_quantize_false_does_not_report_active(self) -> None:
        model = _make_small_model(n_channels=4)
        rt = RealtimeInference(model, modality="emg", quantize=False)
        assert rt.quantization_active is False


class TestModelRegistry:
    def test_register_and_get(self, tmp_path: Path) -> None:
        # The registry only loads checkpoints staged in its configured directory.
        model = _make_small_model(n_channels=4)
        registry_dir = tmp_path / "registry"
        reg = ModelRegistry(storage_dir=registry_dir)
        ckpt = registry_dir / "m.pt"
        model.save(ckpt)

        model_id = reg.register_from_path(
            path="m.pt",
            name="test_model",
            modality="emg",
            n_channels=4,
            signal_length=64,
            quantize=False,
        )
        assert len(model_id) == 36  # UUID

        info = reg.get_info(model_id)
        assert info["name"] == "test_model"
        assert info["modality"] == "emg"
        assert info["n_channels"] == 4
        assert len(info["sha256"]) == 64

    def test_register_rejects_absolute_and_escaping_paths(self, tmp_path: Path) -> None:
        reg = ModelRegistry(storage_dir=tmp_path / "registry")
        with pytest.raises(ValueError, match="relative"):
            reg.register_from_path(tmp_path / "outside.pt", "test", "emg", 4, 64)
        with pytest.raises(ValueError, match="relative"):
            reg.register_from_path("../outside.pt", "test", "emg", 4, 64)

    def test_get_missing_raises(self) -> None:
        reg = ModelRegistry(storage_dir=Path("/tmp/test_registry_missing"))
        with pytest.raises(KeyError):
            reg.get("nonexistent-uuid")

    def test_list_and_remove(self, tmp_path: Path) -> None:
        model = _make_small_model(n_channels=4)
        registry_dir = tmp_path / "r"
        reg = ModelRegistry(storage_dir=registry_dir)
        ckpt = registry_dir / "m.pt"
        model.save(ckpt)

        mid = reg.register_from_path("m.pt", "test", "emg", 4, 64, quantize=False)
        assert len(reg.list_models()) == 1

        assert reg.remove(mid) is True
        assert len(reg.list_models()) == 0
        assert reg.remove(mid) is False


class TestFastAPIApp:
    def test_health_endpoint(self) -> None:
        from fastapi.testclient import TestClient

        reg = ModelRegistry(storage_dir=Path("/tmp/test_api_registry"))
        app = create_app(registry=reg, api_key="test-key")
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_predict_requires_api_key(self) -> None:
        from fastapi.testclient import TestClient

        reg = ModelRegistry(storage_dir=Path("/tmp/test_api_registry2"))
        app = create_app(registry=reg, api_key="secret")
        client = TestClient(app)

        # Without API key -> 401
        response = client.post("/predict", json={"signal": [[0.0, 1.0]], "model_id": "x"})
        assert response.status_code == 401

    def test_predict_with_api_key(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        # Save the model in the registry's controlled staging directory.
        model = _make_small_model(n_channels=4)
        registry_dir = tmp_path / "r"
        reg = ModelRegistry(storage_dir=registry_dir)
        ckpt = registry_dir / "m.pt"
        model.save(ckpt)

        mid = reg.register_from_path("m.pt", "test", "emg", 4, 64, quantize=False)

        app = create_app(registry=reg, api_key="secret")
        client = TestClient(app)

        signal = np.random.randn(4, 64).astype(np.float32).tolist()
        response = client.post(
            "/predict",
            json={"signal": signal, "model_id": mid},
            headers={"X-API-Key": "secret"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "cls_token" in data
        assert "latency_ms" in data
        assert data["model_id"] == mid

    def test_predict_rejects_wrong_length_and_nonfinite_values(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        model = _make_small_model(n_channels=4)
        registry_dir = tmp_path / "r"
        reg = ModelRegistry(storage_dir=registry_dir)
        ckpt = registry_dir / "m.pt"
        model.save(ckpt)
        model_id = reg.register_from_path("m.pt", "test", "emg", 4, 64, quantize=False)
        client = TestClient(create_app(registry=reg, api_key="secret"))

        wrong_length = np.zeros((4, 63), dtype=np.float32).tolist()
        response = client.post(
            "/predict",
            json={"signal": wrong_length, "model_id": model_id},
            headers={"X-API-Key": "secret"},
        )
        assert response.status_code == 422
        assert "Expected signal shape" in response.json()["detail"]

        nonfinite = np.zeros((4, 64), dtype=np.float32).tolist()
        nonfinite[0][0] = float("nan")
        response = client.post(
            "/predict",
            content=json.dumps({"signal": nonfinite, "model_id": model_id}, allow_nan=True),
            headers={"X-API-Key": "secret", "Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "finite" in response.json()["detail"]

    def test_register_endpoint_rejects_absolute_path(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        app = create_app(registry=ModelRegistry(storage_dir=tmp_path / "r"), api_key="secret")
        client = TestClient(app)
        response = client.post(
            "/models/register",
            json={
                "path": str(tmp_path / "outside.pt"),
                "name": "test",
                "modality": "emg",
                "n_channels": 4,
                "signal_length": 64,
            },
            headers={"X-API-Key": "secret"},
        )
        assert response.status_code == 422
        assert "relative" in response.json()["detail"]

    def test_no_api_key_disables_mutating(self) -> None:
        from fastapi.testclient import TestClient

        reg = ModelRegistry(storage_dir=Path("/tmp/test_api_no_key"))
        app = create_app(registry=reg, api_key=None)
        client = TestClient(app)

        response = client.post("/predict", json={"signal": [[0.0]], "model_id": "x"})
        assert response.status_code == 503  # Service Unavailable


class TestWebSocketPredict:
    """Tests for the /ws/predict/{model_id} streaming endpoint.

    This endpoint didn't exist at all until now, despite being claimed in
    the README's module table, the Deploy UI page, the FastAPI app
    description, CHANGELOG.md, and ARCHITECTURE.md. TestClient supports
    WebSocket testing natively via websocket_connect(), so this is tested
    the same way as the REST endpoints above, not with a separate live
    server + external client library.
    """

    def _make_app_with_model(self, tmp_path: Path) -> tuple[FastAPI, str]:
        model = _make_small_model(n_channels=4)
        registry_dir = tmp_path / "r"
        reg = ModelRegistry(storage_dir=registry_dir)
        ckpt = registry_dir / "m.pt"
        model.save(ckpt)
        mid = reg.register_from_path("m.pt", "test", "emg", 4, 64, quantize=False)
        app = create_app(registry=reg, api_key="secret")
        return app, mid

    def test_streaming_predictions(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        app, mid = self._make_app_with_model(tmp_path)
        client = TestClient(app)

        with client.websocket_connect(f"/ws/predict/{mid}") as ws:
            ws.send_json({"api_key": "secret"})
            for _ in range(3):
                signal = np.random.randn(4, 64).astype(np.float32).tolist()
                ws.send_json({"signal": signal})
                resp = ws.receive_json()
                assert "cls_token" in resp
                assert "latency_ms" in resp

    def test_bad_window_keeps_connection_open(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient

        app, mid = self._make_app_with_model(tmp_path)
        client = TestClient(app)

        with client.websocket_connect(f"/ws/predict/{mid}") as ws:
            ws.send_json({"api_key": "secret"})
            # Wrong channel count -> error message, connection stays open
            ws.send_json({"signal": [[1.0, 2.0], [3.0, 4.0]]})
            resp = ws.receive_json()
            assert "error" in resp

            # Same connection still works for a valid window afterward
            signal = np.random.randn(4, 64).astype(np.float32).tolist()
            ws.send_json({"signal": signal})
            resp = ws.receive_json()
            assert "cls_token" in resp

    def test_wrong_api_key_rejected(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        app, mid = self._make_app_with_model(tmp_path)
        client = TestClient(app)

        with client.websocket_connect(f"/ws/predict/{mid}") as ws:
            ws.send_json({"api_key": "WRONG"})
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()

    def test_unknown_model_rejected(self, tmp_path: Path) -> None:
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        app, _mid = self._make_app_with_model(tmp_path)
        client = TestClient(app)

        with client.websocket_connect("/ws/predict/00000000-0000-0000-0000-000000000000") as ws:
            ws.send_json({"api_key": "secret"})
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
