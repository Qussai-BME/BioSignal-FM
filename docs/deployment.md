# Deployment Guide

## Scope

BioSignal-FM deployment features are intended for **research workflows**. They do not establish clinical validity, a medical intended use, a latency guarantee, or a secure public deployment by themselves. Treat the API, ONNX export, and Streamlit interface as optional application layers around the research core.

## Install the capability you need

A pure model/ONNX workflow does not need UI or API packages:

```bash
python -m pip install 'biosignal-fm[ml,deployment]'
```

A local API service requires the API extra as well:

```bash
python -m pip install 'biosignal-fm[ml,deployment,api]'
```

The Streamlit dashboard is separate:

```bash
python -m pip install 'biosignal-fm[ui]'
```

## ONNX export and numerical parity

```python
from biosignal_fm.deployment import OnnxExporter
from biosignal_fm.models import FoundationModel

model = FoundationModel.load("checkpoint.pt")
exporter = OnnxExporter()
onnx_path = exporter.export(
    model,
    "model.onnx",
    modality="emg",
    n_channels=16,
    signal_length=400,
)
assert exporter.verify(model, onnx_path, "emg", 16, 400)
```

`verify` compares PyTorch and ONNX Runtime outputs numerically. It validates an exported model path, not a model's scientific performance, clinical safety, or hardware-specific latency.

## Quantized inference

```python
from biosignal_fm.deployment import RealtimeInference

runtime = RealtimeInference(model, modality="emg", quantize=True)
representation = runtime.predict(signal)
print(runtime.quantization_active)
```

Dynamic int8 quantization may be hardware- and model-dependent. BioSignal-FM reports whether quantization actually activated and falls back to full precision when its supported runtime workaround cannot activate it. Always benchmark both paths on the target hardware:

```bash
bsfm benchmark --checkpoint checkpoint.pt --n-channels 16 --signal-length 400 --n-runs 100
bsfm benchmark --checkpoint checkpoint.pt --n-channels 16 --signal-length 400 --n-runs 100 --no-quantize
```

## API service

The service binds to loopback by default. Use it locally first:

```bash
BSFM_API_KEY='replace-with-a-strong-secret' \
  bsfm serve --model-dir ./checkpoints --port 8000
```

Use `--public` only when an operator intentionally deploys behind TLS, an authenticated reverse proxy, request/body limits, rate limits, and audited logs:

```bash
BSFM_API_KEY='replace-with-a-strong-secret' \
  bsfm serve --public --model-dir ./checkpoints --port 8000
```

The model directory is an operator-controlled staging area. The API accepts only a **relative** checkpoint path under that directory. It rejects absolute paths and `..` traversal. `torch.load(weights_only=True)` reduces unsafe deserialization risk, but operators must still stage trusted artifacts only.

| Endpoint | Authentication | Contract |
|---|---|---|
| `GET /health` | None | Service status, package version, and number of loaded models. |
| `GET /models` | None | Registered model metadata. Do not expose publicly if model metadata is sensitive. |
| `GET /models/{id}/info` | None | Metadata and SHA-256 for one registered model. |
| `POST /models/register` | `X-API-Key` | Registers a checkpoint staged at a relative path within the configured model directory. |
| `DELETE /models/{id}` | `X-API-Key` | Removes an in-memory registry entry. |
| `POST /predict` | `X-API-Key` | Runs one finite signal window with the exact registered `(channels, samples)` shape. |
| `WS /ws/predict/{id}` | First WebSocket message | Authenticates with `{"api_key": "..."}` and then receives finite, exact-shape signal windows. |

### Register a model

With `--model-dir ./checkpoints`, stage `./checkpoints/demo.pt` locally and register it as `demo.pt`:

```bash
curl -X POST http://127.0.0.1:8000/models/register \
  -H 'X-API-Key: replace-with-a-strong-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "path": "demo.pt",
    "name": "demo",
    "modality": "emg",
    "n_channels": 16,
    "signal_length": 400,
    "quantize": true
  }'
```

### Call a prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'X-API-Key: replace-with-a-strong-secret' \
  -H 'Content-Type: application/json' \
  -d '{"signal": [[0.0]], "model_id": "<registered-uuid>"}'
```

A real request must supply the registered channel count and sample length. The API rejects malformed, non-finite, or wrong-shape signals before inference.

### Stream with WebSocket

The API key is intentionally **not** accepted in the URL. Query strings are routinely retained by access logs and intermediary telemetry. Authenticate in the first message instead:

```python
import asyncio
import json

import numpy as np
import websockets


async def stream(model_id: str, api_key: str) -> None:
    uri = f"ws://127.0.0.1:8000/ws/predict/{model_id}"
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({"api_key": api_key}))
        while True:
            window = np.random.randn(16, 400).astype(np.float32)
            await websocket.send(json.dumps({"signal": window.tolist()}))
            print(json.loads(await websocket.recv()))


asyncio.run(stream("<registered-uuid>", "replace-with-a-strong-secret"))
```

A malformed window returns an error message while leaving an authenticated session open. Deploy a rate limit and maximum WebSocket message size at the reverse proxy or ASGI-server boundary.

## Docker and Compose

Build the image locally:

```bash
docker build -t biosignal-fm:local .
```

The image runs as a non-root user. Compose makes the root filesystem read-only, drops Linux capabilities, uses `no-new-privileges`, mounts model/data paths read-only, and binds the API only to `127.0.0.1:8000` on the host. It intentionally does **not** publish the UI port; expose it only through an authenticated reverse proxy when needed.

Before running Compose, set a strong API key in the deployment environment:

```bash
export BSFM_API_KEY='replace-with-a-long-random-secret'
docker compose up --build
```

Compose fails early if `BSFM_API_KEY` is missing. It also sets `BSFM_MODEL_DIR=/home/bsfm/checkpoints`; therefore, stage checkpoint files in `./checkpoints` and register relative names such as `demo.pt`.

For any network-facing deployment, terminate TLS, restrict origins and ingress, impose request and message-size limits, add rate limiting, rotate the API key, and review application/proxy logs for sensitive content. Do not mount credentialed or restricted data into a public deployment. See [Data Governance](data_governance.md) and the [Security Policy](https://github.com/qussaiadlbi/biosignal-fm/blob/main/SECURITY.md).
