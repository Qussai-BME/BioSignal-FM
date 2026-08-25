"""Deploy page: ONNX export, benchmark, REST playground."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from biosignal_fm.ui.theme import inject_css, stage_indicator

st.set_page_config(page_title="Deploy — BioSignal-FM", layout="wide")
inject_css()
st.markdown(stage_indicator("Deploy"), unsafe_allow_html=True)

st.title("Deploy")
st.markdown("### Deployment & Serving")

st.markdown(
    """
    BioSignal-FM supports deployment via:

    - **ONNX export** with REAL numerical parity verification
    - **Dynamic int8 quantization** for sub-50ms CPU inference
    - **FastAPI REST + WebSocket** with API-key auth
    - **Docker** multi-stage build with non-root user
    """
)

st.markdown("### ONNX Export & Verification")
st.code(
    """
    from biosignal_fm.deployment import OnnxExporter
    from biosignal_fm.models import FoundationModel

    model = FoundationModel.load("model.pt")
    exporter = OnnxExporter()
    onnx_path = exporter.export(model, "model.onnx", modality="emg",
                                 n_channels=16, signal_length=400)
    # REAL numerical parity check (not just a load test)
    assert exporter.verify(model, onnx_path, modality="emg",
                           n_channels=16, signal_length=400, n_samples=100)
    """,
    language="python",
)

st.markdown("### PassMark Benchmark")

st.markdown(
    """
    Latency is measured with BLAS forced to a single thread to ensure
    the reported numbers are true single-thread measurements (not
    inflated by parallel BLAS). The `single_thread` field in the output
    reflects whether `threadpoolctl` was actually available.
    """
)

st.markdown("#### Live benchmark (measured right now, on this session's CPU)")
st.caption(
    "Builds a small model, exports it to ONNX with real numerical parity "
    "verification, then measures real full-precision vs int8-quantized "
    "latency. These are genuinely measured numbers for this session's "
    "hardware, not the illustrative scale-comparison table further down."
)

if st.button("Run live ONNX export + benchmark", type="primary"):
    import tempfile
    from pathlib import Path

    from biosignal_fm.config import Modality, ModelConfig
    from biosignal_fm.deployment import OnnxExporter, RealtimeInference
    from biosignal_fm.models import FoundationModel

    cfg = ModelConfig(d_model=96, n_heads=4, n_layers=3, d_ff=192, patch_length=32, patch_stride=16)
    n_ch = {m.value: 16 for m in Modality}
    model = FoundationModel(cfg, n_ch)
    n_params = sum(p.numel() for p in model.parameters())

    with st.spinner("Exporting to ONNX and verifying numerical parity..."):
        exporter = OnnxExporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            exporter.export(model, onnx_path, modality="emg", n_channels=16, signal_length=400)
            parity_ok = exporter.verify(
                model, onnx_path, modality="emg", n_channels=16, signal_length=400, n_samples=10
            )
    st.write(
        f"ONNX export: **{n_params:,} parameters**, numerical parity "
        f"{'**passed**' if parity_ok else '**FAILED**'} (PyTorch vs ONNX Runtime outputs)."
    )

    with st.spinner("Benchmarking full precision..."):
        rt_fp = RealtimeInference(model, modality="emg", quantize=False)
        bench_fp = rt_fp.benchmark(n_channels=16, signal_length=400, n_runs=20, warmup=3)

    with st.spinner("Benchmarking int8 quantized..."):
        rt_q = RealtimeInference(model, modality="emg", quantize=True)
        bench_q = rt_q.benchmark(n_channels=16, signal_length=400, n_runs=20, warmup=3)

    live_df = pd.DataFrame(
        [
            {
                "Configuration": "Full precision",
                "Mean (ms)": bench_fp["mean_ms"],
                "P50 (ms)": bench_fp["p50_ms"],
                "P99 (ms)": bench_fp["p99_ms"],
            },
            {
                "Configuration": f"Int8 quantized ({'active' if rt_q.quantization_active else 'FELL BACK to full precision'})",
                "Mean (ms)": bench_q["mean_ms"],
                "P50 (ms)": bench_q["p50_ms"],
                "P99 (ms)": bench_q["p99_ms"],
            },
        ]
    )
    st.dataframe(
        live_df.style.format({"Mean (ms)": "{:.3f}", "P50 (ms)": "{:.3f}", "P99 (ms)": "{:.3f}"}),
        width="stretch",
    )
    if not rt_q.quantization_active:
        st.warning(
            "Quantization fell back to full precision on this session's hardware/torch "
            "build — the two rows above measure the same thing."
        )
    elif bench_q["mean_ms"] > 0:
        ratio = bench_fp["mean_ms"] / bench_q["mean_ms"]
        if ratio >= 1.0:
            st.success(f"Quantization was {ratio:.2f}x faster on this session's CPU.")
        else:
            st.info(
                f"Quantization was actually {1 / ratio:.2f}x **slower** here, not faster. "
                "This is a real, known effect: dynamic int8 quantization has fixed "
                "per-call packing/dispatch overhead that can outweigh its compute "
                "savings for smaller models or CPUs without optimized int8 kernels — "
                "it doesn't always help, and this benchmark reports what actually "
                "happened rather than assuming it would."
            )

st.markdown("#### Scale comparison (illustrative)")
st.caption(
    "The live benchmark above measures one model size on this session's "
    "CPU. The table below is an illustrative *scale* comparison across "
    "model sizes on a representative CPU — do NOT cite these as measured; "
    "run the CLI or the live benchmark above for real numbers on your hardware."
)

# Demo benchmark table — clearly labeled as illustrative.
bench_data = pd.DataFrame(
    {
        "Configuration (illustrative)": [
            "Base d=512 L=12 (full precision)",
            "Base d=512 L=12 (int8 quantized)",
            "Distilled d=256 L=6 (full precision)",
            "Distilled d=256 L=6 (int8)",
        ],
        "Mean (ms, illustrative)": [42.3, 18.7, 11.2, 6.8],
        "P50 (ms, illustrative)": [41.8, 18.4, 11.0, 6.7],
        "P95 (ms, illustrative)": [48.1, 22.3, 13.5, 8.1],
        "P99 (ms, illustrative)": [52.0, 24.1, 14.8, 9.2],
    }
)
st.dataframe(bench_data, width="stretch")
st.caption(
    "Numbers above are illustrative — do NOT cite them as measured. "
    "Run `biosignal-fm benchmark` on your hardware for real numbers."
)

st.markdown("### REST API Playground")
st.markdown("#### Live request (real FastAPI app, in-process — not a mock)")
st.caption(
    "Builds a small model, registers it with a real ModelRegistry, and "
    "sends a real request through the actual FastAPI app object via "
    "Starlette's TestClient. The JSON below is a genuine response, not a "
    "hand-written example."
)

if st.button("Send live /predict request"):
    import tempfile
    from pathlib import Path

    from biosignal_fm.config import Modality, ModelConfig
    from biosignal_fm.deployment import ModelRegistry, create_app
    from biosignal_fm.models import FoundationModel
    from fastapi.testclient import TestClient

    cfg = ModelConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, patch_length=16, patch_stride=8)
    n_ch = {m.value: 16 for m in Modality}
    model = FoundationModel(cfg, n_ch)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_dir = Path(tmpdir) / "registry"
        registry = ModelRegistry(storage_dir=registry_dir)
        ckpt_path = registry_dir / "model.pt"
        model.save(ckpt_path)
        app = create_app(registry=registry, api_key="playground-demo-key")
        client = TestClient(app)

        st.markdown("**1. Register the model** — `POST /models/register`")
        reg_resp = client.post(
            "/models/register",
            headers={"X-API-Key": "playground-demo-key"},
            json={
                "path": "model.pt",
                "name": "playground-demo",
                "modality": "emg",
                "n_channels": 16,
                "signal_length": 64,
                "quantize": False,
            },
        )
        st.code(f"HTTP {reg_resp.status_code}\n{reg_resp.text}", language="json")
        registration_body = reg_resp.json()
        model_id = registration_body.get("model_id")
        if model_id is None:
            st.error("The local registration request failed; prediction was not attempted.")
            st.stop()

        st.markdown("**2. Predict** — `POST /predict`")
        import numpy as np

        signal = np.random.randn(16, 64).astype(np.float32).tolist()
        pred_resp = client.post(
            "/predict",
            headers={"X-API-Key": "playground-demo-key"},
            json={"signal": signal, "model_id": model_id},
        )
        body = pred_resp.json()
        if "cls_token" in body:
            body["cls_token"] = [
                [round(x, 4) for x in row[:4]] + ["..."] for row in body["cls_token"]
            ]
        st.code(f"HTTP {pred_resp.status_code}\n{body}", language="json")
        st.caption(
            "cls_token truncated to the first 4 values per row for display. "
            "`latency_ms` here is the *first-ever* call on a freshly-built "
            "model in this session — first-call latency is measurably higher "
            "than steady-state (confirmed: repeated calls on the same model "
            "drop to sub-millisecond), and can be substantial in a shared/"
            "constrained environment. The live benchmark above already warms "
            "up before measuring, which is why it reports steady-state "
            "numbers instead; this playground intentionally doesn't hide the "
            "cold-start cost a real first request would pay."
        )

        st.markdown("**3. List models (no auth)** — `GET /models`")
        list_resp = client.get("/models")
        st.code(f"HTTP {list_resp.status_code}\n{list_resp.text}", language="json")

st.markdown("#### Equivalent curl (against a real running server)")
st.code(
    """
    # Predict (requires API key)
    curl -X POST http://localhost:8000/predict \\
         -H "X-API-Key: $BSFM_API_KEY" \\
         -H "Content-Type: application/json" \\
         -d '{"signal": [[...]], "model_id": "uuid-here"}'

    # List models (no auth)
    curl http://localhost:8000/models
    """,
    language="bash",
)
st.caption(
    "For the streaming WebSocket protocol (continuous inference over one "
    "persistent connection), see the Deployment guide's WebSocket section."
)
