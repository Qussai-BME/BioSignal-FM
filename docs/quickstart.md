# Quickstart

This guide gets a clean environment to an inspected registry, a clearly labeled synthetic smoke path, and an optional local service. It does **not** produce a real-data benchmark, a validated representation, or a clinical result.

## Installation

```bash
git clone https://github.com/qussaiadlbi/biosignal-fm.git
cd biosignal-fm
python -m pip install --upgrade pip
python -m pip install -e '.[ml,scientific,dev]'
```

Add optional capabilities only when needed:

```bash
python -m pip install -e '.[deployment,api]'  # ONNX and local API
python -m pip install -e '.[ui]'              # Streamlit interface
```

Inspect the installed package and modality registry:

```bash
bsfm info
bsfm inspect
bsfm inspect --modality ecog
```

## Synthetic smoke path

A synthetic path is useful for verifying software wiring. It is labeled synthetic and benchmark-ineligible.

```bash
bsfm pretrain --config configs/exp.yaml --output-dir runs/smoke --steps 1 --synthetic-demo
```

The explicit `--synthetic-demo` flag is required. Do not report outputs from this command as training on a real dataset, a benchmark, or a scientific conclusion.

## Fine-tuning and evaluation utilities

The bundled CLI fine-tune and evaluate demonstrations use synthetic data and emit a warning. They are intended to exercise the code path only:

```bash
bsfm finetune \
  --checkpoint runs/smoke/checkpoint_final.pt \
  --n-classes 4 --n-channels 16 --signal-length 400 \
  --strategy linear --output-dir runs/finetune-smoke

bsfm evaluate \
  --checkpoint runs/smoke/checkpoint_final.pt \
  --n-classes 4 --n-channels 16 --signal-length 400 \
  --protocol loso
```

For a real study, build a dataset-specific loader/adapter path, document the dataset license and version, choose a split before training, fit preprocessing only on training data, and preserve `RunManifest` output. See [Research Chain](research_chain.md) and [Data Governance](data_governance.md).

## ONNX export and local benchmark

```bash
bsfm export-onnx \
  --checkpoint runs/smoke/checkpoint_final.pt \
  --output model.onnx --n-channels 16 --signal-length 400

bsfm benchmark \
  --checkpoint runs/smoke/checkpoint_final.pt \
  --n-channels 16 --signal-length 400 --n-runs 100
```

ONNX export includes numerical parity verification. Benchmark values are local hardware measurements only. See [Quantized Inference](deployment.md#quantized-inference) for interpretation and limitations.

## Local API

Stage trusted checkpoint files in one directory and bind to loopback by default:

```bash
export BSFM_API_KEY='replace-with-a-strong-secret'
bsfm serve --model-dir ./checkpoints --port 8000
```

For a network-facing deployment, use `--public` only behind TLS, a reverse proxy, rate limits, and request-size controls. The registration endpoint accepts a relative filename inside `--model-dir`, never an absolute path. See [Deployment](deployment.md).

## Verify a checkout

```bash
python -m pytest -q
ruff check biosignal_fm tests scripts
ruff format --check biosignal_fm tests scripts
mypy biosignal_fm scripts
mkdocs build --strict
```

The release verification report records the measured result for the final package.
