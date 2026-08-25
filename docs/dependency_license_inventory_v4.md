# Dependency and License Inventory — V4

## Scope

This document records dependencies declared in `pyproject.toml` and their architectural placement. It is not legal advice and does not replace review of the exact installed versions, notices, vulnerabilities, and licenses in the target release environment.

## V4 principle

The core must not install every package needed by a UI, HTTP service, dataset reader, or deep-learning workflow. V4 divides dependencies into capability-specific extras so users can install the smallest suitable set.

| Extra | Principal dependencies | Use | Core requirement? |
|---|---|---|---|
| Base installation | NumPy, PyYAML, Typer, Rich | Signal contracts, configuration, lightweight CLI. | Yes |
| `scientific` | SciPy, scikit-learn, pandas, matplotlib, tqdm, joblib, tabulate, PyWavelets | Processing, evaluation, conventional pipelines. | No |
| `ml` | PyTorch | Models and training. | No |
| External MLflow tracking | Deliberately deferred; no release extra. | Upstream MLflow currently conflicts with the Cryptography security baseline. | No |
| `deployment` | ONNX, ONNX Runtime, threadpoolctl | Export, verification, and runtime inference. | No |
| `data-eeg` | MNE | EEG/ECoG reader and adapter paths. | No |
| `data-ecg` | WFDB | ECG reader and adapter path. | No |
| `data-fnirs` | h5py | Legacy fNIRS loader. | No |
| `api` | FastAPI, Uvicorn | Local or operator-managed API service. | No |
| `ui` | Streamlit | User interface. | No |
| `stats` | statsmodels | Optional statistical models. | No |
| `dev` | pytest, pytest-cov, httpx, ruff, mypy, pre-commit, statsmodels, threadpoolctl | Development and verification. | No |
| `docs` | MkDocs Material, mkdocstrings | Documentation build. | No |

The legacy `fm` and `data` extras remain for v3.3 compatibility but are not the preferred integration path. External MLflow tracking is excluded until a released upstream version supports the secure Cryptography baseline and passes the dependency audit.

## Dependency boundaries

| Area | Direct dependencies that are prohibited or intentionally absent | Reason |
|---|---|---|
| `biosignal_fm.core` | MNE, WFDB, PyTorch, Streamlit, FastAPI | The core accepts canonical contracts, not reader, model, or UI objects. |
| `biosignal_fm.modalities` | Optional adapters may be declared but registry discovery must not import heavy readers by default. | The registry remains inspectable in a clean core installation. |
| `biosignal_fm.services` | UI, HTTP, and dataset-reader dependencies. | CLI, UI, and API are clients of one research service. |
| `biosignal_fm.data` smoke path | Remote data or cloud-service dependency. | Tests remain local and reproducible. |

## License and distribution controls

The project itself is licensed under Apache License 2.0. Before distributing a package, container, or artifact, a maintainer must:

1. Review licenses and notices for the exact installed versions, not only package names.
2. Preserve any required third-party notices in the distribution or image.
3. Review each real dataset license independently; a reader-library license does not grant data redistribution rights.
4. Never place raw biosignals, identifiers, credentialed data, API keys, or data-use agreements in the repository or public image.
5. Avoid presenting dependencies or their outputs as regulatory compliance or a privacy guarantee.
6. Run a dependency-vulnerability audit against the exact release environment and record any accepted exceptions.

## Release verification baseline

```bash
python -m pip install -e '.[all]'
python -m pytest -q
ruff check biosignal_fm tests scripts
ruff format --check biosignal_fm tests scripts
mypy biosignal_fm scripts
mkdocs build --strict
pip-audit --strict
```

A distribution should also generate an environment-appropriate software bill of materials or lockfile review. A package name in configuration does not establish the legal status or security of a remote release.
