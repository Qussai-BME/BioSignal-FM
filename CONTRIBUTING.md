# Contributing to BioSignal-FM

First off, thank you for considering a contribution. BioSignal-FM is an
open-source research project and welcomes contributions from biomedical
engineers, ML researchers, and software engineers alike.

## Quick Checklist Before Submitting a PR

- [ ] Code is formatted: `ruff format biosignal_fm tests`
- [ ] Linting passes: `ruff check biosignal_fm tests`
- [ ] Type checks pass: `mypy biosignal_fm`
- [ ] Tests pass: `pytest -v`
- [ ] Coverage stays above 80%: `pytest --cov=biosignal_fm`
- [ ] No `pickle.load` on user-supplied paths
- [ ] No silent `try/except` that swallows errors
- [ ] No hardcoded paths (use `pathlib.Path`)
- [ ] No emojis in production UI (only in dev logs)
- [ ] Public symbols re-exported in `__init__.py`
- [ ] Google-style docstrings with Args/Returns/Raises/Example
- [ ] LICENSE remains full Apache 2.0 (202 lines)
- [ ] Author remains "Qussai Adlbi" in new files (unless external contributor)

## Development Setup

```bash
# Clone the repo
git clone https://github.com/qussaiadlbi/biosignal-fm.git
cd biosignal-fm

# Create a virtual environment (Python 3.10+ required)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install in editable mode with all extras
pip install -e ".[fm,dev]"

# Install CPU-only PyTorch if you don't have a GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install pre-commit hooks
pre-commit install

# Verify the install
pytest -v
```

## Project Structure

```
biosignal_fm/
├── config.py          # Frozen dataclasses, YAML loader
├── reproducibility.py # Seeds, RunManifest, env fingerprint
├── data/              # Modality-specific loaders
├── preprocessing/     # Filters, normalization, patching
├── models/            # FoundationModel, SSL heads, task heads
├── training/          # SSLPretrainer, FineTuner
├── evaluation/        # LOSO, LODO, statistics
├── deployment/        # ONNX, FastAPI, realtime
├── tracking/          # MLflow + Local JSON
├── api/               # REST + WebSocket
├── ui/                # Streamlit dashboard
└── cli/               # Typer CLI
```

## Coding Standards

- **Python:** 3.10+ (use `match` statements, `X | None` syntax, etc.)
- **Type hints:** Required on all public functions.
- **Docstrings:** Google style with Args/Returns/Raises/Example sections.
- **Line length:** 100 chars max (enforced by ruff).
- **Imports:** Sorted by ruff (isort-compatible). No unused imports.
- **Tests:** `pytest`, parametrized where possible, ≥80% coverage required.
- **Error handling:** No silent `except: pass`. Always log or re-raise.
- **File paths:** Always use `pathlib.Path`, never string concatenation.
- **Reproducibility:** All training runs must produce a `RunManifest`.

## Statistical Rigor (for evaluation code)

When contributing evaluation code, the following rules are non-negotiable:

1. Cross-validation is subject-aware (LOSO or LODO). Never random K-fold
   on biosignal data (causes subject-level leakage).
2. The Holm-Šídák step-down correction uses formula `1 - (1 - p)^(m - k + 1)`,
   **NOT** Bonferroni-Holm `p * (m - k + 1)`.
3. Bootstrap confidence intervals use BCa (bias-corrected and accelerated),
   not naive percentile.
4. Effect sizes use Hedges' g (small-N corrected Cohen's d).
5. The UI confusion matrix must aggregate predictions across all CV folds,
   never in-sample predictions.

## Pull Request Process

1. Fork the repo and create a feature branch
   (`git checkout -b feature/my-new-feature`).
2. Make your changes following the standards above.
3. Run the full check suite locally:
   ```bash
   ruff format biosignal_fm tests
   ruff check --fix biosignal_fm tests
   mypy biosignal_fm
   pytest -v
   ```
4. Commit with a clear message following
   [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(models): add span-masked reconstruction head
   fix(evaluation): correct Holm-Šídák formula
   docs(readme): add quickstart section
   ```
5. Open a PR against `develop`. Fill in the PR template.
6. Wait for CI to pass on Python 3.10 / 3.11 / 3.12.
7. Request review from a maintainer.

## Reporting Issues

When opening an issue, please include:

- Python version (`python --version`)
- OS and architecture
- BioSignal-FM version (`python -c "import biosignal_fm; print(biosignal_fm.__version__)"`)
- A minimal reproducer (ideally a code snippet)
- The full traceback if an exception was raised
- Whether you can reproduce on the latest `main` branch

## License

By contributing, you agree that your contributions will be licensed under the
Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Attribution

This contributing guide was inspired by the contributing guides of
[scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md),
[PyTorch](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md),
and [OpenBCI](https://github.com/OpenBCI/OpenBCI_GUI).
