# Pull Request Template

## Description

Briefly describe what this PR changes and why.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactor (no functional changes)
- [ ] Performance improvement
- [ ] Test addition/improvement

## Checklist

- [ ] Code is formatted: `ruff format biosignal_fm tests`
- [ ] Linting passes: `ruff check biosignal_fm tests`
- [ ] Type checks pass: `mypy biosignal_fm`
- [ ] Tests pass: `pytest -v`
- [ ] Coverage stays above 75%: `pytest --cov=biosignal_fm`
- [ ] No `pickle.load` on user-supplied paths
- [ ] No silent `try/except` that swallows errors
- [ ] No hardcoded paths (use `pathlib.Path`)
- [ ] No emojis in production UI (only in dev logs)
- [ ] Public symbols re-exported in `__init__.py`
- [ ] Google-style docstrings with Args/Returns/Raises/Example
- [ ] LICENSE remains full Apache 2.0 (202 lines)
- [ ] Author remains "Qussai Adlbi" in new files (unless external contributor)

## Statistical Rigor (for evaluation code)

- [ ] Cross-validation is subject-aware (LOSO or LODO)
- [ ] Holm-Šídák formula is `1-(1-p)^(m-k+1)` (NOT Bonferroni-Holm `p*(m-k+1)`)
- [ ] Bootstrap CIs use BCa (not naive percentile)
- [ ] Effect sizes use Hedges' g (small-N corrected Cohen's d)
- [ ] UI confusion matrix aggregates across folds (not in-sample)

## Testing

Describe the tests you added or modified.

## Reproducibility

- [ ] New training runs produce a RunManifest
- [ ] SHA-256 of all outputs recorded
- [ ] Env fingerprint captured
- [ ] Config snapshot saved

## Related Issues

Closes #XXX (if applicable)
