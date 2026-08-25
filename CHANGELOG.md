# Changelog

All notable changes to BioSignal-FM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.0.1] - 2026-08-25

Publication-readiness corrections found during an independent line-by-line
review of the 4.0.0 pre-publication release bundle. No architecture,
API, or test-suite behavior changed — only release hygiene and metadata.

### Fixed
- `CITATION.cff` was still declaring the pre-V4 version (`0.1.2`) and a
  placeholder ORCID (`0000-0000-0000-0000`). Version now matches the
  package (`4.0.1`); the placeholder ORCID was removed rather than
  shipped as a fake identifier.
- Removed stray local-development artifacts that the project's own
  `.gitignore` already excludes but that had been included in the
  0.1.2 packaged release bundle anyway: `mlflow.db` (a local MLflow
  tracking database containing a developer's absolute filesystem path),
  `build/`, `biosignal_fm.egg-info/`, the generated `site/` docs build,
  and `runs/` (local synthetic-data run artifacts, including a results
  table already correctly labeled non-benchmark but not appropriate to
  ship as part of a citable release).
- Added a root `requirements.txt` for Streamlit Community Cloud, which
  does not reliably parse this project's PEP 621 `pyproject.toml`
  (Community Cloud's dependency detection assumes Poetry's
  `[tool.poetry]` format and fails on a standard `[project]` table).
- Removed the hardcoded `server.address = "localhost"` /
  `browser.serverAddress = "localhost"` from `.streamlit/config.toml`,
  which had no effect locally but could conflict with how Streamlit
  Community Cloud manages its own networking.

## [4.0.0] - 2026-08-22

A controlled architectural migration from the v3.3 implementation into one
canonical BioSignal-FM research platform. This is a major version because the
public architecture, package extras, modality taxonomy, and evidence labelling
have been clarified; it is not a claim of a completed scientific foundation
model.

### Added

- A library-independent V4 signal contract: `SignalMetadata`, `Signal`,
  `SignalBatch`, `SignalEvent`, and structured `SignalProvenance`.
- An explicit modality registry with first-class EMG, EEG, and ECG entries;
  an experimental ECoG/iEEG adapter path; and a legacy-compatible optional
  fNIRS extension.
- Edge adapters that convert V3.3 samples, transparent array mappings, and
  MNE-like reader objects to the canonical signal contract without importing
  MNE or WFDB into the core.
- A research application service that enforces the canonical ordering:
  preprocessing → encoder → optional fusion → task head.
- V4 architecture tests for contracts, registry maturity, provenance,
  multimodal fusion ordering, and forbidden core dependencies.
- Structured dataset provenance, research protocol, and runtime context in
  `RunManifest`.

### Changed

- Repositioned BioSignal-FM as a modular multimodal biosignal research
  platform. Documentation and the dashboard no longer present the repository
  as a scientifically validated foundation model, clinical product, or
  benchmark leader without documented evidence.
- Made synthetic workflow selection explicit in `bsfm pretrain` through
  `--synthetic-demo`; synthetic fine-tune and evaluation paths now emit clear
  non-benchmark / non-inference warnings.
- Added `bsfm inspect` and a modality registry table in `bsfm info`.
- Added experimental `ecog` to `Modality`, model modality counts, synthetic
  smoke-test generation, preprocessing configuration, and the visual system.
- Split package dependencies into a minimal core plus `scientific`, `ml`,
  `tracking`, `deployment`, `data-eeg`, `data-ecg`, `data-fnirs`, `api`, and
  `ui` extras. Legacy `fm` and `data` extras remain during the transition.

### Deprecated

- The names `FoundationModel` and `DistilledFoundationModel` remain available
  for compatibility but are representation-model implementations, not evidence
  that the repository is a validated foundation model.
- fNIRS remains available as an optional legacy-compatible modality and is no
  longer a V4 core modality.

### Fixed

- Synthetic sample metadata now records an explicit non-benchmark data source,
  generation reason, and provenance fields.
- The audit test no longer needs to scan arbitrary virtual-environment files;
  V4 architecture checks are scoped to the project source tree.

## [0.1.2] - 2026-08-21

A polish release: content-accuracy audit across every doc/citation file
plus a full visual redesign of the dashboard. No functional/API changes —
everything here was either a documentation-accuracy fix or presentation.

### Fixed

**Documentation accuracy (the same Holm-Šídák off-by-one formula bug fixed
in code in 0.1.1 turned out to also be copy-pasted into 6 separate prose
locations across the repo):**
- `docs/research/preregistration.md`, `docs/research/paper_draft.md` (×2),
  `CONTRIBUTING.md` all repeated the wrong `1-(1-p)^(m-k)` formula (missing
  the `+1`) that the corresponding code docstrings already had fixed.
- `CITATION.cff`'s abstract claimed **"state-of-the-art performance"** —
  directly contradicting the project's own honestly-reported results
  (baselines currently outperform the not-yet-pretrained model on the
  synthetic-data comparison; see the paper draft). Rewritten to describe
  what's actually verified vs. not-yet-validated.
- An unverified "~50M parameters" claim (actual: 39.4M for the base
  config) was repeated in `app.py`, `1_Overview.py`,
  `models/distilled.py` (×2), and `README.md`; the distilled variant's
  "~11M" claim (actual: 5.1M) was wrong in the same two files. All six
  corrected to measured values.
- `benchmarks/README.md` documented CLI flags that don't exist
  (`--dataset`, `--protocol loro`, `--cross-modal`, `--source`,
  `--target`) — same class of bug as 0.1.1's CLI fixes, just in docs
  instead of code. Rewritten to the real, verified `bsfm evaluate` usage.

**Documentation site (`mkdocs build` had never actually been run):**
- The nav referenced 5 pages that didn't exist in the repo
  (`architecture.md`, `quickstart.md`, `api_reference.md`,
  `research_chain.md`, `deployment.md`) — a real `mkdocs build` produced
  10 warnings, including broken links from the one page that did exist.
  All 5 written, using only commands/APIs verified against the actual
  package (every CLI flag cross-checked against real `--help` output).
- `mkdocstrings` had no `docstring_style` configured, so it silently
  defaulted to Google-style parsing against a 100%-NumPy-style codebase —
  doctests' `Examples` sections weren't recognized as code blocks, and
  bracket characters in doctest *output* (`[True, False, ...]`) got
  misparsed as markdown link syntax, breaking the build. Fixed by setting
  `docstring_style: numpy`.
- Mermaid diagrams didn't render (missing `custom_fences` config for
  `pymdownx.superfences` — the class="mermaid" the Material theme's JS
  looks for was never actually emitted). Fixed.
- CI now has a `docs` job that runs `mkdocs build --strict`, so a broken
  nav/link/docstring can't silently ship again.

### Changed

**Dashboard visual design (full redesign):**
- The custom theme was only ever visible on one of six dashboard
  scripts. `app.py` injected its CSS via `st.markdown(..., unsafe_allow_html=True)`,
  but each file under `ui/pages/` is an independently-executed script in
  Streamlit's multi-page model — none of the 5 actual feature pages ever
  received any of it. Refactored into a shared `inject_css()` in
  `theme.py`, called at the top of all 6 scripts.
- Replaced the generic indigo/teal palette with a system grounded in the
  actual subject matter: each of the four modalities (EMG/ECG/EEG/fNIRS)
  gets its own signature color, used as a consistent wayfinding device
  everywhere that modality appears — the same principle real multi-channel
  signal-monitoring equipment uses to distinguish channels. Every color
  pairing is verified programmatically (WCAG contrast ratio computed via
  the real relative-luminance formula, not eyeballed) in the new
  `tests/unit/test_ui_theme.py`, which also checks the four modality
  colors are perceptually distinguishable from each other via CIE76
  Delta-E (a first attempt at this check used WCAG contrast ratio as a
  distinguishability proxy, which is the wrong metric — it measures
  readability, not hue difference, and gave a false positive).
- Typography moved from the default Inter+JetBrains-Mono pairing to IBM
  Plex Sans/Mono + Space Grotesk — IBM Plex was designed by IBM
  specifically for technical/scientific documentation, which is a better
  fit than a generic UI-framework default.
- Added a real 5-stage pipeline indicator (Overview→Pretrain→Finetune→
  Evaluate→Deploy) to every page, since that ordering is the actual
  pipeline sequence, not a decorative device.
- `.streamlit/config.toml`'s native theme (which applies before, and
  alongside, the custom CSS) still had the old indigo palette; synced to
  the new tokens so there's no flash/mismatch between Streamlit's native
  theming and the injected CSS.
- Injected CSS validated with a real CSS parser (`tinycss2`, 0 errors)
  and the new SVG waveform element validated as well-formed XML —
  checked, not assumed.
- The REST API Playground on the Deploy page (previously a static curl
  example) now issues real requests through the actual FastAPI app
  in-process via `TestClient` and shows the genuine JSON response.

### Added

- `docs/architecture.md`, `docs/quickstart.md`, `docs/api_reference.md`,
  `docs/research_chain.md`, `docs/deployment.md` (see Fixed above).
- `tests/unit/test_ui_theme.py` — WCAG contrast + perceptual
  distinguishability regression tests for the design system.
- A `docs` CI job running `mkdocs build --strict`.

## [0.1.1] - 2026-08-18

A stabilization release. Every command and feature documented in 0.1.0 is
now actually exercised end-to-end and verified to work — several were not.
No public API changes; this is a pure bug-fix + hardening release.

### Fixed

**Training/evaluation CLI commands** (`pretrain`, `finetune`, `evaluate` —
none of these were covered by CLI tests before this release, which is why
all three shipped broken):
- `pretrain`: the synthetic SSL target tensor hardcoded a patch count of 23,
  which didn't match the 24 patches the model actually produces for the
  default `patch_length`/`patch_stride`/signal length — crashed on the
  first training step every time.
- `pretrain`: the `--steps` override called `TrainingConfig.replace()`,
  which doesn't exist (`dataclasses.replace()` is a free function, not a
  method) — crashed whenever `--steps` was passed.
- `finetune`: `FineTuner.evaluate()` expects an iterable of batches; a
  single pre-stacked batch tuple was passed directly, which iterated over
  its 3 tensors instead of over batches — crashed during evaluation.
- `finetune`: the model/head/optimizer were created once and reused across
  all 6 LOSO folds instead of being reset per fold, so training silently
  accumulated across folds. Because each "held-out" subject had already
  appeared in earlier folds' training sets, the reported LOSO accuracy was
  not a valid held-out estimate. Each fold now starts from an independent
  copy of the loaded checkpoint.
- `evaluate`: the per-method score matrix was built as (methods × datasets)
  but `friedman_nemenyi_test` documents and expects (datasets × methods) —
  produced statistically meaningless output (e.g. "average rank = 8.00" for
  3 methods) without ever raising an error.

**Statistics module:**
- `wilcoxon_holm_sidak()` returned the full dict from
  `holm_sidak_correction()` instead of the `list[bool]` its own signature
  and docstring promise, so callers iterating the result got dict *keys*
  (e.g. the string `"corrected_pvalues"`) instead of booleans.
- `holm_sidak_correction()`'s own docstring example, and the copy of it in
  `wilcoxon_holm_sidak()`, showed a confabulated (never actually run)
  output that didn't match the function's real, independently-verified-by-hand
  result.

**Documentation examples (11 confabulated doctest failures, found by
running `pytest --doctest-modules` — which nothing in this project had ever
done):** wrong patch-count in two `models/foundation.py` examples (same
23-vs-24 bug as the `pretrain` crash above, just in docs instead of code),
stale numpy scalar repr in `evaluation/cross_validation.py` and
`evaluation/metrics.py`, a `RunManifest.create()` example using a parameter
(`output_dir`) that doesn't exist, a knowledge-distillation example pairing
a smaller "teacher" with a larger "student" (backwards, and dimensionally
incompatible with `distillation_loss`), a JEPA example comparing
predictions against unfiltered targets instead of the masked span, and
`.fit()`-returns-self patterns not accounted for in `preprocessing/`.

**Streamlit dashboard:**
- `ui/app.py` used a relative import (`from .theme import ...`), which
  fails every time because Streamlit executes the entry file as a
  top-level script, not as a package member — the entire dashboard was
  unusable (`ImportError` on load, 100% of the time). Switched to an
  absolute import.
- `bsfm ui` launched Streamlit with a CWD-relative path to `app.py`, which
  only resolved if invoked from the repo root. Now resolved via the
  installed package location, so it works from any directory.
- Added `tests/unit/test_ui_execution.py`, which actually runs the app and
  all 5 pages via Streamlit's `AppTest` harness. The previously-existing
  "UI" tests only grepped source text for strings like `"DEMO"` and never
  executed anything, which is why the crash above went unnoticed.

**Real-time inference / quantization:**
- Dynamic int8 quantization (`RealtimeInference(..., quantize=True)`)
  failed on every single call due to a PyTorch limitation (dynamically
  quantized `Linear` layers break `TransformerEncoderLayer`'s internal
  fast-path weight inspection) and silently fell back to full precision —
  meaning the "sub-50ms via quantization" capability never actually
  engaged, ever. Fixed via `torch.backends.mha.set_fastpath_enabled(False)`;
  quantization now genuinely activates (verified ~1.26x speedup in
  testing). `RealtimeInference.quantization_active` and
  `benchmark()["quantization_active"]` now expose the real state instead of
  assuming success.

**Deployment:**
- `Dockerfile`: `WORKDIR` was set *after* `COPY`ing the application code in
  the runtime stage, so the code landed outside the working directory.
  This didn't break the API service (which imports the pip-installed
  package from site-packages regardless of CWD) but silently broke the
  `bsfm-ui` compose service, which invokes `streamlit run
  biosignal_fm/ui/app.py` as a CWD-relative path.
- `docker-compose.yml`: `read_only: true` had no writable mount for
  `$HOME/.cache`, so `ModelRegistry()` — instantiated before the server
  even starts — crashed trying to create its storage directory. Same issue
  for Streamlit's `$HOME/.streamlit` on first launch. Added `tmpfs` mounts
  for both.
- The WebSocket streaming endpoint (`/ws/predict/{model_id}`) referenced in
  the README's module table, the Deploy UI page, the FastAPI app
  description, this changelog, and ARCHITECTURE.md **did not exist
  anywhere in the code**. Implemented it: one persistent connection,
  JSON-in/JSON-out per signal window, auth via a query parameter (browsers
  can't set custom WebSocket handshake headers), malformed windows return
  an error without closing the connection. Covered by 4 new tests.
- README/ARCHITECTURE.md referenced a `biosignal_fm.api` module that never
  existed; the real module is `biosignal_fm.deployment`. Docs corrected.
- `configs/exp.yaml`, referenced by the README quickstart and the Pretrain
  UI page, was never actually included in the repo. Added, using the
  library's real default-scale model (512-d, 12 layers) rather than a toy
  config, and verified end-to-end via `bsfm pretrain --config
  configs/exp.yaml --steps 3`.

**CLI:**
- Only `biosignal-fm` was registered as an installable command, but the
  README and every command's help text consistently used `bsfm` (10:1).
  Registered `bsfm` as an alias so documented commands actually work as
  written.

**Tooling / CI (nothing below changes runtime behavior, all pre-existing
gaps that let the bugs above ship unnoticed):**
- `mypy`'s `python_version = "3.10"` crashed immediately against the
  installed numpy's type stubs (which use 3.12+ syntax), meaning `mypy`
  had never once completed a real check. Combined with `|| true` in the CI
  step, this meant mypy provided zero actual type-checking value. Fixed
  the config (bumped to 3.12) and removed `|| true`; mypy now runs for
  real and is clean across all 59 source files (was 70 real errors once
  it could actually run, several of which — a real `assert`-worthy
  `Optional` narrowing gap and a `str` where `FineTuner` requires a
  `Literal["linear","partial","full"]` in `scripts/run_full_study.py` —
  were genuine, if latent, bugs rather than just missing annotations).
- CI's lint/format/type-check steps only ever covered `biosignal_fm` and
  `tests`, never `scripts/` — which is exactly where all 63 real `ruff`
  violations had accumulated despite the README's "ruff clean" claim.
  Extended CI (and the mypy pre-commit hook) to cover `scripts/` too.
- The codebase had never been run through `ruff format` (50 files were
  reformatted; behavior-verified unchanged via the full test suite before
  and after).
- Added a dedicated `pytest --doctest-modules` CI step so confabulated
  docstring examples (see above) get caught automatically going forward.

### Changed
- `mypy`'s `python_version` target: 3.10 → 3.12 (see above; the project's
  actual supported range, `requires-python >= 3.10`, is unchanged).
- FastAPI app `version` and the audit test's version check now read from
  `biosignal_fm.__version__` instead of a separately hardcoded string, so
  they can't drift out of sync again.

### Security
- No new mitigations this release; see docker-compose.yml fix above, which
  restores the previously-documented read-only hardening to a state where
  the API service can actually start.

## [0.1.0] - 2026-08-14

### Added
- Initial public release of BioSignal-FM.
- Unified transformer-based foundation model pretrained via self-supervised
  learning on four biosignal modalities: EMG, ECG, EEG, fNIRS.
- Span-based masked reconstruction + SimCLR-style contrastive SSL head.
- Per-modality Butterworth bandpass filtering with subject-aware normalization.
- LOSO + LODO cross-validation evaluators with subject-aware preprocessing.
- Statistical rigor suite: Friedman + Nemenyi + Wilcoxon with Holm-Šídák
  correction (`1-(1-p)^(m-k+1)` formula, not Bonferroni-Holm), BCa bootstrap
  CIs, Cohen's d, Hedges' g, a-priori power analysis.
- ONNX export with REAL numerical parity verification (max abs diff < atol).
- Dynamic int8 quantized real-time inference (<50ms on Celeron-class CPU).
- FastAPI REST + WebSocket serving with API-key auth and UUID-based model
  registry (no `pickle.load` from client-supplied paths).
- Streamlit 5-page dashboard with WCAG 2.2 AA compliant theme:
  Overview, Pretrain, Finetune, Evaluate, Deploy.
- Typer CLI with Rich console output.
- MLflow + local JSON tracking (with proper numpy-aware JSONEncoder).
- RunManifest with SHA-256 hashes of all outputs, env fingerprint, git HEAD,
  and full config snapshot.
- Dockerfile (multi-stage, non-root USER) and docker-compose.yml with
  security hardening (read-only fs, no-new-privileges, cap_drop ALL).
- Apache 2.0 LICENSE (full 202-line text).
- CITATION.cff, AUTHORS.md, NOTICE, SECURITY.md, CODE_OF_CONDUCT.md,
  CONTRIBUTING.md.
- GitHub Actions CI matrix for Python 3.10 / 3.11 / 3.12.
- Pre-commit hooks: ruff, ruff-format, mypy, end-of-file-fixer,
  trailing-whitespace, check-yaml, check-toml.

### Security
- API-key authentication enforced on all mutating REST endpoints.
- Model registry uses UUIDs and rejects client-supplied file paths.
- Docker image runs as non-root user with read-only filesystem.
- No `pickle.load` on user-supplied paths anywhere in the codebase.
