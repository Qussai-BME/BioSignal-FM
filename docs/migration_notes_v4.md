# Migration Notes: BioSignal-FM v3.3 to V4

## Summary

V4 is a controlled architectural migration inside the same repository. It does not create a parallel V4 tree or merge unrelated research platforms. Valid components remain available; canonical contracts, the modality registry, and provenance are the approved boundaries for new capabilities.

| Area | v3.3 | V4 |
|---|---|---|
| Signal | Loader-context `BiosignalSample` | Reader-independent `Signal`, `SignalMetadata`, and `SignalProvenance`. |
| Modalities | Scattered enums and application branches | Explicit tested `ModalityRegistry`. |
| ECoG/iEEG | Not represented | Explicit experimental adapter path; no benchmark claim. |
| fNIRS | Part of a single modality list | Optional legacy-compatible extension. |
| Demo data | Free-form metadata label | Structured provenance, benchmark-ineligible flag, and CLI disclosure. |
| Fusion | No general contract | Representation, optional fusion, then task head. |
| Dependencies | UI/API/scientific packages in base installation | Capability-specific extras. |
| Product language | Unified foundation-model framing | Research-platform framing with evidence boundaries. |

## Preserved compatibility

| Interface or component | V4 status | Migration path |
|---|---|---|
| `BiosignalSample` | Supported | Convert through `default_registry().get(modality).adapter_factory().to_signal(sample)`. |
| `BiosignalDataset` | Supported | Keep the loader; add/use a modality adapter at the V4 boundary. |
| `NinaProDB5Loader` | Supported | Core EMG path. |
| `MITBIHLoader` | Supported | Core ECG path; WFDB remains optional. |
| `EEGMMIDLoader` | Supported | Core EEG path; MNE remains optional. |
| `FnirsLoader` | Supported with de-emphasis | Use only when the optional extension is needed. |
| `PreprocessingPipeline` | Supported | Build through the modality factory or directly for compatibility. |
| `FoundationModel` / `DistilledFoundationModel` | Supported legacy names | Treat as encoder/representation implementations, not evidence of a validated foundation model. |
| `bsfm` and `biosignal-fm` | Supported | `bsfm inspect` is available and synthetic pretraining is explicit. |

## Move to the canonical contract

Move the boundary between a source reader and preprocessing into a modality adapter:

```python
from biosignal_fm.modalities import default_registry

registry = default_registry()
loader = ...  # an existing dataset loader
legacy_sample = loader[0]
plugin = registry.get(legacy_sample.modality.value)
signal = plugin.adapter_factory().to_signal(legacy_sample)
```

New services should accept `Signal` or `SignalBatch` after this point rather than a reader- or dataset-specific object.

## Modality indexes and checkpoints

`Modality.ECOG` was inserted after `EEG` and before `FNIRS`. Do not use fixed numeric modality indexes. Use `Modality.from_str(name)` or the registry:

```python
from biosignal_fm.config import Modality

modality_index = list(Modality).index(Modality.from_str("fnirs"))
```

V4 derives default modality counts from the current enum. Preserve an old checkpoint's original configuration and scientific scope; adding ECoG does not make an old checkpoint capable of representing ECoG automatically.

## Installation

V4 installs capabilities selectively. Contract/configuration users do not need Streamlit, FastAPI, MNE, WFDB, or PyTorch.

| Previous choice | Preferred V4 path |
|---|---|
| `.[fm]` | Retained temporarily; prefer `.[ml,deployment]` as needed. External MLflow tracking is deferred pending a secure compatible upstream release. |
| `.[data]` | Retained temporarily; prefer `.[data-eeg]`, `.[data-ecg]`, or `.[data-fnirs]`. |
| Everything | `.[all]` in an isolated development environment. |
| Minimal research core | `biosignal-fm`, followed by selected extras. |

## Intentional CLI change

`bsfm pretrain` no longer creates synthetic data silently. The technical demo path requires explicit opt-in:

```bash
bsfm pretrain --config configs/exp.yaml --steps 1 --synthetic-demo
```

Omitting the flag produces guidance because V4 must not make a synthetic demonstration appear to be real-data training or a benchmark. The demo `finetune` and `evaluate` commands emit explicit data-origin warnings.

## Removed or deprecated positions

| Item | Status | Reason |
|---|---|---|
| Validated-foundation-model product framing | Removed from public positioning | An architectural name alone is insufficient scientific evidence. |
| State-of-the-art or numeric claims without provenance/protocol | Removed from public documentation | The migration package does not include licensed benchmark data and reproducible real-data results. |
| fNIRS as a V4-core requirement | De-emphasized | It is outside the current core-modality scope. |
| Mandatory heavy dependencies | Removed from core | Enables clean smaller installations and boundary isolation. |
| Audit scanning arbitrary local environments | Removed | Audit scope is project source and tests only. |

## Do not change automatically

Do not change LOSO/LODO definitions, fitting scope, feature selection, target-participant exposure, or metrics as part of migration. Put study-specific protocols in dedicated configuration/artifact directories and record them in `RunManifest`. Do not label an output a real benchmark unless provenance is `real` and dataset, license, version, and protocol information are available.

## Migrator checklist

- [ ] Install only required extras.
- [ ] Move the data entry point to a registered adapter.
- [ ] Inspect `Signal.metadata.provenance` before training or evaluation.
- [ ] Replace hard-coded indexes with a `Modality` or registry-derived index.
- [ ] Add study protocol and dataset provenance to `RunManifest`.
- [ ] Use `--synthetic-demo` only for smoke paths.
- [ ] Preserve old checkpoints and their configurations when reproducing earlier work.
