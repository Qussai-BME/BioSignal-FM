# Phase 5 — Real EEG Adapter and Protocol Validation

This phase validated the PhysioNet EEG Motor Movement/Imagery (EEGMMID) loader against two official EDF+ recordings from run 4, stored only in an isolated local validation workspace. The selected source is EEGMMID version 1.0.0 under the Open Data Commons Attribution License v1.0.[1]

| Evidence item | Result |
|---|---|
| Source recordings | `S001R04.edf` and `S002R04.edf`, independently downloaded from the official PhysioNet record and identified by local SHA-256 in the external run manifest. |
| Real data structure | `S001R04.edf` contained 64 channels, 160 Hz sampling, 20,000 samples, and 30 EDF+ annotations. |
| Annotation semantics | Run 4 was deliberately restricted to its task-specific meaning: `T1` and `T2` are used as left- and right-fist motor-imagery events. `T0` is excluded from the initial binary protocol. |
| Loader correction | The loader now derives fixed windows from EDF+ annotations rather than selecting arbitrary recording centers or assigning approximate run labels. It restricts the initial path to runs 4, 8, and 12, where the selected binary semantics are defined. |
| Canonical conversion | Dataset identity, version, license, source URI, source hash, run identifier, annotation, and normalized source channel names are preserved through the EEG modality adapter. |
| Real integration test | The opt-in EEGMMID loader/adaptor test passed against `S001R04.edf`. |
| Protocol smoke | A deterministic 8–30 Hz log-power nearest-centroid path trained on participant 1 and held out participant 2. It produced 15 training and 15 held-out event windows, accuracy 0.5333, and macro-F1 0.3478. |

## Interpretation boundary

The result is a **two-participant adapter and protocol smoke test**, not a benchmark, model comparison, population inference, clinical result, or foundation-model result. It confirms source-event parsing, participant separation, train-only feature normalization, manifest completeness, and sanitized artifact generation. Event-window counts are operational counts, not independent statistical units.[2]

The metric is retained without optimization. Any subsequent benchmark must pre-register a participant-level split policy across an appropriately sized cohort, decide on montage/channel normalization and artifact handling, lock preprocessing fitting scope, retain all protocol-relevant runs, and report subject/fold-level uncertainty.

## Reproduction command

Raw EDF files are intentionally external to the repository. An independently authorized user can reproduce the smoke path after obtaining governed source recordings:

```bash
python3 scripts/run_eegmmid_protocol_smoke.py \
  /path/to/eegmmid-root \
  /path/to/external/output
```

The runner writes sanitized metrics, prediction-count summary, and a complete `RunManifest` with deterministic experiment identity and output hashes.

## References

[1]: data_source_notes_eeg.md "PhysioNet EEGMMID source notes"
[2]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
