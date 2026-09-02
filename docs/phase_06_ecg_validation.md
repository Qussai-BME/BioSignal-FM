# Phase 6 — Real ECG Adapter and Protocol Validation

This phase validated the MIT-BIH Arrhythmia loader against two official PhysioNet WFDB records, `100` and `101`, kept only in an isolated local validation workspace. The selected source is MIT-BIH Arrhythmia Database version 1.0.0 under the Open Data Commons Attribution License v1.0.[1]

| Evidence item | Result |
|---|---|
| Source records | Complete header, waveform, and reference annotation files for records `100` and `101`; local SHA-256 identities are retained in the external run manifest. |
| Real data structure | Record `100` contained two leads, MLII and V5, at 360 Hz with 650,000 samples and 2,274 reference annotations. Record `101` contained MLII and V1 at the same sampling rate with 1,874 annotations. |
| Loader correction | The loader retains actual WFDB record identifiers as held-out units, discovers nested records, validates the source sampling rate, and retains dynamic source lead names rather than falsely declaring every record as MLII/V1. |
| Annotation handling | Windows are centered on the source reference annotation samples. Source beat symbols are mapped to the declared AAMI-style grouping, with source symbol and peak location retained in metadata. |
| Canonical conversion | Dataset ID, version, license, source URI, per-file SHA-256 values, record name, lead names, units, and beat annotation metadata survive canonical ECG adaptation. |
| Real integration test | The opt-in MIT-BIH loader/adaptor test passed against record `100`. |
| Protocol smoke | A deterministic morphology-summary nearest-centroid path trained on record `100` and held out record `101`. It used 2,269 training and 1,861 test beat-centered windows from the two shared observed classes, accuracy 0.1274, and macro-F1 0.1137. |

## Interpretation boundary

This is a **two-record adapter and protocol smoke test**. It is not an arrhythmia benchmark, clinical-performance result, model comparison, or population inference. It verifies WFDB parsing, annotation alignment, record-level partitioning, training-only feature normalization, complete manifest formation, and sanitized artifact generation. Beat windows are correlated observations; their count must not be treated as a statistical sample size.[2]

The poor observed metrics are preserved as evidence of the exercise’s purpose: no optimistic claim is justified. A future benchmark must define its annotation inclusion/exclusion policy, patient/record split semantics, class-imbalance method fitted only on training records, preprocessing protocol, participant/record-level uncertainty analysis, and comparison baselines before results can be interpreted.

## Reproduction command

Raw WFDB files remain outside the repository. A user with the governed source records can reproduce the smoke path:

```bash
python3 scripts/run_mitbih_protocol_smoke.py \
  /path/to/mitbih-root \
  /path/to/external/output
```

The runner produces sanitized metrics, aggregate prediction counts, and a complete `RunManifest` with deterministic experiment identity and output hashes.

## References

[1]: data_source_notes_ecg.md "MIT-BIH source and license notes"
[2]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
