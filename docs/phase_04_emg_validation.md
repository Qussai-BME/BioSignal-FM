# Phase 4 — Real EMG Adapter and Protocol Validation

This phase validated the corrected NinaPro DB5 EMG loader against real, locally retrieved data. The selected source is the official Zenodo version v1 record for NinaPro DB5, licensed under CC BY-ND 4.0. Two subject archives were retrieved outside the repository, checked against the publisher-provided MD5 values, and unpacked into an isolated local validation workspace.[1]

| Evidence item | Result |
|---|---|
| Source archives | `s1.zip` and `s2.zip`, each verified against the publisher-provided MD5 value. |
| Real data structure | Each archive contained three MATLAB exercise files; inspection of `S1_E1_A1.mat` confirmed 16 channels, 200 Hz sampling, source `stimulus`/`restimulus` labels, and six repetitions. |
| Loader correction | The adapter now discovers nested subject directories, preserves the source-corrected `restimulus` labels, records source file SHA-256, uses 200 Hz rather than the former incorrect 2 kHz declaration, retains all exercises per participant, and requires explicit synthetic fallback. |
| Canonical conversion | The EMG modality adapter preserves dataset identifier, version, license, source URI, source file hash, and real-data origin in `SignalProvenance`. |
| Real integration test | The opt-in NinaPro loader/adaptor test passed with one real participant archive. |
| Protocol smoke | A deterministic RMS-nearest-centroid path trained on participant 1 and held out participant 2. It produced 533 training windows, 536 held-out windows, 23 observed classes, accuracy 0.0933, and macro-F1 0.0711. |

## Interpretation boundary

The two-participant protocol smoke result is **not a benchmark, model-comparison result, population inference, clinical result, or foundation-model claim**. It demonstrates that the real-data path preserves source semantics, keeps train and held-out participants separate, constructs features using training-subject normalization only, writes sanitized outputs, and produces a complete reproducibility manifest. Window counts are operational counts, not independent inferential samples.[2]

The low observed metric is retained precisely to prevent optimistic presentation. It should not be optimized without first locking a scientifically appropriate NinaPro label subset, acquisition/exercise scope, preprocessing protocol, and full participant-level evaluation plan.

## Reproduction command

The external validation workspace is intentionally excluded from the repository. A user who has independently retrieved the governed source files can execute the following command after setting an appropriate local path:

```bash
python3 scripts/run_ninapro_protocol_smoke.py \
  /path/to/extracted/ninapro-db5 \
  /path/to/external/output
```

The runner writes a sanitized metrics file, prediction-count summary, and a `RunManifest`. The manifest includes a deterministic experiment ID, dataset/license evidence, protocol definition, code/environment context, metrics, and output hashes.

## References

[1]: data_source_notes_emg.md "NinaPro DB5 source and retrieval notes"
[2]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
