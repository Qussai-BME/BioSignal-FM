# Real-Data Source Register

This register selects real-data sources for the core modality tracks. It records source and license facts before acquisition and prohibits committing raw data, participant identifiers, or credentialed access materials. Selection establishes an adapter-validation path only; it does not create a benchmark result.[1]

| Modality | Dataset ID | Source and version | License | Initial validation scope | Governance decision |
|---|---|---|---|---|---|
| EMG | `zenodo.1000116` | NinaPro DB5, Zenodo v1 | CC BY-ND 4.0 | One or more subject archives for adapter/parser validation; participant-held-out movement protocol after label review. | Approved for controlled retrieval with attribution and integrity checks; raw MATLAB data remains untracked. |
| EEG | `physionet.eegmmidb.1.0.0` | EEG Motor Movement/Imagery Dataset v1.0.0 | ODC-By v1.0 | Selected EDF+ recordings and annotation files for adapter, montage, and epoch validation. | Approved for controlled retrieval with attribution; event/run selection must be declared before evaluation. |
| ECG | `physionet.mitdb.1.0.0` | MIT-BIH Arrhythmia Database v1.0.0 | ODC-By v1.0 | Selected waveform/header/annotation records for adapter and record-held-out protocol validation. | Approved for controlled retrieval with attribution; raw waveform/annotation files remain untracked. |

## Required record fields

Every source-specific manifest must include the dataset ID, source URI, immutable dataset version, license identifier, adapter and adapter version, retrieved file identity/hash, channel mapping, source sample rate, label mapping, participant or record mapping, and preprocessing history. A study run must record the precise source subset rather than only the dataset family.[1]

## Retrieval rules

The validation workflow downloads the minimum data necessary to test adapters and protocols. It stores data under an ignored local directory, checks available file hashes when supplied by the source, and generates only sanitized artifacts for the repository. If a source becomes unavailable or access conditions cannot be met, that modality track must be marked blocked and cannot be replaced by synthetic data for a scientific claim.[1]

## References

[1]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
[2]: data_source_notes_emg.md "NinaPro DB5 source notes"
[3]: data_source_notes_eeg.md "PhysioNet EEG motor movement/imagery source notes"
[4]: data_source_notes_ecg.md "MIT-BIH Arrhythmia source notes"
