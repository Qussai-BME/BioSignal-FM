# ECG Data Source Notes

## MIT-BIH Arrhythmia Database

The official PhysioNet record identifies MIT-BIH Arrhythmia Database version 1.0.0. It contains 48 half-hour excerpts of two-channel ambulatory ECG from 47 subjects, sampled at 360 Hz with 11-bit resolution over a 10 mV range. The record includes approximately 110,000 annotations and is published under the Open Data Commons Attribution License v1.0.[1]

The ODC-By v1.0 license permits sharing, modification, and use subject to attribution conditions. Public conveyance of the database or a derivative database requires retention of relevant notices and inclusion of the license or its URI in the database and relevant documentation.[2]

| Governance field | Selected value |
|---|---|
| Dataset identifier | `physionet.mitdb.1.0.0` |
| Modality path | Two-channel ambulatory ECG with reference beat annotations |
| Dataset version | 1.0.0 |
| License | Open Data Commons Attribution License v1.0 |
| Initial protocol candidate | Record-held-out beat/rhythm analysis, using strictly record-level splits and clearly declared annotation mapping |
| Data minimization | Retrieve only records required for adapter and preprocessing validation; do not commit waveform or annotation files |

## References

[1]: https://www.physionet.org/content/mitdb/1.0.0/ "Official MIT-BIH Arrhythmia Database record"
[2]: https://www.physionet.org/content/mitdb/view-license/1.0.0/ "Open Data Commons Attribution License v1.0"

## Retrieval record

On 26 August 2026, the official WFDB header and waveform files for record `100` were requested from the PhysioNet MIT-BIH version 1.0.0 record for minimal adapter validation. The required annotation file is retrieved separately, and local SHA-256 identities will be recorded in the sanitized run manifest. Raw waveform and annotation files remain outside the repository.
