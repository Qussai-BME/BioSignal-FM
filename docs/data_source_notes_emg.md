# EMG Data Source Notes

## NinaPro DB5

The official NinaPro DB5 page states that the dataset contains sEMG and kinematic data from 10 intact participants, includes 52 hand movements plus rest, and records six repetitions. The data were acquired using two Thalmic Myo armbands at 200 Hz. Each subject/exercise is stored as a MATLAB file with synchronized variables including 16 EMG channels, accelerometer, glove, stimulus/restimulus, repetition labels, and participant descriptors.[1]

The Zenodo dataset record is published as version v1 on 1 October 2017 under the Creative Commons Attribution–NoDerivatives 4.0 International license. It exposes separate subject archives such as `s1.zip` and provides MD5 values, permitting an integrity check after retrieval.[2]

| Governance field | Selected value |
|---|---|
| Dataset identifier | `zenodo.1000116` |
| Modality path | EMG, with optional kinematic and accelerometer fields not used by the initial EMG study |
| Dataset version | v1 |
| License | CC BY-ND 4.0 |
| Initial protocol candidate | Participant-held-out EMG movement classification using a restricted, documented label subset and subject/fold-level aggregation |
| Data minimization | Retrieve only the subject archives required for an adapter smoke test and protocol validation; do not commit raw data |

## References

[1]: https://ninapro.hevs.ch/instructions/DB5.html "Official NinaPro DB5 instructions"
[2]: https://zenodo.org/records/1000116 "NinaPro dataset 5 Zenodo record"

## Retrieval record

On 26 August 2026, the public `s1.zip` archive was initiated through the official Zenodo record for minimal adapter validation. The expected source MD5 is `bb63c22179b4a750bac2499aecd72021`. The archive is not a repository artifact; it is retained only in the local validation workspace and must be verified before use.

A second public subject archive, `s2.zip`, was initiated for a minimal held-out-participant protocol check. Its published MD5 is `9d91c5a524123e950cd7cab3b42887ae`.

Both minimally retrieved archives passed the publisher-provided MD5 check: `s1.zip` matched `bb63c22179b4a750bac2499aecd72021` and `s2.zip` matched `9d91c5a524123e950cd7cab3b42887ae`. Each archive contained the three expected exercise MATLAB files for its participant.
