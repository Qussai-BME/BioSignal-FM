# EEG Data Source Notes

## PhysioNet EEG Motor Movement/Imagery Dataset

The official PhysioNet record identifies EEG Motor Movement/Imagery Dataset version 1.0.0 and the Open Data Commons Attribution License v1.0. It contains 64-channel EEG sampled at 160 Hz, stored as EDF+ with annotation channels and associated event files. The record describes 14 experimental runs per participant: two baseline runs and 12 motor/motor-imagery runs representing three repetitions of four task conditions. The participant directories extend through at least S108 in the record listing.[1]

An OpenNeuro BIDS-hosted version was examined as a possible structured alternative, but it did not render usable information in the current environment. The official PhysioNet source remains the selected governance and adapter source for this phase.

| Governance field | Selected value |
|---|---|
| Dataset identifier | `physionet.eegmmidb.1.0.0` |
| Modality path | EEG motor movement and motor imagery |
| Dataset version | 1.0.0 |
| License | Open Data Commons Attribution License v1.0 |
| Initial protocol candidate | Held-out-participant motor-imagery classification using clearly declared runs, event labels, epoching, and montage policy |
| Data minimization | Retrieve only the recordings required for adapter and protocol validation; do not commit EDF files or participant data |

## References

[1]: https://www.physionet.org/content/eegmmidb/1.0.0/ "Official PhysioNet EEG Motor Movement/Imagery Dataset record"
