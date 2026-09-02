# Evaluation

Evaluation is protocol-aware rather than a generic metric call. A study must define its dataset version, units of analysis, participant/session split, preprocessing fitting scope, calibration budget, metrics, baselines, random seeds, artifacts, and statistical inference plan before interpreting results.[1]

| Study type | Minimum comparison set | Typical metrics |
|---|---|---|
| Classification | Declared baseline and held-out participant/fold results. | Accuracy, macro-F1, balanced accuracy, and kappa where justified. |
| Regression | Declared baseline and held-out participant/fold results. | MAE, RMSE, R², and task-specific trajectory error. |
| Representation learning | Downstream baseline and pre-defined transfer analysis. | Transfer performance, retrieval/clustering criteria, and justified subject-invariance analysis. |
| Multimodal | Each unimodal path, explicit fusion, and declared modality-ablation path. | Task metric plus participant/fold-level comparison. |

## Statistical unit

For cross-subject work, the participant or protocol-defined fold is the statistical unit. Window-level predictions may be aggregated for a participant-level outcome, but treating correlated windows as independent observations inflates evidence and is not accepted by the platform protocol.[1]

## Current utilities

The evaluation package includes LOSO and LODO helpers, participant-aware aggregation, classification metrics, calibration/confidence tools, and statistical functions. These utilities do not automatically make an experiment scientifically valid; the run manifest and study protocol must state how each utility was used.[2]

## References

[1]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
[2]: https://github.com/qussaiadlbi/biosignal-fm/tree/main/biosignal_fm/evaluation "Evaluation implementation package"
