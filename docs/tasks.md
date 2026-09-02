# Tasks

Task heads consume a stable `Representation` rather than raw modality-specific arrays. This keeps modality science in adapters, preprocessing, and encoders while allowing tasks to be changed without rewriting the platform boundary.[1]

| Task family | Intended outputs | Required evaluation framing |
|---|---|---|
| Classification | Motor-intent class, motor-imagery class, or biosignal event class. | Accuracy, macro-F1, balanced accuracy, and participant/fold results as appropriate. |
| Regression | Continuous control, physiological estimation, or trajectory values. | MAE, RMSE, R², trajectory error, and subject-level comparisons. |
| Embedding | Subject representation, retrieval vector, clustering/transfer feature. | Downstream transfer performance and scientifically justified separability or invariance analyses. |

`TaskHead` is a protocol with a single `predict(representation)` operation. A pipeline must route multiple modality representations through an explicit fusion strategy before invoking a task head; late implicit task-head fusion is rejected.[2]

## Research constraints

A task definition must state its labels or targets, units of analysis, split protocol, calibration budget, preprocessing fitting scope, primary metric, baseline, and statistical analysis plan. The number of signal windows must not be substituted for the number of independent subjects or protocol-defined experimental units.[1]

## References

[1]: master_specification.md "MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION"
[2]: https://github.com/qussaiadlbi/biosignal-fm/blob/main/biosignal_fm/services/research.py "TaskHead and ResearchPipeline"
