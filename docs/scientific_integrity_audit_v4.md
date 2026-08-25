# Scientific Integrity Audit — BioSignal-FM V4

## Audit decision

**Decision:** V4 is suitable for release as a testable research-software platform. The release does not bundle licensed real benchmark data or reproducible real-data experiments that establish a foundation-model claim, multimodal transfer, superiority over baselines, or clinical utility.

> V4 documentation, CLI text, and UI messaging therefore use the term **research platform**. A foundation-model outcome remains an evidence-dependent research direction, not an established fact.

## Risk register and controls

| Risk | V4 control | Status |
|---|---|---|
| Synthetic fallback presented as real data | `DataOrigin`, `SignalProvenance`, synthetic result labels, explicit CLI/UI disclosure, and contract tests. | Implemented. |
| Architectural class name interpreted as scientific evidence | Product language and legacy compatibility names are documented as implementation terminology only. | Implemented. |
| ECoG generalization without a selected dataset | Experimental adapter and registry entry only. | Implemented; no benchmark claim. |
| Window count treated as independent participant count | Protocol guidance requires participant/experimental-unit aggregation where appropriate. | Study-specific enforcement remains required. |
| Target-participant leakage in LOSO | Preserved split utilities and manifest protocol recording. | Study-specific protocol verification remains required. |
| Hidden normalization/feature-selection scope | Configuration hashes and protocol recording support explicit study transfer. | Each study must encode its own fitting scope. |
| Clinical or regulatory implication | Explicit research-only and non-clinical boundaries. | Implemented in public release material. |

## Data-origin policy

Every signal entering a V4 service must carry `SignalProvenance`. Origin is a contract property, not a UI hint.

| `origin` value | Meaning | Permitted description of output |
|---|---|---|
| `real` | A documented real source, preferably with dataset name, version, license, and adapter path. | A real experiment output only when protocol and analysis are also documented. |
| `synthetic` | Local generator, fallback path, or demonstration. | Smoke test, technical demonstration, or engineering verification only. |
| `unknown` | Source is insufficiently documented. | Not a benchmark or scientific evidence until documentation is completed. |

The V4 synthetic generator sets `benchmark_eligible: false`. Legacy adapters preserve that field, generator information, fallback reason, and the `synthetic://biosignal-fm` origin rather than silently upgrading it.

## Minimum evidence for a real-data result

Before labeling an output a real benchmark or scientific inference, preserve at least the following:

1. A real dataset identified by name, version, and license.
2. Locked preprocessing configuration and its recorded hash.
3. An explicit statistical unit, especially for LOSO or LODO.
4. Reviewable separation of training, validation, and test data with no target-participant leakage.
5. Metrics and baselines selected before inspecting results.
6. Seed, Git state, dependencies, runtime context, and protocol in `RunManifest`.
7. Re-runnable outputs and hashes for material artifacts.

## Claim language

| Topic | Appropriate language | Unsupported without additional evidence |
|---|---|---|
| Product | “Modular multimodal biosignal research platform” | “Validated unified foundation model” |
| Code models | “Trainable encoder or representation model” | “General biosignal foundation model” |
| Synthetic data | “Labeled technical demonstration or smoke test” | “Benchmark result” or “clinical validation” |
| ECoG | “Experimental adapter path” | “Benchmarked ECoG support” |
| Performance | “Local measurement with documented context” | “State of the art” or “best performance” |
| Regulation | “Informational regulatory mapping” | “Compliant”, “cleared”, or “certified” |

## Remaining boundary

A migration cannot substitute for real data, peer review, or a study-specific protocol. V4 does not download datasets, grant licenses, or establish permission to redistribute an external source. Passing synthetic tests does not establish performance for EMG, EEG, ECG, or ECoG.

## Release position

This audit permits V4 to be released as an **architectural and engineering release**. Any future scientific, benchmark, or clinical statement remains conditional on real data, a documented protocol, reproducible execution, and appropriate independent review.
