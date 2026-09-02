# MASTER END-TO-END MULTIMODAL BIOSIGNAL RESEARCH PLATFORM SPECIFICATION

**Project Owner:** Qussai Adlbi
**Nature:** Research-grade multimodal biosignal platform / representation-learning infrastructure / research operating layer
**Status:** Existing V4 architecture is the target baseline; migration/convergence is authorized where applicable
**Specification Role:** SINGLE SOURCE OF TRUTH for architecture, science, engineering, reproducibility, extensibility, release and product boundaries

---

# 0. EXECUTIVE DIRECTIVE

Build **BioSignal-FM** as a research-grade extensible multimodal biosignal platform.

It must provide a unified architecture for:

* EMG;
* EEG;
* ECG;
* experimental ECoG/iEEG.

Longer-term:

* PPG;
* EOG;
* IMU;
* fNIRS;
* other biosignals.

The architecture is:

```text
MODALITY SIGNAL
      ↓
UNIFIED SIGNAL CONTRACT
      ↓
MODALITY ADAPTER
      ↓
MODALITY-SPECIFIC PROCESSING
      ↓
MODALITY REPRESENTATION
      ↓
SHARED REPRESENTATION
      ↓
TASK HEAD
      ↓
CLASSIFICATION / REGRESSION / EMBEDDING
      ↓
MULTIMODAL FUSION
      ↓
ROBUST / MISSING-MODALITY INFERENCE
```

BioSignal-FM is the **representation and multimodal infrastructure layer** of the Human Motor Intelligence research ecosystem.

It is not:

* a collection of unrelated apps;
* an EMG library;
* a generic signal-processing toolbox;
* a claim of a universal foundation model.

At V1/V2 public documentation must call it:

> research platform / representation-learning platform

unless rigorous evidence establishes genuine foundation-model properties.

---

# 1. SCIENTIFIC PURPOSE

BioSignal-FM exists to support this research trajectory:

```text
biosignal processing
→ cross-subject generalisation
→ domain adaptation
→ motor-intent decoding
→ intelligent prostheses
→ EEG / BCI
→ neural interfaces
→ multimodal biosignal learning
→ medical robotics
```

The central architectural hypothesis is:

> heterogeneous biosignals can share reusable infrastructure while preserving modality-specific scientific handling.

The architecture should eventually enable:

```text
EMG ──────┐
          │
EEG ──────┤
          │
ECG ──────┼→ modality encoders
          │          ↓
ECoG ────┘    shared representation
                     ↓
            task-specific heads
                     ↓
          multimodal / robust inference
```

---

# 2. CURRENT MODALITY STATUS

## 2.1 EMG

Priority:

**Primary/current modality**

Must support:

* metadata;
* channel information;
* sampling rate;
* preprocessing;
* feature/representation pipelines;
* task interfaces;
* evaluation;
* provenance.

MyoControl/MyoAdapt may be integrated through adapters.

Do not copy their internals into BioSignal-FM.

---

## 2.2 EEG

Priority:

**Second core modality**

Primary bridge toward:

* BCI;
* neural interfaces;
* motor imagery;
* multimodal motor-intent decoding.

Must support:

* channel metadata;
* montage;
* sampling rate;
* preprocessing;
* epochs;
* artifact handling;
* subject/session identity;
* reproducible evaluation.

---

## 2.3 ECG

Priority:

**Third core modality**

Purpose:

* verify the architecture is genuinely multimodal;
* establish broader physiological signal support.

Do not force EMG assumptions onto ECG.

---

## 2.4 ECoG / iEEG

Status:

**Experimental plugin path**

Do not claim mature support until:

* datasets;
* metadata;
* preprocessing;
* evaluation;
* provenance

are genuinely implemented.

---

# 3. NON-NEGOTIABLE PRINCIPLES

## 3.1 One platform

Never implement:

```text
EMG application
EEG application
ECG application
ECoG application
```

as separate products.

Instead:

```text
BioSignal-FM
    ↓
Unified Signal API
    ↓
Modality Registry
    ↓
Modality-specific components
    ↓
Shared research infrastructure
```

---

## 3.2 Shared contracts, modality-specific science

Shared:

* metadata;
* provenance;
* configuration;
* experiment registry;
* evaluation contracts;
* artifact handling.

Not shared blindly:

* preprocessing;
* filtering;
* artifact removal;
* channel interpretation;
* epoch semantics;
* feature extraction.

Never create:

```python
preprocess(signal)
```

that silently assumes an EMG protocol.

---

# 4. CLEAN-ROOM / MIGRATION RULE

Current V4 architecture is the target.

Historical versions are implementation evidence, not architectural authority.

Migration is preferred over unnecessary rewrite when:

* the existing component is correct;
* its behavior is tested;
* it can be incorporated without architectural compromise.

Do NOT maintain:

```text
old BioSignal-FM
+
BioSignal-FM V4
```

as two parallel systems.

Converge to exactly one canonical platform.

The current migration strategy explicitly requires preserving valuable working capabilities while introducing canonical contracts and removing duplication.

---

# 5. CORE ARCHITECTURE

```text
                         BIOSIGNAL-FM
                              │
                      Unified Signal API
                              │
                      Modality Registry
                              │
        ┌─────────────────────┼──────────────────────┐
        ↓                     ↓                      ↓
       EMG                   EEG                    ECG
        │                     │                      │
 modality adapter       modality adapter       modality adapter
        │                     │                      │
        ↓                     ↓                      ↓
 modality processing    modality processing    modality processing
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              ↓
                    shared representation
                              ↓
           ┌──────────────────┼──────────────────┐
           ↓                  ↓                  ↓
     classification       regression         embedding
                              ↓
                    multimodal fusion
                              ↓
              robust / missing modalities
                              ↓
                    reusable biosignal AI
```

---

# 6. SIGNAL CONTRACT

Every signal object must define conceptually:

* modality;
* data reference;
* shape;
* sampling rate;
* channel names;
* units;
* timestamp information;
* subject ID;
* session ID;
* task;
* preprocessing status;
* dataset ID;
* provenance.

Never rely on filename conventions alone.

---

# 7. METADATA CONTRACT

Metadata must distinguish:

* signal properties;
* acquisition properties;
* biological subject information;
* task information;
* processing history;
* provenance.

Sensitive or restricted metadata must not be exposed accidentally.

---

# 8. CONFIGURATION

No hard-coded:

* paths;
* sample rates;
* channel counts;
* label maps;
* dimensions;
* subject splits.

Use typed validated configurations.

A configuration change affecting scientific behavior creates a new experiment identity.

---

# 9. MODALITY REGISTRY

The registry must support:

```text
register modality
resolve modality
validate modality
list modality
```

Each modality plugin declares:

* capabilities;
* required metadata;
* adapter;
* preprocessing;
* encoder;
* supported tasks;
* validation tests.

---

# 10. DATA ADAPTERS

Adapters must convert external datasets into canonical contracts.

They must NOT modify scientific semantics silently.

For each adapter provide:

* dataset identity;
* source;
* version;
* licensing;
* channel mapping;
* sample rate;
* labels;
* subject/session mapping;
* preprocessing provenance.

---

# 11. PREPROCESSING

Preprocessing is modality-specific.

The framework must separate:

```text
raw data
→ acquisition normalization
→ modality preprocessing
→ model input
```

Every preprocessing step must be:

* explicit;
* configurable;
* versioned;
* reproducible.

No silent preprocessing.

No hidden filtering.

No implicit normalization.

---

# 12. ENCODER INTERFACE

Encoders convert validated modality data into representations.

Encoder interface should conceptually specify:

* input contract;
* output shape;
* modality;
* version;
* configuration;
* training state;
* provenance.

Models are replaceable components.

The architecture is the durable asset.

---

# 13. REPRESENTATION LEARNING

Support:

* supervised representation learning;
* self-supervised approaches;
* transfer;
* embedding extraction.

Do not claim general-purpose foundation-model behavior without evidence.

Evidence would require, at minimum:

* substantial pretraining;
* reusable representation;
* multiple downstream tasks;
* transfer across subjects/tasks/modality where claimed;
* rigorous benchmark comparisons.

---

# 14. TASK HEADS

The framework must support:

### Classification

Examples:

* motor-intent class;
* motor imagery;
* biosignal event classification.

### Regression

Examples:

* continuous control;
* physiological estimation.

### Embedding

Examples:

* subject representations;
* transfer;
* retrieval;
* clustering.

Task heads must depend on stable representations, not raw modality-specific assumptions.

---

# 15. MULTIMODAL FUSION

Architecture must permit:

```text
EMG representation
        +
EEG representation
        ↓
fusion
        ↓
task
```

Fusion strategies should remain replaceable:

* early;
* intermediate;
* late;
* learned gating;
* attention;
* missing-modality robust strategies.

Do not implement all at once.

Start with a scientifically interpretable baseline.

---

# 16. MISSING-MODALITY SUPPORT

A major future capability.

Examples:

```text
EEG + EMG
EEG only
EMG only
```

The model must explicitly know which modalities are present.

Never silently substitute zeros and claim robust multimodal inference without testing it.

---

# 17. EVALUATION

Metrics must depend on task.

Classification:

* accuracy;
* macro-F1;
* balanced accuracy;
* kappa where appropriate;
* per-subject results.

Regression:

* MAE;
* RMSE;
* R²;
* trajectory error.

Representation:

* transfer performance;
* separability where scientifically justified;
* subject invariance analyses.

Multimodal:

* unimodal;
* fusion;
* missing modality;
* subject-level comparisons.

---

# 18. SUBJECT-LEVEL EVALUATION

For cross-subject research:

> Subject/fold is the statistical unit.

Do not perform inference on window count alone.

A valid protocol must explicitly define:

* train subjects;
* validation subjects;
* test subject;
* calibration budget;
* preprocessing scope.

No test-set tuning.

---

# 19. PROVENANCE ENGINE

Every experiment records:

* code commit;
* environment;
* config hash;
* dataset ID;
* subject split;
* preprocessing version;
* random seed;
* model version;
* metrics;
* runtime;
* artifact hashes.

The provenance manifest is part of the scientific result.

---

# 20. EXPERIMENT REGISTRY

Every experiment must be reproducible by identity:

```text
experiment_id
protocol_id
dataset_id
model_id
config_hash
commit
environment
seed
results
artifacts
```

Protocol changes create new protocol versions.

---

# 21. ARTIFACTS

Required outputs may include:

* predictions;
* embeddings;
* checkpoints;
* figures;
* tables;
* reports;
* metrics;
* provenance manifests.

Do not commit protected raw datasets.

---

# 22. RESEARCH INTEGRATION WITH HUMAN MOTOR

BioSignal-FM feeds:

```text
representation
 ↓
generalisation
 ↓
intent decoding
```

then MyoControl / MyoAdapt or compatible decoders may produce:

```text
IntentRecord
```

which can be consumed by MyoSim.

Canonical system:

```text
BioSignal-FM
     ↓
representation
     ↓
decoder / task
     ↓
IntentRecord
     ↓
MyoSim
```

Integration must occur through thin adapters.

---

# 23. CLINICAL / SCIENTIFIC CLAIM BOUNDARIES

Do not claim:

* clinically validated;
* universally generalizable;
* foundation model superiority;
* medical readiness.

unless directly demonstrated.

Correct language:

* research platform;
* experimental;
* benchmarked under protocol X;
* preliminary;
* representation-learning framework.

---

# 24. CURRENT ENGINEERING PHASES

The canonical build sequence is:

## PHASE 0 — GREENFIELD / ARCHITECTURAL BOOTSTRAP

Deliver:

* repository;
* package structure;
* config;
* CI;
* tests;
* docs;
* ADRs;
* dependency/license inventory.

---

## PHASE 1 — CORE CONTRACTS

Implement:

* Signal schema;
* metadata;
* modality registry;
* config;
* error model;
* validation.

Gate:

Core imports and schema tests pass.

---

## PHASE 2 — DATA ADAPTERS

Order:

1. EMG;
2. EEG;
3. ECG;
4. ECoG/iEEG experimental.

Each adapter receives independent validation.

---

## PHASE 3 — PREPROCESSING CONTRACTS

Implement:

* common interfaces;
* modality-specific preprocessing;
* explicit configuration;
* provenance.

---

## PHASE 4 — ENCODERS

Implement minimal working encoders.

Do not prematurely implement enormous model families.

---

## PHASE 5 — TASK HEADS

Implement:

* classification;
* regression;
* embeddings.

---

## PHASE 6 — EVALUATION + PROVENANCE

Implement:

* metrics;
* subject-level evaluation;
* experiment registry;
* artifact tracking;
* reproducibility.

---

## PHASE 7 — MULTIMODAL FUSION

Start with:

* EEG;
* EMG.

Add ECG where useful.

Then:

* missing modality evaluation.

---

## PHASE 8 — DASHBOARD

Only after backend scientific contracts stabilize.

UI must visualize:

* dataset;
* protocol;
* run;
* metrics;
* artifacts;
* provenance.

---

## PHASE 9 — PACKAGING / RELEASE

Deliver:

* package;
* CLI;
* documentation;
* CI;
* environment;
* changelog;
* licensing;
* release dossier.

---

## PHASE 10 — BENCHMARK DEMONSTRATION

Run:

* one EMG experiment;
* one EEG experiment;
* one multimodal or missing-modality demonstration if scientifically ready.

No benchmark claims without real evidence.

---

# 25. VALIDATION MATRIX

Maintain:

| Layer         | Test                           |
| ------------- | ------------------------------ |
| Core          | imports/schema                 |
| Registry      | modality registration          |
| Adapter       | sample data                    |
| Preprocessing | deterministic fixture          |
| Encoder       | shape/contract                 |
| Task          | output validity                |
| Evaluation    | metric correctness             |
| Provenance    | manifest completeness          |
| Fusion        | modality combinations          |
| Release       | clean-environment installation |

---

# 26. REAL-DATA VALIDATION

For each claimed active modality:

* real dataset;
* provenance;
* preprocessing;
* evaluation;
* statistics;
* reproducibility rerun.

Synthetic data is not evidence.

If real data access fails:

* stop that track;
* document source;
* document failure;
* continue independent work.

---

# 27. PERFORMANCE

Measure when relevant:

* ingestion;
* preprocessing;
* training;
* inference;
* memory;
* artifact generation.

Do not optimize before profiling identifies a bottleneck.

---

# 28. OBSERVABILITY

Expose:

* experiment ID;
* modality;
* dataset;
* protocol;
* model;
* status;
* duration;
* errors;
* artifact references.

---

# 29. SECURITY / DATA GOVERNANCE

Enforce:

* no credentials;
* no restricted raw data;
* no secret URLs;
* license records;
* access restrictions;
* sanitized public artifacts.

---

# 30. COMMERCIAL PRODUCT BOUNDARY

BioSignal-FM may become infrastructure behind a future:

> Biosignal Research OS

Potential future capabilities:

* experiment registry;
* benchmark;
* reproducibility;
* team workflow;
* artifact management;
* report generation.

Do NOT add initially:

* SaaS accounts;
* billing;
* cloud;
* payment;
* telemetry;
* multi-tenant infrastructure.

Prove researcher value first.

---

# 31. RELEASE DISCIPLINE

Every public/research release includes:

* semantic version;
* Git tag;
* changelog;
* environment specification;
* provenance;
* benchmark summary;
* known limitations;
* license/third-party notices;
* reproducibility instructions;
* citation.

Never overwrite scientific releases.

---

# 32. DOCUMENTATION

Required:

```text
docs/
├── architecture.md
├── signal-contract.md
├── modality-registry.md
├── preprocessing.md
├── encoders.md
├── tasks.md
├── evaluation.md
├── provenance.md
├── multimodal.md
├── reproducibility.md
├── security.md
└── roadmap.md
```

README must explain:

```text
Signal
 ↓
Modality Adapter
 ↓
Representation
 ↓
Task
 ↓
Fusion
 ↓
Evaluation
 ↓
Artifact
```

---

# 33. DEFINITION OF DONE — V1

BioSignal-FM V1 is complete when:

* core imports succeed;
* tests pass;
* EMG/EEG/ECG contracts work;
* registry works;
* configurations validate;
* CLI works;
* dashboard launches;
* tiny end-to-end runs work;
* provenance works;
* no hard-coded local paths remain;
* restricted datasets are absent;
* reproducibility instructions work;
* CI passes;
* a new modality can be added through documented contracts without editing unrelated core logic.

This definition is consistent with the current master specification.

---

# 34. MIGRATION DEFINITION OF DONE

Migration is complete when:

* there is one canonical BioSignal-FM;
* duplicated legacy architecture is removed or isolated;
* existing valuable capabilities are preserved where valid;
* canonical contracts are used;
* scientific overclaims are removed;
* tests pass;
* real-data smoke tests pass;
* public documentation describes the canonical architecture;
* a clean environment can install and run the platform.

---

# 35. PHASE GATE

Every phase must pass:

### Research lens

* scientific validity;
* correct protocol;
* no leakage;
* meaningful evaluation.

### Engineering lens

* architecture;
* interfaces;
* tests;
* maintainability.

### Product lens

* understandable workflow;
* useful CLI/UI;
* no needless complexity.

### IP/release lens

* license compatibility;
* clean dependencies;
* reproducible releases;
* no proprietary data.

---

# 36. AGENT EXECUTION METHOD

Never dump the entire implementation into one uncontrolled change.

Use:

```text
AUDIT
 ↓
IMPLEMENT PHASE
 ↓
TEST
 ↓
END-TO-END SMOKE
 ↓
REVIEW
 ↓
DOCUMENT
 ↓
PHASE REPORT
 ↓
NEXT PHASE
```

Never implement future phases early.

Do not rewrite working components just for stylistic reasons.

If ambiguity exists:

1. preserve documented architecture;
2. choose simplest scientifically defensible option;
3. prefer reversible choices;
4. document the decision;
5. ask the user only if a true blocker exists.

---

# 37. FINAL REPORT

The final agent report must contain:

* architecture;
* migration status;
* modality matrix;
* tests;
* real-data validation;
* provenance;
* benchmark results;
* reproducibility status;
* performance;
* known limitations;
* security/licensing;
* release readiness;
* future roadmap;
* unresolved issues.

# END OF SPECIFICATION
