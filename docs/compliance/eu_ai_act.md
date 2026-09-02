# EU AI Act Research-Use Position

**Status:** Informational research-software note; not a legal classification, conformity assessment, or legal advice.  
**Release scope:** BioSignal-FM `4.0.2` research platform.

> BioSignal-FM is released for research and education. It is not presented as a medical device, a clinical decision-support product, a validated foundation model, or a deployed high-risk AI system. This repository does not assign itself an EU AI Act risk category.

## Why a repository cannot self-classify every deployment

Any legal assessment depends on the concrete intended purpose, functionality, deployment context, actors, data, geographic scope, and applicable product-law framework. A research repository can be reused in a way that changes these facts. Therefore, statements such as “limited risk,” “high risk,” “Annex IV compliant,” or “ready for conformity assessment” are not made for this release.

## What the release does provide

| Engineering element | Current, verifiable purpose | What it does not prove |
|---|---|---|
| Run and artifact manifests | Reconstructing a declared research execution and its inputs/outputs by reference. | Legal record-keeping sufficiency for a regulated deployment. |
| Provenance and dataset policy | Preserving source, version, license, processing, and evidence context. | Lawful access, GDPR compliance, or authorized clinical use. |
| Tests, CI, static checks, and release files | Improving software quality and reproducibility. | Medical-device safety, clinical performance, cybersecurity certification, or regulatory conformity. |
| Research-only labeling and limitations | Preventing accidental overstatement of present evidence. | Prevention of all downstream misuse. |

## Conditions before considering a regulated or clinical pathway

A separate product program would need a fixed intended use, accountable legal manufacturer/deployer roles, quality and risk-management processes, data-governance and cybersecurity plans, human-factors work where relevant, validated performance for the intended population/use environment, and qualified regulatory/legal review. Those activities are outside this repository's current evidence scope.

## Research-use boundary

Do not use BioSignal-FM outputs for diagnosis, treatment, monitoring, patient management, or decisions about an individual. Do not describe this software, its models, its benchmarks, or its documentation as EU AI Act compliant, certified, limited-risk, high-risk, medically validated, or regulatory-ready without a separate evidence package and qualified review.

See [`docs/data_governance.md`](../data_governance.md) for data handling and [`RELEASE_MANIFEST.md`](../../RELEASE_MANIFEST.md) for the scientific claim boundary.
