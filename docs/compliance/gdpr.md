# GDPR Compliance

BioSignal-FM does not collect, process, or store personal data.

## Data Sources

All datasets used by BioSignal-FM are publicly available research datasets:

- **NinaPro DB5/DB7:** Released under research-use license; no PII
- **PhysioNet MIT-BIH / EEGMMID:** Public Domain (ODC PDDL); de-identified
- **Brain-BIDS fNIRS:** Per-dataset license; research-use only

## Data Processing

- **No personal data is collected** by BioSignal-FM itself
- **No PII is stored** in the repository or in checkpoints
- **RunManifests** contain only environment metadata (Python version, package
  versions) — no user-identifiable information

## User Rights

Because BioSignal-FM does not process personal data, GDPR rights (access,
rectification, erasure, portability) do not apply to the software itself.
Users running BioSignal-FM on their own datasets are responsible for
ensuring their data processing complies with GDPR.

## Data Minimization

The package design follows data minimization principles:

- Only biosignal arrays and necessary metadata are loaded
- Cache files contain only processed numerical arrays (no PII)
- No telemetry or usage analytics collected

## International Transfers

The repository is hosted on GitHub (USA). Users in the EU should be aware
that downloading the package from GitHub constitutes an international
transfer of the (non-personal) package code.
