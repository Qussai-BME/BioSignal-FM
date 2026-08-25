# BioSignal-FM V4 Pre-Publication Release Manifest

## Release identity

| Field | Value |
|---|---|
| Package | `biosignal-fm` |
| Version | `4.0.1` |
| Release type | Publication-ready source release |
| Git commit | Tag `v4.0.1`; run `git rev-parse v4.0.1` for the exact hash. A tag is used instead of a hardcoded hash so this file never again drifts from what is actually shipped (see [What changed since 4.0.0](docs/test_report_v4.md#what-changed-since-400)). |
| Source language | English |
| Project license | Apache License 2.0 |

## Verification record

The final verified source state passed the following measured checks on 25 August 2026,
run against the exact tagged commit (`v4.0.1`) in the same pass that produced the tag:

| Gate | Result |
|---|---|
| Tests | 329 passed; 112 non-blocking warnings; 62.94 s |
| Coverage | 80.42% against a 75% threshold |
| Lint and formatting | Ruff clean; 92 files formatted |
| Static typing | Mypy clean across 68 source files |
| Documentation | Strict MkDocs build passed |
| Core installation | Fresh isolated installation and `verify_core_install.py` passed |
| Distributed extras | Fresh `.[all]` installation passed |
| Dependency posture | `pip check` clean; `pip-audit` reported no known vulnerabilities |
| Git state | Clean commit; live clean/dirty `RunManifest` test passed |

Full evidence is in [`docs/test_report_v4.md`](docs/test_report_v4.md), the risk decisions are in [`docs/prepublication_risk_register_2026.md`](docs/prepublication_risk_register_2026.md), and external sources are recorded in [`docs/references_prepublication_review_2026.md`](docs/references_prepublication_review_2026.md).

## Included material

The archive contains source code, tests, English documentation, GitHub workflow
configuration, Docker/Compose assets, configuration, and this manifest. It
excludes local virtual environments, caches, coverage artifacts, temporary
verification environments, private credentials, and generated build output
(including the `site/` MkDocs build — regenerate it with `mkdocs build`, or
serve docs via a CI-driven GitHub Pages workflow rather than committing the
build). This matches the project's own `.gitignore`; the 4.0.0 archive
shipped several of these excluded artifacts anyway (see the CHANGELOG's
`[4.0.1]` entry), which this release corrects.

## Publication boundary

This is research software. It does not include licensed real biosignal datasets, published real-data benchmarks, clinical validation, regulatory clearance, or a validated foundation-model claim. Public deployment still requires operator-owned TLS, gateway limits, authentication/secret management, data-governance review, and a study-specific protocol where real data is used.
