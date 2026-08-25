# Security Policy

## Supported versions

Security fixes are applied to the latest supported V4 release line on the `main` branch.

| Version | Supported |
|---|---|
| 4.0.x | Yes |
| Earlier releases | No |

## Reporting a vulnerability

Please report a suspected vulnerability privately. Do not open a public issue before a fix or coordinated disclosure is agreed.

1. Email **qussai.adlbi@proton.me** with a description, affected version, reproduction steps, and impact.
2. Include a minimal proof of concept where safe and lawful to do so.
3. The maintainer will acknowledge receipt within 72 hours when possible.
4. Target handling is 30 days for high-severity issues, 90 days for medium-severity issues, and best effort for lower-severity findings. Timelines may change if a coordinated disclosure or upstream dependency fix is required.
5. Credit is provided in release notes unless anonymity is requested.

## Security model and boundaries

BioSignal-FM is research software. It is not a multi-tenant service, a clinical device, or a complete network-security boundary. A public deployment requires operator controls outside this repository, including TLS, reverse-proxy limits, monitoring, secret management, and a data-governance review.

### Authentication and authorization

- Mutating REST endpoints and `POST /predict` require an API key supplied in the `X-API-Key` header.
- API keys use `secrets.compare_digest` for constant-time comparison.
- WebSocket sessions authenticate with the first JSON message, `{"api_key": "..."}`. Credentials are not accepted in URL query strings.
- Read endpoints are unauthenticated by design for local observability. Do not expose them publicly if model metadata is sensitive.
- The built-in registry is single-tenant. Use an external gateway and tenant isolation for multi-tenant systems.

### Input and artifact controls

- REST and WebSocket inference validate a finite two-dimensional signal with the exact `(channels, samples)` shape registered for the model.
- Checkpoint registration accepts only a relative path within the operator-configured model directory. Absolute paths and traversal components are rejected.
- Checkpoints are loaded with `torch.load(weights_only=True)`, reducing unsafe deserialization risk. This does not make arbitrary artifacts trustworthy; operators must stage only expected artifacts in the model directory.
- Internal exception details are logged server-side and are not returned to clients in a generic 500 response.

### Filesystem and container controls

- File operations use `pathlib.Path` and the registry resolves paths under its configured directory.
- The Docker image runs as non-root. Compose enables a read-only root filesystem, `no-new-privileges`, and dropped Linux capabilities.
- Compose requires `BSFM_API_KEY` and binds the API to host loopback by default. A public service should be placed behind a hardened reverse proxy.

### Supply chain controls

- CI runs linting, formatting, typing, doctests, tests, and dependency vulnerability auditing.
- GitHub Actions are pinned to reviewed full commit SHAs in the repository workflow.
- Dependabot is configured for Python dependencies and Actions updates.
- Release uploads should use PyPI Trusted Publishing where supported instead of storing a long-lived upload token in CI.[1]

### Data handling

Do not commit, upload, mount into public services, or transmit credentialed health data through examples or demos. Dataset access, licensing, provenance, and permitted model release must be reviewed separately. See [Data Governance](docs/data_governance.md).

## Known security considerations

- There is no built-in rate limiter, request-body size limiter, TLS termination, or tenant isolation. Enforce these at a reverse proxy, API gateway, or ASGI-server boundary.
- The API key is a single shared secret. Use a managed identity system if per-user authorization, audit trails, or revocation are required.
- Synthetic examples are software smoke paths only; they do not validate clinical safety, privacy, or scientific performance.

## Reference

[1]: https://pypi.org/help/#trusted-publishers "PyPI Trusted Publishers"
