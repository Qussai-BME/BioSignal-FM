# Multi-stage Dockerfile for BioSignal-FM
# Builds a small, secure, non-root image for serving the FastAPI app.

# ---------- Stage 1: Builder ----------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency manifest first (better layer caching)
COPY pyproject.toml README.md LICENSE ./
COPY biosignal_fm ./biosignal_fm

# Install into a prefix we can copy later
RUN pip install --prefix=/install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --prefix=/install --no-cache-dir ".[fm,api,ui]"

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/bsfm/.local/bin:${PATH}" \
    HOME=/home/bsfm

LABEL org.opencontainers.image.title="BioSignal-FM" \
      org.opencontainers.image.description="Research platform for reproducible multimodal biosignal workflows" \
      org.opencontainers.image.authors="Qussai Adlbi <qussai.adlbi@proton.me>" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/qussaiadlbi/biosignal-fm"

# Create non-root user (security hardening)
RUN groupadd -r bsfm && useradd -r -g bsfm -m -d /home/bsfm -s /sbin/nologin bsfm

WORKDIR /home/bsfm

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code (relative COPY destinations below resolve against
# WORKDIR, so this must come after WORKDIR is set — otherwise application
# code lands outside the directory anything running here will look in)
COPY --chown=bsfm:bsfm biosignal_fm ./biosignal_fm
COPY --chown=bsfm:bsfm pyproject.toml README.md LICENSE ./

# Switch to non-root user
USER bsfm

# Read-only filesystem + no-new-privileges (set at runtime via docker-compose or k8s)
EXPOSE 8000

# Healthcheck via FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status==200 else sys.exit(1)"

ENTRYPOINT ["python", "-m", "biosignal_fm.cli.main"]
CMD ["serve", "--public", "--port", "8000"]
