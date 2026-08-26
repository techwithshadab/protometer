# AMLGuard app: the FastAPI backend + the single-page UI, one image.
#
# This image is the APPLICATION only. Its runtime dependencies are NOT bundled here (they cannot
# be): Protegrity Discovery/Guardrail are vendor Developer-Edition images, and Bedrock is external
# AWS. The app reaches them over the network via the AMLGUARD_*_URL env vars (see docker-compose).
# Replay mode and the healthcare/support batch demos need none of them; live chat and paid batch
# stages do.
FROM python:3.12-slim

# Non-root user: the app never needs root, and an exposed service should not run as it.
RUN groupadd -r amlguard && useradd -r -g amlguard -m -d /home/amlguard amlguard

WORKDIR /app

# System deps: git is pulled in by MLflow's provenance stamping; the rest of the wheel set is pure
# Python or ships manylinux wheels, so no build toolchain is needed. --no-install-recommends keeps
# the layer small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (cached across source edits). requirements.txt pins the full set, including
# the fastapi/uvicorn serving layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source. The app imports `amlguard` from src/ (PYTHONPATH) and reads the corpus + committed
# artifacts from data/. config/ carries the model registry. ui/ is the served frontend + API.
COPY src/ src/
COPY scripts/ scripts/
COPY config/ config/
COPY ui/ ui/
COPY data/ data/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && chown -R amlguard:amlguard /app

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    AMLGUARD_UI_PORT=8600

EXPOSE 8600
USER amlguard

# python:slim has no curl; probe with urllib against the app's own health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8600/api/health',timeout=3).status==200 else 1)" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
