# BOTOX chatbot backend: FastAPI + GraphRAG pipeline.
FROM python:3.12-slim

# Non-root, no build tools in the final layer beyond what wheels need.
RUN groupadd -r botox && useradd -r -g botox -m -d /home/botox botox
WORKDIR /app

# System deps for sentence-transformers/torch wheels are pulled by pip; keep the image lean.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Protegrity Data Protection SDK (`appython`), REQUIRED for tokenization. It is licensed vendor
# software not on public PyPI, so it is not in requirements.txt. Vendor it into `backend/vendor/`
# (a wheel, an sdist, or the `protegrity-ai-developer-python` source tree) and it is installed here.
# If the directory is empty/absent the image still builds, but tokenization is unavailable and the
# backend fails closed at runtime (health reports protection:down) rather than serving unprotected.
COPY vendor/ /tmp/pty-sdk/
# Install any wheel / sdist present (ignore the placeholder README). If none is vendored, build
# succeeds but tokenization is unavailable and the backend fails closed at runtime.
RUN set -e; \
    pkgs="$(find /tmp/pty-sdk -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' \) 2>/dev/null)"; \
    if [ -n "$pkgs" ]; then \
        pip install --no-cache-dir $pkgs; \
    elif [ -f /tmp/pty-sdk/pyproject.toml ] || [ -f /tmp/pty-sdk/setup.py ]; then \
        pip install --no-cache-dir /tmp/pty-sdk; \
    else \
        echo "WARNING: no Protegrity SDK in backend/vendor/; tokenization unavailable, backend fails closed."; \
    fi; \
    rm -rf /tmp/pty-sdk

# The embedding model is downloaded at first ingest and cached in a mounted volume (see compose),
# so it is not baked into the image.
COPY app/ app/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# The embedding cache lives on a named volume; keep it under a stable HF_HOME the entrypoint owns.
ENV HF_HOME=/home/botox/.cache/huggingface

# Data (crawled pages, built index) is a mounted volume in compose; the dir exists for standalone.
# Ownership of the mounted volumes is fixed at container start by the entrypoint (named volumes
# mount as root the first time), which then drops to the unprivileged `botox` user.
RUN mkdir -p /app/data/raw /app/data/processed /app/data/index && chown -R botox:botox /app /home/botox

EXPOSE 8000
# The entrypoint chowns the mounted volumes (as root) then drops to `botox` before exec-ing the CMD.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# One worker: the pipeline holds process-wide singletons (protector, model client, index) and the
# demo is single-tenant. Scale horizontally behind the nginx proxy if ever needed.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
