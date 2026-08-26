#!/usr/bin/env sh
# AMLGuard app entrypoint: wait for Postgres, load the corpus mirror, then serve.
#
# The app is Postgres-only (ADR-0053): it returns 503 until its corpus mirror is loaded, so the
# container must load it before serving or the parties view and chatbot are dead on first use.
# load_corpus_db.py is idempotent, so a restart re-syncs cheaply rather than duplicating.
set -eu

PORT="${AMLGUARD_UI_PORT:-8600}"

echo "[entrypoint] waiting for Postgres to accept connections..."
# Poll via the app's own db layer (same URL resolution the app uses) so we wait on the exact
# dependency the app needs, not a guessed host/port. Bounded so a truly-down DB fails loudly.
i=0
until python -c "import sys; sys.path.insert(0,'src'); from amlguard import db; sys.exit(0 if db.available() else 1)" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge 60 ]; then
        echo "[entrypoint] Postgres did not become available after 60 tries; giving up." >&2
        exit 1
    fi
    sleep 2
done
echo "[entrypoint] Postgres is up."

# Load the corpus mirror. Only AML ships a party corpus in this edition; --domain aml is the one
# that has data (the loader skips domains with no corpus dir). Idempotent.
echo "[entrypoint] loading corpus into Postgres (idempotent)..."
python scripts/load_corpus_db.py --domain aml || {
    echo "[entrypoint] corpus load failed; the parties view / chatbot will 503 until it succeeds." >&2
}

echo "[entrypoint] starting uvicorn on 0.0.0.0:${PORT}"
# 0.0.0.0 inside the container; the compose maps it to loopback on the host.
exec uvicorn ui.api.app:app --host 0.0.0.0 --port "${PORT}"
