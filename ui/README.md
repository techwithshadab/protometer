# Aegis — Protected-Pipeline Intelligence (the AMLGuard UI)

A judge- and executive-facing console (product name **Aegis**; codebase name AMLGuard) that
replays the verified batch run and the chatbot over the same protection primitives the pipeline
uses. Light editorial design, Title-Case headings, per-domain views.

Two use cases, each with a Replay (committed run) and a Live (real, billed) mode:
- **Pipeline** — a per-domain stage stepper; each stage reveals its measured artifact (AP-vs-scope
  curve with live SHAP + PR plots, semantic-erasure recall, the LLM curve, hybrid precision, HIPAA
  risk table, the support dual-gate).
- **Assistant** — Replay plays a committed multi-turn transcript
  (`data/eval/<domain>/chat_replay.json`) showing the per-turn protection boundary (tokenized in →
  model over tokens → egress scan → role-gated reveal); Live runs a real protected turn.

## Run with Docker (one command)

The app + its Postgres are dockerised; the container's entrypoint waits for Postgres and loads the
corpus mirror, so this is the whole setup:

    make docker-up                 # build + start app + Postgres
    open http://localhost:8600     # Replay + healthcare/support demos work immediately

`make docker-up` (= `docker compose -f docker/app/ui/compose.yml up -d --build`) gives you a working
UI with **no manual corpus load**. Replay mode and the healthcare/support batch demos need nothing
else.

**Live chat** additionally needs the vendor Developer-Edition services (for tokenization) plus a
model — either AWS Bedrock, or a free local open-source model (see the next section). **Paid batch
stages** specifically need AWS Bedrock, because their committed artifacts were produced by it.
`make docker-full` brings up **the entire stack as ONE Compose project** — the app + Postgres + the
four vendor DE services + every observability plane (MLflow, Prometheus/Grafana, Langfuse), 20
services on one shared network, one lifecycle:

    docker login ghcr.io           # once, with a Developer-Edition token (free registration)
    # put DEV_EDITION_* and AWS_* creds in .env (see .env.example)
    make docker-full               # = docker compose -f docker/app/ui/compose.full.yml … up -d --build
    # UI → :8600 · MLflow → :5001 · Langfuse → :3000 · Grafana → :3001

The 20 services are **categorized into three groups** (`app`, `vendor-de`, `observability`) via
Compose labels + profiles, so you can list and start them by category:

    make docker-ps            # list running services grouped by category
    make docker-vendor        # app + Protegrity DE services only (no observability)
    make docker-observability # app + observability planes only (no vendor DE)
    make docker-full          # all three groups (20 services)

`make docker-down` stops **everything together**; `make docker-logs` tails the app. Because this one
project owns all those containers, do NOT also start the same stacks from `docker/*` or
`vendor-de/*` (that would clash on container names). The vendor images are amd64-only (emulated on
Apple Silicon) and this is the memory-heavy configuration (Langfuse's ClickHouse alone caps at 3g);
see the header of `docker/app/ui/compose.full.yml`. Langfuse nests via
`docker/observability/langfuse/compose.nested.yml`, which renames its generic `postgres`/`redis`/
`minio`/`clickhouse` services so they don't collide with the app Postgres — the vendored Langfuse
compose is left untouched.

## Run without Docker (local dev)

The app reads party data from Postgres (its source of truth; no JSON fallback), so
start the corpus mirror and load it BEFORE the server, or the parties view and chatbot return
503:

    pip install -r requirements.txt                     # includes fastapi/uvicorn
    cd docker/app/postgres && docker compose up -d    # start the corpus mirror (port 5433)
    python scripts/load_corpus_db.py --domain aml        # load the AML corpus into Postgres
    uvicorn ui.api.app:app --port 8600
    open http://localhost:8600

One server serves both the API (`/api/*`) and the single-page frontend (`/`).

## Running live chat without cloud credentials

The Live-mode chat needs a model. If you have AWS Bedrock credentials it uses the hosted model
(`bedrock-sonnet-5`), matching the committed evaluation artifacts. **If you don't, it automatically
falls back to an open-source model served locally by [Ollama](https://ollama.com)** — same protected
pipeline (tokenization still goes through Protegrity Developer Edition), just a local model doing the
reasoning, at $0 per turn. The UI's Live-mode banner tells you which model is active.

**One-time setup** (only needed for the local fallback):

1. Install Ollama — <https://ollama.com/download> — and make sure it's running.
2. Pull the model, either explicitly or automatically:
   - Explicit: `make setup-local-model` (pulls `llama3.2`, ~2GB; override with
     `make setup-local-model MODEL=qwen2.5:7b`).
   - Automatic: set `AMLGUARD_AUTO_PULL_MODEL=true` and the app pulls it on the first live turn.

**How the model is chosen** (all env-driven, so a fork runs as-is):

| Variable | Default | Effect |
|---|---|---|
| `AMLGUARD_UI_MODEL` | *(unset)* | Force a specific `config/models.yaml` model for live chat; wins over everything below. |
| `AMLGUARD_HOSTED_MODEL` | `bedrock-sonnet-5` | The hosted model used **when AWS credentials are present**. |
| `AMLGUARD_LOCAL_MODEL` | `llama3.2` | The open-source Ollama model used **when there are no cloud credentials**. |
| `AMLGUARD_AUTO_PULL_MODEL` | `false` | Pull the local model on first use instead of requiring `make setup-local-model`. |
| `OLLAMA_URL` | host: `localhost:11434`; container: `host.docker.internal:11434` | Where Ollama listens. An explicit value always wins; unset, it defaults per environment (a container's own `localhost` is not the host). |

Precedence: `AMLGUARD_UI_MODEL` → hosted (if AWS creds) → local. `GET /api/health` reports the
resolved model, its provider, and whether it's ready.

**With Docker**, the app reaches Ollama on your **host** by default (`host.docker.internal:11434`),
so `ollama serve` on the host just works — set nothing. To run Ollama **inside** the stack instead
(no host install), use the `local-model` profile and set `OLLAMA_URL` in your `.env`:

    echo 'OLLAMA_URL=http://ollama:11434' >> .env
    make docker-full                 # (or: docker compose --env-file .env … --profile local-model up -d)
    docker exec ollama ollama pull llama3.2      # or set AMLGUARD_AUTO_PULL_MODEL=true in .env

That Ollama container is CPU-only here (fine for the demo, slower than a host GPU), reached over the
network as `ollama:11434` (it publishes no host port, so it never clashes with a host Ollama), and
its model cache persists in the `ollama-models` volume.

## What it shows

- **Global header:** Use case (Batch | Chatbot) · Domain (AML / healthcare / customer-support)
  · Replay/Live toggle.
- **Batch:** a **per-domain** architecture flow as a clickable stepper. AML shows the
  7-stage curve (`INGEST → TRAIN → EMBED → RETRIEVE → INFER → EGRESS → PRESENT`); healthcare shows
  Safe Harbor → Expert Determination (HIPAA risk before/after k-anonymization); customer-support
  shows the two-gate dual-gate (classify+tokenize, then role-gated detokenize). In **Replay** mode
  each stage reveals the **real committed-artifact metrics** for the verified run. In **Live** mode
  a stage runs FOR REAL: local stages ($0) run immediately; a paid stage (AML infer/egress) first
  returns a **cost estimate + Confirm button** and executes only after you click it. Provenance
  header names the domain, corpus fingerprint, and model.
- **Chatbot:** two tabs — *Conversation* (a live protected turn via `ConversationSession`) and
  *Pipeline internals* (that turn's boundary: inbound tokenized → what the model saw → egress
  scan → role-gated re-identification). Live chat runs in all three domains, each protected
  against its own party corpus (loaded by the container entrypoint); the healthcare/support
  batch demos additionally run in the stepper.

## Slices

- [x] Slice 0 — FastAPI seam (`ui/api/app.py`): read endpoints + live chat primitive.
- [x] Slice 1 — Replay batch (the stepper + metric panels).
- [x] Slice 2 — Live chatbot (two-tab).
- [x] Slice 3 — Live-batch execution: `POST /api/batch/run-stage` behind the cost rails
      (estimate-first + server-issued single-use confirm token, cross-process run-lock → 409 on
      contention, `data/eval/_live/<run_id>/` isolation), plus per-domain batch views.
- [ ] Slice 4 — polish/theming.

Replay reads the verified artifacts and never mutates them. The chatbot's Live turns are cheap
(~$0.01) real Bedrock calls. Live-batch never writes the committed artifacts: every stage writes
to an isolated `_live/<run_id>/` dir, so a demo can never clobber the verified Replay results.

## Security posture

Two rails gate anything that bills or exposes data, both off by default for a loopback demo but
ready for a real deployment:

- **`AMLGUARD_UI_API_TOKEN`** — when set, the billed endpoints (`/api/chat/turn`,
  `/api/batch/run-stage`) and the parties preview (`/api/corpus/parties`) require an
  `X-AMLGuard-Token` header. Bind beyond loopback only with this set.
- **PII never leaves the parties preview in the clear.** `/api/corpus/parties` returns only a safe
  column allow-list (id, name, type, jurisdiction, risk, PEP); clear ssn/credit-card/DOB/account
  fields are dropped at the seam regardless of auth — the tokenized serving path is the only way to
  reach protected identity data.
