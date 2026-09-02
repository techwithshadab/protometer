# Developer guide

## Run the whole stack

The demo shares one Protegrity DE tier and one observability stack (Langfuse v4, MLflow, Prometheus,
Grafana) with the AMLGuard submission on the same host. Bring those **shared tiers** up first (from
the repo root), then the botox stack:

```bash
make shared-up                     # repo root: shared Protegrity DE + shared observability (Langfuse v4)

cd botox_demo
cp .env.example .env               # set DEV_EDITION_* + BOTOX_LANGFUSE_* keys
docker compose up -d --build       # neo4j + backend + frontend (joins protegrity-shared + observability-shared)
docker compose run --rm ingest     # crawl + build the GraphRAG index (one time)
open http://localhost:9001
```

Services (all loopback-bound). The 9xxx band is botox's; the Protegrity (6xxx) and observability
(5xxx) tiers are shared and reached by container name over external networks:

| Service | Port | What |
|---|---|---|
| frontend | 9001 | React landing page + chat widget, served by nginx, proxies `/api` → backend |
| backend | 9000 | FastAPI: `POST /api/chat`, `POST /api/chat/stream`, `POST /api/feedback`, `POST /api/support/reveal` (role-gated), `GET /api/health` |
| neo4j | 9002 / 9003 | knowledge-graph browser UI / bolt |
| Langfuse (shared) | 5006 | agent-tracing UI at http://127.0.0.1:5006 (see [Observability](#observability)) |
| Protegrity Discovery (shared) | 6000 | `pty-classification`, the PII-detection service the backend calls in-network on :8050 |

The `--profile bundled` composes still exist for a fully self-contained run (a private Langfuse +
Protegrity copy inside this project); the default path uses the shared tiers.

## Observability

Every chat turn is traced to the **shared Langfuse v4** (UI at http://127.0.0.1:5006), into botox's
**own** project (org *Protegrity* → project *Botox*): the protected input, retrieval (which chunks and
scores fed the answer), the model as a first-class **generation** (model + token usage + linked prompt
version) with latency, the guard verdict and grounding score, and the outcome (answered / refused /
blocked). **Detected** PII is tokenized before tracing, so the trace holds the protected text plus an
entity **count**; detection is Protegrity Discovery (no mock/regex fallback) and its coverage at the
configured threshold is the residual limit (see the README's *Honest limitations* and
[observability.md](observability.md)). Sources are recorded in the trace (operator-facing), not shown
in the chat UI.

Tracing turns on once botox's project keys are in `.env`. The shared Langfuse is one instance under
org **Protegrity** with a project per demo; the Botox project is created **once in the UI** (the
project-provisioning API is Enterprise-only) and its keys pasted in:

```bash
open http://127.0.0.1:5006           # sign in with the shared operator login:
#   email:    admin@protegrity.local
#   password: protegrity-admin
# Create (once) the "Botox" project under org Protegrity, copy its key pair into .env:
#   BOTOX_LANGFUSE_PUBLIC_KEY=pk-lf-...
#   BOTOX_LANGFUSE_SECRET_KEY=sk-lf-...
# Then send a chat turn and open the Botox project to see the trace.
```

Botox's keys are read **first** in the compose (falling back to the shared default), so its traces
never leak into AMLGuard's project. Set `BOTOX_TRACING=off` to force-disable tracing even when keys
are present; if the keys are cleared the tracing layer becomes a no-op and the bot runs unchanged.
The SDK is Langfuse **v4** (`langfuse==4.14.4`); the shared instance runs Langfuse v4 in events-only
mode, so traces land in the unified event store.

See [observability.md](observability.md) for the full KPI/score reference, the dashboards to build,
and the privacy guarantees.

## The reasoning model

Chosen at request time, precedence: `BOTOX_MODEL` (override) → Anthropic/Bedrock (if credentials
present) → local Ollama (`llama3.2`, the $0 default). The backend reaches a **host-run** Ollama via
`host.docker.internal`. Pull the model once on the host:

```bash
ollama pull llama3.2
```

`GET /api/health` reports the resolved model and provider, and which protection backend is active.

## Rebuilding the index

The crawled pages and built index live under `data/` (a mounted volume). To refresh:

```bash
docker compose run --rm ingest                      # re-crawl + rebuild everything
# or, granular, inside the backend container:
docker compose exec backend python -m app.ingest.crawl --limit 40
docker compose exec backend python -m app.ingest.build_all
```

The crawl step fetches the live botox.com site; the rest of the ingest step (chunk → graph → index)
is deterministic and calls no LLM ($0). The embedding model downloads once and caches in the
`hf-cache` volume.

## Running the backend locally (without Docker)

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OLLAMA_URL=http://localhost:11434            # host Ollama
export NEO4J_URL=bolt://localhost:9003              # the compose Neo4j (bolt is published on 9003)
python -m app.ingest.build_all                      # build the index first
uvicorn app.api.main:app --reload --port 9000
```

The bare-`uvicorn` CORS default already allows the Vite dev origin (`:5173`) and the frontend's
`:9001`; override `BOTOX_ALLOWED_ORIGINS` for anything else.

## Running the frontend locally

```bash
cd frontend
npm install
VITE_API_BASE=http://localhost:9000/api npm run dev
```

## Protegrity (required)

PII protection is Protegrity, and it is mandatory, there is no mock and no regex fallback.
**Detection** uses the local Data Discovery service; **tokenization** uses the appython SDK against
hosted Data Protection. Configure:

```bash
# Detection: the SHARED Discovery service, reached in-network by name (default). The shared tier also
# publishes it on the host at 127.0.0.1:6000 for out-of-container use.
PROTEGRITY_DISCOVERY_URL=http://pty-classification:8050/pty/data-discovery/v2/classify/text
PROTEGRITY_DISCOVERY_THRESHOLD=0.6
# Tokenization (hosted Data Protection via appython SDK)
DEV_EDITION_EMAIL=...
DEV_EDITION_PASSWORD=...
DEV_EDITION_API_KEY=...
```

The `appython` SDK must be installed in the backend image for tokenization. If Discovery or the
tokenizer is unreachable, the backend **fails closed**: `GET /api/health` returns `ready: false`
with `protection: down`, and every chat turn refuses (no unprotected text reaches retrieval, the
model, or the trace). An urgent-symptom message still surfaces during an outage, it is classified on
the raw text before protection.

Optional egress ML scan (layered on the regex guard):

```bash
PROTEGRITY_SEMANTIC_GUARD=on
PROTEGRITY_SEMANTIC_GUARD_URL=http://pty-guardrail:8001/pty/semantic-guardrail/v1.1/conversations/messages/scan
```

## Tests

```bash
cd backend && python -m pytest -q
```

The tests pin the protection boundary (PII never reaches the model, no-op tokens redacted) and the
egress guards (grounding floor, medical-advice block, PII-leak block, fail-closed refusals).
