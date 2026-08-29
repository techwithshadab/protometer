# Setup — one-time, for anyone who clones or forks this repo

This gets you from a fresh clone to a running demo. **You do not need an AWS or any paid-model
account** — the live assistant falls back to a free local open-source model. You *do* need Protegrity
Developer Edition credentials (free) for the tokenization itself, and Docker.

For what the product does see [product-and-use-cases.md](product-and-use-cases.md); for how it's built
see [architecture.md](architecture.md).

---

## What you need

| Requirement | Why | Cost |
|---|---|---|
| **Docker** (Desktop or Engine) | runs the app, Postgres, and the Protegrity services | free |
| **Protegrity Developer Edition credentials** | the tokenization the whole demo is built on | free — register at [protegrity.com/developers/dev-edition-api](https://www.protegrity.com/developers/dev-edition-api) |
| **A GitHub token for `ghcr.io`** | to pull the vendor Developer-Edition images | free (Developer-Edition registration) |
| **[Ollama](https://ollama.com)** *(only for live chat without a cloud account)* | serves the free local model | free |
| **AWS Bedrock credentials** *(optional)* | live chat + the paid evaluation on the hosted model | pay-per-use |

> **Apple Silicon note:** the Protegrity vendor images are amd64-only and run under emulation (slower
> but works).

---

## Path A — the demo UI (recommended, ~10 minutes)

This is the fastest way to see the product. Replay mode and the healthcare/support views work
immediately; live chat works too, on a local model, once you do the one Ollama step.

```bash
# 1. Clone, and fetch the Protegrity Developer-Edition images repo (gitignored here)
git clone https://github.com/techwithshadab/amlguard.git && cd amlguard
git clone https://github.com/Protegrity-AI-Developer-Edition/protegrity-ai-developer-edition.git vendor-de

# 2. Credentials — copy the template and fill in DEV_EDITION_* (see .env.example for every option)
cp .env.example .env
#    Edit .env: set DEV_EDITION_EMAIL / DEV_EDITION_PASSWORD / DEV_EDITION_API_KEY.
#    Leave AWS_* blank — live chat will use the local model automatically.

# 3. Log in to the vendor image registry (once)
docker login ghcr.io      # username = your GitHub username; password = a GH token with read:packages

# 4. (For credential-free live chat) install Ollama, then pull the local model once
#    Install: https://ollama.com/download  — then:
make setup-local-model    # pulls llama3.2 (~2GB). Skip this if you only want Replay mode.

# 5. Bring up the whole stack as one project and open the UI
make docker-full          # app + Postgres + Protegrity services + observability, one lifecycle
open http://localhost:8600
```

That's it. In the UI:

- **Replay mode** (default) replays a recorded, verified run — needs no model at all.
- **Live mode** runs a real protected turn. The banner tells you which model is answering; with no
  AWS credentials it says *"runs on the local open-source model llama3.2 · $0 per turn."*
- Switch domains (AML / Clinical / Customer-Support) and views (Batch Analysis / Live Assistant) from
  the top bar.

**Lighter option:** `make docker-up` starts just the app + Postgres (Replay + the healthcare/support
batch demos work; live chat and paid batch stages need the full stack).

### Live chat: the model, explained

The live assistant picks its model automatically:

1. `AMLGUARD_UI_MODEL` in `.env` — forces a specific model (wins over everything).
2. Hosted model (`bedrock-sonnet-5`) — used when AWS credentials are present.
3. Local model (`llama3.2`) via Ollama — used when there are no cloud credentials.

One-time model download, two ways:

- **Explicit:** `make setup-local-model` (override with `make setup-local-model MODEL=qwen2.5:7b`).
- **Automatic:** put `AMLGUARD_AUTO_PULL_MODEL=true` in `.env` and it pulls on the first live turn.

By default the app reaches Ollama on your **host** (`host.docker.internal:11434`). To run Ollama
*inside* the Docker stack instead (no host install), add `OLLAMA_URL=http://ollama:11434` to `.env`
and start the `local-model` profile — see [ui/README.md](../ui/README.md#running-live-chat-without-cloud-credentials).

Full details and the config table: [ui/README.md](../ui/README.md).

---

## Path B — reproduce the measurements (advanced)

Only needed if you want to regenerate the evaluation curve yourself (this is what produced the
committed numbers). The AML curve was measured on a hosted model, so this path needs AWS Bedrock
credentials and will incur cost.

```bash
pip install -r requirements.txt

# vendor services (no API key needed): discovery, then the guardrail that depends on it
cd vendor-de/data-discovery
docker compose -f docker-compose.yml -f ../../docker/vendor/discovery.override.yml up -d
cd ../semantic-guardrail
docker compose -f docker-compose.yml -f ../../docker/vendor/guardrail.override.yml up -d
cd ../..

python scripts/check_determinism.py          # verify tokenization is deterministic first
python scripts/build_corpus.py               # build the synthetic corpus
python scripts/ingest_all.py                 # protect it at every scope + build vector indexes
python scripts/demo.py                        # walk the whole pipeline in one command

# the paid evaluation (estimate cost first)
python scripts/estimate_cost.py --model bedrock-sonnet-5
python scripts/run_eval.py --model bedrock-sonnet-5
python scripts/generate_results.py --domain aml > docs/results-aml.md
```

You can also run the eval on a **local** model at $0 (lower quality, but free):
`python scripts/run_eval.py --model qwen2.5-14b`.

---

## Verify it's working

```bash
curl -s http://localhost:8600/api/health | python3 -m json.tool
```

You should see `"ok": true` and a `"live_chat"` block naming the resolved model and whether it's
ready. For the local path it should read `"provider": "ollama"`, `"ready": true`.

Run the test suite anytime: `make test` (or `python -m pytest tests/ -q`).

---

## Troubleshooting

| Symptom | Cause → fix |
|---|---|
| UI loads but **live chat fails** with "Ollama not reachable" | Ollama isn't running → install from [ollama.com](https://ollama.com/download), start it, then `make setup-local-model`. |
| Live banner says **model "not downloaded yet"** | run `make setup-local-model` once, or set `AMLGUARD_AUTO_PULL_MODEL=true` in `.env`. |
| Vendor services **fail to pull** | `docker login ghcr.io` with a GitHub token (read:packages). The app + Postgres + observability still come up and Replay mode works without them. |
| Parties view / chat returns **503** | the Postgres corpus mirror isn't loaded → `make docker-full` loads it automatically; for local dev run `python scripts/load_corpus_db.py --all`. |
| `make docker-full` **out of memory** (guardrail 500s) | the full stack is memory-heavy (Langfuse's ClickHouse alone caps at 3g). Use `make docker-up` (app + Postgres only) or start observability selectively. |
| Credentials in `.env` **not reaching the container** | the make targets pass `--env-file .env`; if you run raw `docker compose`, add `--env-file .env` yourself. |

More operational detail: [docker/README.md](../docker/README.md), [ui/README.md](../ui/README.md).
