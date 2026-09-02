PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

.PHONY: test lint corpus ingest train eval hybrid attacks erasure results demo

test:
	PROTOMETER_NO_TRACKING=1 PROTOMETER_NO_TRACING=1 python -m pytest tests/ -q

lint:
	ruff check src scripts tests

corpus:
	python scripts/build_corpus.py

# The UI "data journey" artifact (one record traced through the stages). $0; reads the committed
# clear corpus + protected artifacts, so run it after ingest (or any time the corpus changes).
.PHONY: journey
journey:
	python scripts/build_journey.py

ingest:
	python scripts/ingest_all.py

train:
	python scripts/run_training.py

eval:
	python scripts/run_eval.py --model bedrock-sonnet-5

hybrid:
	python scripts/run_hybrid.py --scope none --capacity 50 --model bedrock-sonnet-5 --grain alert

attacks:
	python scripts/run_attacks.py

erasure:
	python scripts/measure_semantic_erasure.py

results:
	$(PYTHON) scripts/generate_results.py --domain aml > docs/results-aml.md

demo:
	python scripts/demo.py

# One-time download of the open-source local model, so live chat runs without cloud credentials.
# Needs Ollama installed + running (https://ollama.com/download). Pulls PROTOMETER_LOCAL_MODEL
# (default llama3.2, ~2GB). Override the model: `make setup-local-model MODEL=qwen2.5:7b`.
.PHONY: setup-local-model
setup-local-model: ## pull the open-source local model for credential-free live chat
	$(PYTHON) scripts/setup_local_model.py $(if $(MODEL),--model $(MODEL),)

# Regenerate every derived doc from the current artifacts, then verify nothing drifted.
# Run this after any new measurement so the docs (and the cross-domain README summary) update
# everywhere with no stale data points.
.PHONY: docs
docs:
	$(PYTHON) scripts/key_metrics.py --write
	$(PYTHON) scripts/generate_results.py --domain aml > docs/results-aml.md
	$(PYTHON) scripts/generate_summary.py --write
	$(PYTHON) scripts/check_docs.py

# ── Docker ────────────────────────────────────────────────────────────────────────────────────
# up:   the app + its Postgres only (Replay + healthcare/support demos work fully; live chat and
#       paid batch stages additionally need the vendor DE services + AWS creds).
# full: ONE project with ALL 20 services — app + Postgres + the vendor DE services + every
#       observability plane (MLflow, Prometheus/Grafana, Langfuse). One lifecycle: docker-down
#       stops everything together. Do NOT also start those stacks from docker/* or
#       vendor-de/* — this project owns them (clashing container names otherwise).
# The app entrypoint waits for Postgres and loads the corpus mirror, so both are one command to a
# working UI at http://localhost:8000.
# Category profiles: vendor-de + observability are profiled so they can be started selectively; the
# app group has no profile (always starts). COMPOSE_PROFILES activates both for the "everything"
# targets, and is passed to `down` too so it tears the profiled services down as well.
# Point Compose at the repo-root .env for BOTH ${VAR} interpolation and (via each service's env_file)
# the container's environment, so the single documented `.env` drives credentials + config. Compose
# would otherwise look for a .env next to the compose file (docker/app/ui/), which does not exist.
# `--env-file` is included only when .env is present, so Replay mode + the local model still work
# with no .env at all.
ENVFILE := $(if $(wildcard .env),--env-file .env,)
FULL := docker compose $(ENVFILE) -f docker/app/ui/compose.full.yml
ALL_PROFILES := --profile vendor-de --profile observability

# ── Shared infrastructure: decoupled Protegrity DE + observability (bring up FIRST) ───────────
# Two independent shared projects, each on its own external network, so Protometer and BOTOX run at the
# SAME time against ONE tokenizer and ONE observability platform. See docs/adr/shared-infra-decoupling.md.
SHARED_PTY := docker compose $(ENVFILE) -f docker/shared/protegrity/compose.yml
SHARED_OBS := docker compose $(ENVFILE) -f docker/shared/observability/compose.yml
APP        := docker compose $(ENVFILE) -f docker/app/ui/compose.yml

.PHONY: shared-up shared-protegrity-up shared-observability-up shared-down \
        docker-up docker-down docker-ps docker-logs docker-full

shared-protegrity-up: ## shared Protegrity DE tier only (project: protegrity-shared)
	$(SHARED_PTY) up -d
	@echo "Shared Protegrity DE up. classification → :6000 · guardrail → :6001"

shared-observability-up: ## shared observability platform only (project: observability-shared)
	$(SHARED_OBS) up -d
	@echo "Shared observability up. Langfuse → :5006 · MLflow → :5001 · Grafana → :5002 · Prometheus → :5003"

shared-up: shared-protegrity-up shared-observability-up ## BOTH shared tiers — run before any demo
	@echo "Shared tiers up. Now: make docker-up (Protometer) and/or (cd botox_demo && docker compose up -d --build)"

shared-down: ## stop BOTH shared tiers (stop the demos first)
	-$(SHARED_OBS) down
	-$(SHARED_PTY) down

# ── Protometer demo (shared-by-default: needs `make shared-up` first) ────────────────────────────
docker-up:     ## Protometer on the shared tiers → http://localhost:8000  (run `make shared-up` first)
	$(APP) up -d --build
	@echo "Protometer UI → http://localhost:8000  (DE: protegrity-shared · obs: observability-shared)"

docker-down:   ## stop the Protometer demo (leaves the shared tiers up)
	$(APP) down

docker-ps:     ## print the running stack as a tree (project → group → subgroup → services)
	@sh scripts/docker_tree.sh

docker-logs:
	$(APP) logs -f protometer_app

# Legacy self-contained all-in-one (app + bundled DE + bundled observability in ONE project). Kept as
# an escape hatch for a machine that can't run the shared tiers; prefer `make shared-up && make docker-up`.
docker-full:   ## LEGACY: everything bundled in one project (no shared tiers)
	$(FULL) $(ALL_PROFILES) up -d --build
	@echo "Legacy all-in-one up (bundled ports differ from the shared map). UI → :8600"
