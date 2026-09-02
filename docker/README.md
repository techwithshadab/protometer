# docker/

Every container definition for the project, grouped by concern. The whole thing runs as **one
Compose project** from `docker/app/ui/compose.full.yml` (the app + its Postgres + the vendor
Developer-Edition services + the observability planes) — see the repo root README.

## Layout

```
docker/
├── entrypoint.sh                       the app container's entrypoint (wait for Postgres → load corpus → serve)
├── app/                                the application
│   ├── ui/
│   │   ├── Dockerfile                  the FastAPI + UI image
│   │   ├── compose.yml                 app + Postgres (standalone: make docker-up)
│   │   └── compose.full.yml            the whole stack, one project (make docker-full)
│   └── postgres/
│       ├── compose.yml                 app corpus mirror (container protometer_postgres, :5433)
│       └── init/01_schemas.sql         per-domain schemas (aml / healthcare / support)
├── observability/                      the three telemetry planes
│   ├── infra/
│   │   ├── compose.yml                 Prometheus / Grafana / Pushgateway
│   │   ├── prometheus.yml
│   │   └── grafana/                    provisioning + dashboards
│   ├── mlflow/
│   │   ├── compose.yml                 MLflow server (:5001)
│   │   └── store/                      backend db + logged artifacts (gitignored; see below)
│   └── langfuse/
│       ├── compose.nested.yml          Langfuse, renamed to nest under one project
│       ├── compose.override.yml        loopback + memory override for the vendored Langfuse
│       └── env.example
└── vendor/                             our overrides for the vendor Developer-Edition composes
    ├── discovery.override.yml           amd64 pin + loopback for the vendor Discovery compose
    └── guardrail.override.yml           amd64 pin + loopback for the vendor Guardrail compose
```

Compose `include` paths in `app/ui/compose*.yml` are relative to `docker/app/ui/` (repo root at
`../../..` for vendor images, docker stacks at `../../<group>/<stack>/`); the app build context is
the repo root. The vendor DE images themselves stay in `vendor-de/` (a vendored repo we don't
relocate); only our override files live under `docker/vendor/`.

## Bring-up

One command brings the whole stack up as the `protegrity` project:

```bash
make docker-full        # app + Postgres + vendor DE + observability (20 services)
make docker-up          # just app + Postgres (Replay + healthcare/support demos)
make docker-ps          # the running stack as a tree (project → group → subgroup)
make docker-down        # stop everything together
```

All ports are loopback-bound. This is the memory-heavy configuration (Langfuse's ClickHouse caps
at 3g + MLflow + the vendor DE services); it fits a ~15 GB Docker VM with headroom. On a smaller
VM, prefer `make docker-up` and start observability selectively (`make docker-observability`).

Because this one project owns the containers, do **not** also start the same stacks from their
own directories — that clashes on the path-style container names (`app-*`, `vendor-de-*`,
`observability-*`).

## The MLflow store

`mlflow/store/` holds the backend SQLite db (`mlflow.db`) and logged artifacts (the SHAP/PR/ROC
plots the demo UI serves). It is **gitignored** — at scope `none` those artifacts include model
reasoning over clear narratives, which must never enter the repository. The app reads it via
`protometer.tracking.DEFAULT_TRACKING_DIR` (`docker/observability/mlflow/store`, override with
`PROTOMETER_MLFLOW_STORE_DIR`); the full-stack app container mounts it read-only so `/api/plot` can
serve the plots. It is NOT baked into the image (`.dockerignore` excludes `docker/`).

## App data layer: Postgres

`postgres/` runs a small Postgres (host port 5433) that mirrors the corpus into per-domain schemas
(`aml`, `healthcare`, `support`). For the offline pipeline the file corpus (`data/corpus/*.json`)
stays the source of truth; the loader [`scripts/load_corpus_db.py`](../scripts/load_corpus_db.py)
rebuilds the mirror. **The online app treats Postgres as its source of truth with no JSON
fallback**: a parties/chat request returns **503** when the DB is down or unloaded. The
app container's entrypoint runs the loader automatically; for a manual/local run:

```bash
cd docker/app/postgres && docker compose up -d          # DB on 127.0.0.1:5433
python scripts/load_corpus_db.py --domain aml        # load the AML corpus into the aml schema
```

MLflow deliberately does **not** use this Postgres for its backend (the model-registry alias API is
broken against Postgres in the pinned version); only the corpus mirror lives here.
