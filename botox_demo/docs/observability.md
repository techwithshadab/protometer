# Observability & KPI reference

Botox has **two observability planes**, each for a different question:

- **Langfuse** (per-turn detail) — the "what did this turn say and cost" plane. Every chat turn is
  traced to the **shared Langfuse v4** (UI at `http://127.0.0.1:5006`), into botox's own project
  (org *Protegrity* → project *Botox*). Most of this file documents it.
- **Prometheus + Grafana** (operational time-series) — the "how is the service behaving over time"
  plane. The backend exposes a scraped `/metrics` endpoint; Grafana has a **Botox** domain
  dashboard (`http://localhost:5002`, admin/protometer). See *Live-serving metrics* below.

This file is the reference for what is captured and how to turn it into dashboards. The per-turn
numbers live in Langfuse; the aggregate operational series live in Prometheus.

Sign in to the shared Langfuse with the operator login: `admin@protegrity.local` / `protegrity-admin`
(set in the shared observability tier; change it for anything beyond local). The **Botox** project is
created once in the UI (the project-provisioning API is Enterprise-only), and its key pair goes into
`.env` as `BOTOX_LANGFUSE_PUBLIC_KEY` / `BOTOX_LANGFUSE_SECRET_KEY` — read first in the compose so
botox traces stay in the Botox project, never in Protometer's.

## What a trace looks like

One trace per chat turn, named `botox-chat-turn`:

- **input**, the *protected* (tokenized) message. Detected PII is tokenized before tracing, so
  the trace holds surrogate tokens like `[NAME]NA6525…[/NAME]` and an entity **count**. (Detection
  is best-effort, undetected PII is not tokenized; see *Privacy posture* below and the README's
  *Honest limitations*.)
- **userId**, the anonymous visitor id (an opaque `v-…` token from the browser's localStorage; no
  PII, no IP, no device data). Lets you follow a returning visitor and group their turns.
- **sessionId**, the conversation id (one per page load), so all turns in a conversation group.
- **observations**, `protect` (event), `retrieve` (span), and `generate` — a first-class Langfuse
  **generation** carrying the model, token **usage** (input/output), and a link to the prompt version
  it ran (`botox-system`), each with latency + structured I/O.
- **scores**, the KPIs below (first-class Langfuse scores: chartable, filterable, aggregatable).
- **metadata**, `sources` (the cited page URLs), `topic`, `outcome`, `total_ms`, entity info.

## Scores (KPIs) emitted per turn

| Score | Type | Meaning |
|---|---|---|
| **Answer quality** | | |
| `grounding` | number 0–1 | lexical overlap of the answer with retrieved context (anti-hallucination) |
| `retrieval_hits` | number | chunks retrieved (vector seeds + graph expansion) |
| `top_similarity` | number 0–1 | cosine similarity of the best-matching chunk |
| `source_pages` | number | distinct source pages behind the answer |
| `outcome` | categorical | `answered` / `refused` / `blocked` / `error` |
| **Safety & guard** | | |
| `safety_flagged` | 0/1 | the answer discussed risk/side-effects → ISI note shown |
| `guard_action` | categorical | `passed` / `refused` / `blocked` |
| `pii_leak_blocked` | 0/1 | the egress guard blocked a reply that leaked PII/a token |
| `pii_entities_protected` | number | PII spans tokenized in the user's message |
| `protection_backend` | categorical | `protegrity` (Protegrity is required; no mock) |
| `semantic_guard_action` | categorical | Protegrity Semantic Guardrail verdict on the reply (`approved`/`rejected`), when wired |
| `semantic_guard_score` | number 0–1 | the guardrail's risk score for the reply (higher = riskier), when wired |
| **Performance & cost** | | |
| `retrieve_ms` | number | retrieval latency |
| `generate_ms` | number | model generation latency |
| `answer_chars` | number | answer length in characters |
| `est_answer_tokens` | number | rough token estimate (~4 chars/token) for Ollama, which omits usage; hosted models report real token usage on the generation observation |
| `model` | categorical | e.g. `ollama:llama3.2` |
| **Engagement** | | |
| `topic` | categorical | `cost` / `side_effects` / `dosing` / `conditions` / `how_it_works` / `find_provider` / `other` |
| `user_feedback` | -1 / +1 | the visitor's thumbs-down / thumbs-up on the answer |

`user_feedback` is attached later, when the visitor clicks the thumbs control under an answer, it
lands on the same trace via `POST /api/feedback {trace_id, rating}`.

## Dashboards to build (Langfuse → Dashboards → New)

Langfuse v4 dashboards are built in the UI. Build these panels once; each is a chart over the scores
above (and, for cost, over the generation's native token usage).

1. **Answer quality**
   - Avg `grounding` over time (line): the headline anti-hallucination metric.
   - `outcome` distribution (pie/bar): answered vs refused vs blocked share.
   - Avg `top_similarity` and `retrieval_hits` (line): retrieval health.
2. **Safety & compliance**
   - `guard_action` distribution, how often the egress guard refuses/blocks.
   - Count where `pii_leak_blocked = 1`, must stay at/near zero; alert if it rises.
   - Sum `pii_entities_protected`, how much PII the widget is tokenizing.
   - `safety_flagged` rate, share of answers carrying the ISI note.
3. **Performance & cost**
   - p50/p95 `generate_ms` and `retrieve_ms` (line): latency SLOs.
   - Avg `est_answer_tokens` (line): proxy for cost/verbosity.
   - `model` breakdown, which model served traffic.
4. **Engagement & satisfaction**
   - `topic` distribution (bar): what visitors actually ask about.
   - `user_feedback` avg and 👍/👎 counts (line + bar): satisfaction, and a filter to read the
     low-rated answers.
   - Unique `userId` and returning-visitor rate; turns per `sessionId`.

Cross-filter any panel by `topic`, `outcome`, `model`, or `userId`.

## Query the KPIs from the API

The same data is available programmatically (basic auth = `public_key:secret_key`; use the Botox
project's key pair, `BOTOX_LANGFUSE_PUBLIC_KEY:BOTOX_LANGFUSE_SECRET_KEY`):

```bash
# recent traces (newest first)
curl -s -u "$BOTOX_LANGFUSE_PUBLIC_KEY:$BOTOX_LANGFUSE_SECRET_KEY" \
  "http://127.0.0.1:5006/api/public/traces?limit=20"

# all scores of one kind: e.g. every thumbs rating
curl -s -u "$BOTOX_LANGFUSE_PUBLIC_KEY:$BOTOX_LANGFUSE_SECRET_KEY" \
  "http://127.0.0.1:5006/api/public/scores?name=user_feedback&limit=100"

# one trace in full (observations + scores + metadata)
curl -s -u "$BOTOX_LANGFUSE_PUBLIC_KEY:$BOTOX_LANGFUSE_SECRET_KEY" \
  "http://127.0.0.1:5006/api/public/traces/<TRACE_ID>"
```

## Privacy posture (what is captured)

- **Detected PII is tokenized before it reaches the trace.** The trace `input` is the *protected*
  message, the pipeline tokenizes PII first, so detected spans appear as surrogate tokens (e.g.
  `[NAME]NA6525…[/NAME]`) plus an entity **count**, never raw values. **Detection is Protegrity
  Discovery (ML/NER) — there is no mock and no regex fallback** (see the README's *Honest
  limitations*). The residual limitation is Discovery's *coverage at the configured score
  threshold*: PII it does not flag (e.g. a bare first name below the threshold) is not tokenized and
  appears in the trace `input` as typed. The "no cleartext PII" guarantee holds for **detected**
  PII; lower the threshold for broader coverage. If Discovery or the tokenizer is unreachable the bot
  fails closed (health not-ready, every turn refuses) — so an untokenized value never reaches the
  trace via a degraded protection path.
- **No IP, device id, WiFi, or MAC.** A public web widget cannot read the latter three, and raw IP
  against health questions is deliberately not stored. The only identifier is the anonymous,
  opaque `userId`.
- **Best-effort, fail-open.** Tracing never changes an answer and never raises into the request
  path; if Langfuse is down or unconfigured, the bot runs unchanged and traces are simply skipped.

## Live-serving metrics (Prometheus + Grafana)

The backend exposes a Prometheus **`/metrics`** endpoint (`app/obs/metrics.py`), scraped every 15s
by the shared Prometheus. This is the operational time-series plane — a long-running server is
scraped, not push-based like the Protometer batch ingest. Reachable on the backend loopback port only
(nginx does not proxy it publicly, like the other operator endpoints). Turn it off with
`BOTOX_NO_METRICS=1`; a missing `prometheus_client` also degrades it to a no-op, so serving never
depends on it.

**Metrics emitted** (all labelled `demo="botox"`):

| Metric | Type | What it shows |
|---|---|---|
| `botox_serving_turns_total{outcome}` | counter | turn volume by outcome (ok / refused / blocked / error) |
| `botox_serving_turn_latency_seconds` | histogram | end-to-end latency (emergency→protect→retrieve→generate→egress); p50/p90/p99 |
| `botox_serving_entities_protected_total` | counter | visitor PII entities tokenized at ingress |
| `botox_serving_egress_blocks_total` | counter | replies withheld/redirected by the egress guard |
| `botox_serving_protection_down_total` | counter | turns refused because Protegrity was down (fail-closed proof) |
| `botox_serving_emergency_hits_total` | counter | urgent-symptom short-circuits |
| `botox_serving_grounding_score` | histogram | grounding-score distribution over answered turns |
| `botox_serving_llm_tokens_total{direction}` | counter | estimated LLM tokens (~4 chars/token; Ollama does not report usage uniformly) |
| `botox_serving_errors_total{kind}` | counter | genuine errors by kind (excludes egress blocks) |

**Dashboard:** the **Botox** dashboard in Grafana (`http://localhost:5002`, admin/protometer) —
turn-rate-by-outcome, latency percentiles, the protection/emergency stats, grounding histogram, and
token throughput. Provisioned from `docker/observability/infra/grafana/dashboards/domain-botox.json`.
