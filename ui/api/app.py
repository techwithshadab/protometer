"""AMLGuard demo API — the FastAPI seam the UI (and the Streamlit escape hatch) call.

Wraps existing library functions and the committed run artifacts; it does not re-implement any
pipeline logic. Three tiers of endpoint, by cost and effect:
  * Read/REPLAY — serve the committed, verified artifacts (data/eval/...); $0, no side effects.
  * Live-local — run a cheap local primitive for real (a chatbot turn, a re-identification, a
    de-id / dual-gate stage); $0 hosted, but a chat turn is a billed LLM call gated by the abuse
    rails below.
  * Live-batch — run one pipeline stage FOR REAL behind cost rails: estimate-first with
    a server-issued single-use confirm token, a cross-process run-lock (409 on contention), and
    _live/<run_id>/ isolation so a live run can never clobber the committed Replay artifacts.

Party data is served from Postgres, the app's source of truth: 503, never a JSON
fallback, when the corpus mirror is down or unloaded. The stage list is per-domain:
AML's 7-stage curve, healthcare's HIPAA de-id, support's dual-gate.

Run:  uvicorn ui.api.app:app --reload --port 8600
      (requires the Postgres corpus mirror: see ui/README.md for the deploy/ingest steps)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))

from amlguard.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from amlguard.domains import domain_names, get_domain  # noqa: E402
from amlguard.llm import ModelRegistry  # noqa: E402

EVAL = ROOT / "data" / "eval"


def _mlflow_store() -> Path:
    """The local MLflow store dir (backend db + artifacts), from the one source of truth in
    tracking.py so the UI never re-hardcodes the path. Lives under docker/observability/mlflow/store."""
    from amlguard.tracking import DEFAULT_TRACKING_DIR
    return DEFAULT_TRACKING_DIR


app = FastAPI(title="Aegis: Protected-Pipeline Intelligence (AMLGuard)", version="0.1")
# CORS is restricted to loopback, not `*`: this API makes paid LLM calls and re-identifies PII,
# so it must not be callable from an arbitrary origin. The frontend is served from the same
# origin (no cross-origin call needed); localhost variants are allowed for a separate dev server.
# Override with AMLGUARD_UI_ALLOWED_ORIGINS (comma-separated) for a real deployment behind auth.
import os as _os  # noqa: E402

_allowed = _os.getenv("AMLGUARD_UI_ALLOWED_ORIGINS")
_origins = ([o.strip() for o in _allowed.split(",")] if _allowed else
            ["http://localhost:8600", "http://127.0.0.1:8600",
             "http://localhost:5173", "http://127.0.0.1:5173"])
app.add_middleware(
    CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── The architecture flow: the batch stepper's source of truth, PER DOMAIN ────────────────────
# Each stage names the artifact the Replay panel reveals, a one-line "what it measures", and
# whether running it Live is `paid` (a hosted LLM call) — the frontend gates paid stages behind a
# cost confirm. `run` is the internal stage key the live-batch endpoint dispatches on.
_AML_STAGES = [
    {"id": "ingest", "title": "Ingest", "subtitle": "discover + tokenize",
     "measures": "PII found and protected per scope; unprotected values caught; latency",
     "artifact": "protected/ingestion_summary.json", "paid": False, "run": "ingest"},
    {"id": "train", "title": "Train", "subtitle": "RandomForest on protected ledger",
     "measures": "AP-vs-scope curve + SHAP reliance (what protection costs a classifier)",
     "artifact": "eval/training.json", "paid": False, "run": "train"},
    {"id": "embed", "title": "Embed", "subtitle": "local vectors over tokens",
     "measures": "index built over protected narratives (no plaintext embedded)",
     "artifact": None, "paid": False, "run": "embed"},
    {"id": "retrieve", "title": "Retrieve", "subtitle": "semantic search over tokens",
     "measures": "identity search collapses, behavioural survives (Semantic Erasure)",
     "artifact": "eval/semantic_erasure.json", "paid": False, "run": "retrieve"},
    {"id": "infer", "title": "Infer", "subtitle": "LLM reasons over tokens",
     "measures": "investigation-quality curve across protection scopes",
     "artifact": "eval/bedrock-sonnet-5", "paid": True, "run": "infer"},
    {"id": "egress", "title": "Egress", "subtitle": "guardrail leak-check",
     "measures": "responses scanned before a human sees them",
     "artifact": "eval/hybrid_none.json", "paid": True, "run": "egress"},
    {"id": "present", "title": "Present", "subtitle": "role-gated re-identify",
     "measures": "the same protected record, re-identified per role: plaintext only for the role entitled to it",
     "artifact": None, "paid": False, "run": "present"},
]
_HEALTHCARE_STAGES = [
    {"id": "safe_harbor", "title": "Safe Harbor", "subtitle": "remove + tokenize identifiers",
     "measures": "HIPAA Safe-Harbor identifier categories present in the schema removed/tokenized; "
                 "names tokenized; no-op leaks redacted",
     "artifact": "eval/healthcare/deidentify.json", "paid": False, "run": "healthcare"},
    {"id": "expert_determination", "title": "Expert Determination", "subtitle": "quantify residual risk",
     "measures": "Residual re-identification risk under the three standard HIPAA attacker models "
                 "(worst-case, targeted, and average-case), before vs after k-anonymization",
     "artifact": "eval/healthcare/deidentify.json", "paid": False, "run": "healthcare"},
]
_SUPPORT_STAGES = [
    {"id": "gate1", "title": "Gate 1 · Protect", "subtitle": "classify then tokenize PII",
     "measures": "the inbound customer message tokenized before anything downstream sees it",
     "artifact": "eval/support/dual_gate.json", "paid": False, "run": "support"},
    {"id": "gate2", "title": "Gate 2 · Role Dual-Gate", "subtitle": "detokenize per role",
     "measures": "same protected reply: agent sees masked, supervisor sees full",
     "artifact": "eval/support/dual_gate.json", "paid": False, "run": "support"},
]
DOMAIN_STAGES = {
    "aml": _AML_STAGES,
    "healthcare": _HEALTHCARE_STAGES,
    "customer-support": _SUPPORT_STAGES,
}
# Model that produces each domain's paid stages (for the batch stepper header + cost estimate).
DOMAIN_MODEL = {"aml": "bedrock-sonnet-5", "healthcare": "(local, $0)", "customer-support": "(local, $0)"}


def _read(rel: str) -> Any:
    path = (ROOT / "data" / rel)
    if not path.exists():
        raise HTTPException(404, f"artifact not found: {rel}")
    return json.loads(path.read_text())


class DataUnavailable(HTTPException):
    """The app's data store (Postgres) is unreachable or unpopulated. 503, no file fallback."""

    def __init__(self, detail: str):
        super().__init__(status_code=503, detail=detail)


def _load_parties(domain: str = "aml") -> list:
    """A domain's parties from Postgres. Postgres is the app's source of truth: the app
    is a production service backed by a database, NOT the JSON files. If Postgres is down or the
    corpus mirror is not yet loaded, raise 503 rather than silently reading files - a fallback
    would mask a misconfigured deployment and serve data the operator did not load. Run the loader
    (`scripts/load_corpus_db.py`) as a deploy/ingest step to populate the mirror before serving."""
    from amlguard import db
    if not db.available():
        raise DataUnavailable(
            f"Postgres is not reachable at the configured URL; the app requires it. "
            f"Start docker/app/postgres and run scripts/load_corpus_db.py --domain {domain}."
        )
    rows = db.read_parties(domain)
    if not rows:
        # Only AML ships a party corpus in this edition; healthcare/support demos run on their own
        # inline fixtures (de-id CSV, dual-gate message), not a party roster. Say so precisely so a
        # healthcare/support chat turn fails with an accurate message instead of pointing at a
        # loader command that has nothing to load.
        if domain == "aml":
            raise DataUnavailable(
                "The 'aml' corpus is not loaded in Postgres. "
                "Run: python scripts/load_corpus_db.py --domain aml"
            )
        raise DataUnavailable(
            f"Live chat is AML-only in this edition: the {domain!r} domain ships no party corpus "
            f"(its de-id / dual-gate demos run on their own fixtures). Use the AML domain for the "
            f"live chatbot, or the {domain!r} batch stepper for that domain's protection demo."
        )
    return rows


@app.get("/api/health")
def health() -> dict:
    """Liveness plus which model live chat will use and whether it is ready, so the UI can tell a
    user without cloud credentials that live turns will run on the local open-source model (or what
    to do if it is not set up yet). No LLM call is made; the model check is a cheap Ollama probe."""
    from amlguard import llm, settings
    live = {"ready": True}
    try:
        model_key, reason = resolve_ui_model()
        spec = ModelRegistry.load().get(model_key)
        live = {"model": model_key, "provider": spec.provider, "reason": reason}
        if spec.provider == "ollama":
            reachable = llm.ollama_reachable()
            present = reachable and llm.ollama_has_model(spec.model_id)
            live["ready"] = bool(present)
            live["ollama_reachable"] = reachable
            live["model_pulled"] = present
            live["auto_pull"] = settings.auto_pull_model()
        else:
            # Hosted: reachable in principle; the real check is the first live call.
            live["ready"] = True
    except Exception as exc:  # noqa: BLE001, health must never 500
        # Log the detail server-side; the endpoint is unauthenticated, so no exception text
        # (which can carry paths and stack context) leaves the process.
        logging.getLogger("amlguard.ui").warning("live-chat health probe failed: %s", exc)
        live = {"ready": False, "error": "live-chat status unavailable"}
    # Which domains actually support a live chatbot turn (AML only in this edition). The UI reads
    # this to offer Live only where it works, instead of promising it for every domain and then
    # 503-ing. Single source of truth: Domain.supports_live_chat.
    live_domains = [n for n in domain_names() if get_domain(n).supports_live_chat]
    return {
        "ok": True,
        "domains": list(domain_names()),
        "live_domains": live_domains,
        "live_chat": live,
    }


@app.get("/api/domains")
def domains() -> list[dict]:
    return [
        {"name": d.name, "label": d.label,
         "guardrail_model": d.injection_processor,
         "fields": list(d.record_fields)}
        for d in (get_domain(n) for n in domain_names())
    ]


# The ONLY party columns this preview endpoint may return. It exists so the demo can list who is
# in the corpus, NOT to hand out identity data: emitting clear ssn/credit_card/date_of_birth/
# account_number/tax_id/email/phone/address here would contradict the whole "protect data in use"
# thesis (a reviewer found this endpoint dumping them). Anything sensitive is dropped at the seam,
# regardless of auth. A caller that needs protected identity data goes through the tokenized
# serving path, never this list view.
_PARTY_SAFE_COLUMNS = ("party_id", "party_type", "full_name", "jurisdiction",
                       "risk_rating", "is_pep")


def _safe_party(row: dict) -> dict:
    return {k: row.get(k) for k in _PARTY_SAFE_COLUMNS if k in row}


@app.get("/api/corpus/parties")
def corpus_parties(domain: str = "aml", limit: int = 50,
                   x_amlguard_token: str | None = Header(default=None)) -> dict:
    """A page of parties for the web app, from Postgres (the app's source of truth).

    Postgres-only: 503 if the DB is down or the corpus is not loaded (no JSON fallback - this is a
    production service backed by a database). Capped at `limit` rows (max 500): a UI preview, not a
    bulk export.

    Two protections, both load-bearing: (1) the same optional shared secret that gates the billed
    endpoints (`_check_auth`), and (2) a hard column allow-list (`_PARTY_SAFE_COLUMNS`) so clear
    identity fields (ssn, credit_card, dob, accounts, tax_id, email, phone, address) NEVER leave
    this endpoint even to an authenticated caller - the pipeline exists to protect exactly those."""
    _check_auth(x_amlguard_token)
    limit = max(1, min(limit, 500))
    rows = _load_parties(domain)  # raises DataUnavailable(503) if Postgres down / corpus unloaded
    return {"domain": domain, "source": "postgres", "count": len(rows),
            "parties": [_safe_party(r) for r in rows[:limit]]}


@app.get("/api/pipeline")
def pipeline(domain: str = "aml") -> dict:
    """The architecture flow the batch stepper renders for a domain, plus the run's provenance.

    Each domain has its own stage list (AML's 7-stage curve, healthcare's HIPAA de-id, support's
    dual-gate). Unknown domains fall back to AML."""
    try:
        from amlguard.tracking import corpus_source_fingerprint
        fp = corpus_source_fingerprint(ROOT / "data" / "corpus")
    except Exception:  # noqa: BLE001
        fp = "unknown"
    stages = DOMAIN_STAGES.get(domain, _AML_STAGES)
    return {"domain": domain, "use_case": "batch", "corpus_fingerprint": fp,
            "model": DOMAIN_MODEL.get(domain, "bedrock-sonnet-5"), "stages": stages}


@app.get("/api/chat/replay")
def chat_replay(domain: str = "aml") -> dict:
    """A committed multi-turn chat transcript for a domain (the Assistant view's Replay mode).

    Produced by `scripts/demo_chat.py --domain <d> --json` from a real Claude Sonnet 5 run over the
    tokenized corpus. Each turn stores the tokenized reply (`reply_over_tokens`, PII-shape redacted -
    NEVER cleartext) plus the per-turn protection boundary. 404 (with a clear message) if the domain
    has no committed transcript, so the frontend can prompt to generate one."""
    path = ROOT / "data" / "eval" / domain / "chat_replay.json"
    if not path.exists():
        raise HTTPException(
            404,
            f"no committed chat transcript for {domain!r}; generate one with "
            f"`python scripts/demo_chat.py --domain {domain} --json`",
        )
    return json.loads(path.read_text())


@app.get("/api/journey")
def journey(domain: str = "aml") -> dict:
    """The 'data journey': one representative synthetic record traced through the pipeline stages,
    as clear input -> tokenized -> reasoned-over -> role-gated output. Built by
    `scripts/build_journey.py` from the committed synthetic corpus + protected artifacts.

    This is the ONLY endpoint that returns a clear value, and it is a single curated, synthetic
    record (no real person), paired with its protected form, so the UI can show the transformation
    the whole product is about. 404 if the artifact has not been built."""
    path = ROOT / "data" / "eval" / domain / "journey.json"
    if not path.exists():
        # Fall back to AML's journey so the section always renders something illustrative.
        path = ROOT / "data" / "eval" / "aml" / "journey.json"
    if not path.exists():
        raise HTTPException(404, "no journey artifact; run `python scripts/build_journey.py`")
    return json.loads(path.read_text())


@app.get("/api/artifact/{name}")
def artifact(name: str) -> Any:
    """Serve a committed result artifact by short name (replay data source)."""
    mapping = {
        "training": "eval/training.json",
        "attacks": "eval/attacks.json",
        "erasure": "eval/semantic_erasure.json",
        "frontier": "eval/protection_methods.json",
        "hybrid_none": "eval/hybrid_none.json",
        "hybrid_quasi": "eval/hybrid_quasi.json",
        "ingest": "protected/ingestion_summary.json",
        "healthcare_deid": "eval/healthcare/deidentify.json",
        "support_gates": "eval/support/dual_gate.json",
    }
    if name not in mapping:
        raise HTTPException(404, f"unknown artifact {name!r}; known: {sorted(mapping)}")
    return _read(mapping[name])


@app.get("/api/eval-curve")
def eval_curve() -> dict:
    """The per-scope LLM investigation curve (mean checkpoint score), read from the run."""
    d = EVAL / "bedrock-sonnet-5"
    out = {}
    for f in d.glob("*.json"):
        if f.name in ("tasks.json", "comparison.json"):
            continue
        j = json.loads(f.read_text())
        if not j.get("llm_stats", {}).get("billed_calls"):
            continue
        out[j["scope"]] = {
            "mean": j.get("mean_checkpoint_score", j.get("mean_score")),
            "verifiable": j.get("verifiable_mean_score"),
            "models_used": j.get("models_used"),
        }
    return out


def _runs_for_scope(scope: str) -> set[str]:
    """Run UUIDs whose `scope` param equals `scope`, read from the MLflow SQLite backend.

    Read-only, no live MLflow server. This is what makes /api/plot scope-CORRECT: without it the
    endpoint globbed every run's PNG and served the newest by mtime, so scope A's SHAP could be
    returned under scope B, a silently wrong plot in a demo about measuring scopes.
    """
    import sqlite3
    db = _mlflow_store() / "mlflow.db"
    if not db.exists():
        return set()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT run_uuid FROM params WHERE key='scope' AND value=?", (scope,)
        ).fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}


@app.get("/api/plot/{scope}/{name}")
def plot(scope: str, name: str) -> FileResponse:
    """Serve a training plot (SHAP/PR/ROC) for a scope from the MLflow artifact store.

    Resolves the requested scope to its run(s) via the MLflow backend and serves ONLY a PNG that
    lives under one of those runs, so a plot is never served under the wrong scope. If provenance
    can't be established, 404 (never a cross-scope fallback). Read-only; replays the run's output.
    """
    allowed = {"shap_bar", "shap_beeswarm", "shap_waterfall", "shap_dependence",
               "precision_recall", "roc", "score_histogram"}
    if name not in allowed:
        raise HTTPException(404, f"unknown plot {name!r}")
    runs = _runs_for_scope(scope)
    if not runs:
        raise HTTPException(404, f"no run recorded for scope {scope!r}")
    # Plots live under <store>/artifacts/<exp>/<run>/artifacts/plots/<name>.png. Keep only PNGs
    # whose <run> segment is a run of THIS scope; newest of those wins.
    candidates = [
        p for p in (_mlflow_store() / "artifacts").rglob(f"plots/{name}.png")
        if any(part in runs for part in p.parts)
    ]
    if not candidates:
        raise HTTPException(404, f"no {name} plot found for scope {scope!r}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return FileResponse(candidates[0], media_type="image/png")


# ── Live primitives (cheap, local, real) ──────────────────────────────────────────────────────
class TurnRequest(BaseModel):
    message: str
    conversation_id: str = "demo"
    domain: str = "aml"
    role: str = "investigator"


# Process-wide singletons, built once and reused across requests. A fresh Protector per request
# would open a fresh Protegrity session == a POST /auth/login per turn, and /auth/login is
# separately rate-limited (its 429 is misreported as bad credentials with no safe retry): under
# rapid turns that throttles the whole demo. One login per process is a hard project invariant.
# The guardrail is cached per domain (it parses parties.json + builds the forbidden-value index),
# and the LLM client is cached per model with caching OFF (a chatbot must not serve one user's
# reply to another) and fallback OFF (a paid demo must not silently swap models).
_PROTECTOR = None
_GUARDRAILS: dict = {}
_LLM = None
_ROSTERS: dict = {}  # per-domain hybrid roster (ORGANIZATION + identifiers discovery misses)

# Two abuse rails on the PAID chat endpoint (a turn is a real billed LLM call):
#  - an optional shared secret (AMLGUARD_UI_API_TOKEN): unset for the loopback demo, required
#    for any exposure beyond localhost, so an open port can't bill the account.
#  - a per-process turn ceiling (AMLGUARD_UI_MAX_TURNS): a hard backstop beside the SDK spend cap
#    so a stuck client can't loop the endpoint unbounded. Resets on restart; 0 disables.
_TURN_COUNT = 0


def _check_auth(token: str | None) -> None:
    from amlguard import settings
    expected = settings.ui_api_token()
    if expected and token != expected:
        raise HTTPException(401, "missing or invalid X-AMLGuard-Token")


def _charge_turn() -> None:
    global _TURN_COUNT
    from amlguard import settings
    cap = settings.ui_max_turns()
    if cap and _TURN_COUNT >= cap:
        raise HTTPException(
            429, f"live-turn ceiling reached ({cap}); restart the API or raise AMLGUARD_UI_MAX_TURNS")
    _TURN_COUNT += 1


def _get_protector():
    global _PROTECTOR
    if _PROTECTOR is None:
        from amlguard.protect import Protector
        _PROTECTOR = Protector()
    return _PROTECTOR


def resolve_ui_model() -> tuple[str, str]:
    """Pick the model the UI's live chat should use, so a forked repo runs with or without cloud
    credentials. Precedence:

      1. AMLGUARD_UI_MODEL, if set  (explicit operator override; wins unconditionally)
      2. the hosted model (AMLGUARD_HOSTED_MODEL, default bedrock-sonnet-5), when AWS credentials
         are present  (matches the committed evaluation artifacts)
      3. the open-source local model (AMLGUARD_LOCAL_MODEL, default llama3.2) via Ollama

    Returns (model_key, reason) where reason is a short human-readable why, surfaced in logs and the
    /api/health payload. When the local model is chosen and AMLGUARD_AUTO_PULL_MODEL is on, the model
    is pulled on first use (the one-time setup); otherwise a missing model is reported, not fetched."""
    from amlguard import llm, settings

    override = settings.ui_model()
    if override:
        # Validate the override the same way the local path does, so a typo'd model name fails with
        # a clear "unknown model, declared: …" up front rather than only when the client is built.
        try:
            ModelRegistry.load().get(override)
            return override, f"AMLGUARD_UI_MODEL={override}"
        except llm.LLMConfigError as exc:
            # Health is unauthenticated: log the detail, keep exception text out of the response.
            logging.getLogger("amlguard.ui").warning("AMLGUARD_UI_MODEL invalid: %s", exc)
            return override, f"AMLGUARD_UI_MODEL={override} (invalid; see server log)"

    if settings.bedrock_available():
        return settings.hosted_ui_model(), "AWS credentials present"

    # No hosted credentials: fall back to the open-source local model.
    local = settings.local_model()
    spec = ModelRegistry.load().get(local)  # resolves the Ollama model_id (e.g. llama3.2:latest)
    ensure_ok = llm.ensure_ollama_model(spec.model_id, auto_pull=settings.auto_pull_model())
    if not ensure_ok and not llm.ollama_reachable():
        reason = ("no AWS credentials and Ollama is not reachable at "
                  f"{settings.ollama_url()} (install from https://ollama.com and start it)")
    elif not ensure_ok:
        reason = (f"no AWS credentials; local model '{spec.model_id}' is not pulled "
                  f"(run `make setup-local-model` or set AMLGUARD_AUTO_PULL_MODEL=true)")
    else:
        reason = f"no AWS credentials; using local model '{local}'"
    return local, reason


def _get_llm():
    global _LLM
    if _LLM is None:
        from amlguard.llm import get_llm
        model_key, reason = resolve_ui_model()
        logging.getLogger("amlguard.ui").info("Live chat model: %s (%s)", model_key, reason)
        # allow_fallback stays False so a live turn is never silently re-attributed to a different
        # model than the one the UI reports; the selection above already chose a reachable model.
        _LLM = get_llm(model_key, trace_component="serving-ui",
                       enable_cache=False, allow_fallback=False)
    return _LLM


def _get_roster(domain_name: str = "aml"):
    """The hybrid roster for a domain, built once from that domain's clear party list. Load-bearing
    on inbound protect: discovery's ORGANIZATION recall is ~0, so without the roster an org name a
    user types is never tokenized and reaches the model / Langfuse in the clear.

    Keyed by domain so a healthcare/support chat turn does not silently seed its protection from the
    AML party names. Postgres-only: _load_parties raises 503 if that domain's corpus is
    down / unloaded, so the turn fails cleanly rather than protecting against the wrong entity set."""
    if domain_name not in _ROSTERS:
        from amlguard.roster import roster_from_parties

        parties = _load_parties(domain_name)
        # Fold in the domain's OWN high-sensitivity identifiers that the standard roster fields
        # don't cover (healthcare mrn/insurance_id, support order_id). roster_from_parties reads the
        # generic PII fields (name, ssn, email, phone, account, address, dob); a domain-specific id
        # like an MRN is not in that set, so without this a typed MRN would reach the model in the
        # clear. Keyed off the domain's declared high_sensitivity_fields -> their discovery entity
        # type, so this stays declarative (add a field in domains.py, it's protected here too).
        domain = get_domain(domain_name)
        extra: dict[str, list[str]] = {}
        for field_name in domain.high_sensitivity_fields:
            entity_type = domain.record_fields.get(field_name)
            if not entity_type:
                continue
            values = [str(p.get(field_name) or "").strip() for p in parties]
            values = [v for v in values if len(v) >= 5]
            if values:
                extra.setdefault(entity_type, []).extend(values)
        _ROSTERS[domain_name] = roster_from_parties(parties, extra_values=extra or None)
    return _ROSTERS[domain_name]


def _domain_corpus_path(domain_name: str) -> Path:
    """The clear party corpus for a domain. Only AML ships a party corpus in this edition; a
    per-domain file (data/corpus/<domain>/parties.json) wins if present, else the AML corpus is
    used and the caller is responsible for not overstating cross-domain coverage."""
    per_domain = ROOT / "data" / "corpus" / domain_name / "parties.json"
    return per_domain if per_domain.exists() else ROOT / "data" / "corpus" / "parties.json"


def _get_guardrail(domain):
    if domain.name not in _GUARDRAILS:
        parties_path = _domain_corpus_path(domain.name)
        try:
            from amlguard.guardrail import Guardrail
            # for_corpus also installs trace-redaction from the domain's forbidden set.
            _GUARDRAILS[domain.name] = Guardrail.for_corpus(
                parties_path, probe=False, domain=domain,
            )
        except Exception:  # noqa: BLE001, egress guard is optional on the live path...
            # ...BUT trace-redaction must NOT be. If the guard cannot be built, the turn runs
            # WITHOUT an egress scan; without redaction it would ALSO trace unscrubbed bodies to
            # Langfuse. Install redaction from the same forbidden set directly, so a missing guard
            # never silently disables the Langfuse scrub. Best-effort: if even this fails, the LLM
            # client still redacts by its own token/tag rules, but we surface nothing to at-rest
            # storage that carries these exact clear identifiers.
            _install_fallback_redaction(domain, parties_path)
            _GUARDRAILS[domain.name] = None
    return _GUARDRAILS[domain.name]


def _install_fallback_redaction(domain, parties_path) -> None:
    """Seed Langfuse trace-redaction from the corpus even when the egress guard won't build."""
    try:
        from amlguard.guardrail import forbidden_values_from_parties
        from amlguard.observability import set_trace_redaction
        parties = json.loads(parties_path.read_text())
        forbidden = forbidden_values_from_parties(parties, tuple(domain.record_fields))
        set_trace_redaction(forbidden)
    except Exception:  # noqa: BLE001, redaction is a backstop; never break the turn over it
        pass


@app.post("/api/chat/turn")
def chat_turn(
    req: TurnRequest,
    x_amlguard_token: str | None = Header(default=None),
) -> dict:
    """One live protected chatbot turn, returning the reply AND the pipeline internals.

    Reuses the exact serving boundary (protect -> reason over tokens -> egress -> re-identify),
    so the internals tab shows the real TurnResult, not a mock. Protector/guardrail/LLM are
    process-wide singletons (see above) so a turn never re-logs-in to Protegrity. This IS a live
    LLM call (~$0.01) counted against the process spend cap. Gated by an optional shared secret
    and a per-process turn ceiling (see _check_auth / _charge_turn) so an exposed port can't bill
    the account unbounded.
    """
    from amlguard.reidentify import ROLES
    from amlguard.serving import ConversationSession

    _check_auth(x_amlguard_token)
    domain = get_domain(req.domain)
    # Fail fast, and BEFORE charging a turn, on a domain that structurally has no live chatbot in
    # this edition (only AML ships a live party corpus; see Domain.supports_live_chat). This is the
    # same limitation _load_parties would hit deeper in, surfaced here as a clean 503 the UI can show
    # verbatim — not a generic 502 from re-wrapping an exception thrown mid-turn.
    if not domain.supports_live_chat:
        raise DataUnavailable(
            f"Live chat is AML-only in this edition: the {domain.name!r} domain ships no party "
            f"corpus for a live turn (its protection is demonstrated by the Batch Analysis stepper "
            f"instead). Switch to the AML domain for the live assistant, or use Batch Analysis here."
        )
    _charge_turn()
    role = ROLES.get(req.role)
    if role is None:
        raise HTTPException(400, f"unknown role {req.role!r}; known: {sorted(ROLES)}")

    try:
        session = ConversationSession(
            protector=_get_protector(), llm=_get_llm(),
            conversation_id=req.conversation_id, role=role, domain=domain,
            guardrail=_get_guardrail(domain),
            # roster is load-bearing: the serving path protects EVERYTHING detected, and discovery
            # alone misses organizations. scope=None => every discovery hit blocks the roster, so
            # the roster fills only the true gaps (org names, missed identifiers).
            # Keyed on the turn's domain so a healthcare/support turn is protected against ITS own
            # entity set, never silently against the AML party names.
            roster=_get_roster(domain.name),
        )
        result = session.turn(req.message)
    except HTTPException:
        # An intentional, already-shaped HTTP error (e.g. DataUnavailable's 503 "live chat is
        # AML-only in this edition") must reach the UI with ITS OWN status and message, not be
        # re-wrapped as a generic 502 Bad Gateway. Only genuinely unexpected exceptions below
        # become a 502.
        raise
    except Exception as exc:  # noqa: BLE001, surface a genuinely-unexpected failure cleanly
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
    return {
        "reply": result.reply,
        "internals": {
            "protected_input": result.protected_input,
            "model_saw": result.protected_input,   # the tokenized prompt the LLM received
            "raw_completion": result.raw_completion,
            "entities_protected": result.entities_protected,
            "revealed": result.revealed,
            "egress_blocked": result.egress_blocked,
            "egress_detail": result.egress_detail,   # per-processor + conversation-level verdict
            "guardrail_model": domain.injection_processor,
            "role": req.role,
            "domain": req.domain,
        },
        "ok": result.ok,
        "error": result.error,
    }


# ── Live batch execution (Slice 3) — real stage runs behind cost rails ────────
# The batch stepper's Live mode runs a stage FOR REAL and shows the freshly-produced artifact.
# Safety rails, all load-bearing:
#   * estimate-first: a paid stage returns its cost estimate + a confirm_token and runs NOTHING;
#     execution requires the client to POST the token back (a human clicked "Confirm $X").
#   * run-lock: acquires the same process lock the CLI uses; 409 if a run is in progress, so a
#     UI run can never corrupt shared state under a concurrent CLI/UI run.
#   * _live isolation: every stage writes to data/eval/_live/<run_id>/, NEVER the committed
#     artifacts, so a live demo can never clobber the verified results the Replay mode shows.
#   * spend cap: paid stages route through the same AMLGUARD_MAX_SPEND_USD reservation the CLI uses.
class BatchRunRequest(BaseModel):
    domain: str = "aml"
    stage: str                    # the stage id from /api/pipeline
    scope: str = "none"           # which protection scope to run the stage on
    confirm_token: str | None = None   # echo the estimate's token to authorize a paid run


def _live_dir(run_id: str) -> Path:
    d = ROOT / "data" / "eval" / "_live" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stage_estimate(domain: str, stage: dict, scope: str) -> dict:
    """Cost estimate for one live stage. $0 for local stages; a real token estimate for paid ones."""
    if not stage.get("paid"):
        return {"cost_usd": 0.0, "paid": False}
    # Paid = one scope's worth of LLM investigation. Reuse the eval cost model (per-call averages).
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("estc", str(ROOT / "scripts" / "estimate_cost.py"))
        estc = importlib.util.module_from_spec(spec); spec.loader.exec_module(estc)
        from amlguard.llm import ModelRegistry
        model_spec = ModelRegistry.load().get("bedrock-sonnet-5")
        est = estc.estimate(model_spec, scopes=1, tasks=12, judge=True)
        return {"cost_usd": round(est.get("cost_usd", 0.0), 2), "paid": True,
                "calls": est.get("calls")}
    except Exception:  # noqa: BLE001, fall back to a conservative flat estimate
        return {"cost_usd": 0.35, "paid": True, "calls": None}


# Confirm tokens are server-minted, single-use, and short-lived so a client CANNOT self-mint one
# from the public (domain, stage, scope, cost) inputs and auto-confirm a paid run without a human
# ever seeing the cost. Each estimate response issues a fresh random nonce, bound to the exact
# stage+cost it quoted; executing consumes it (pop). A replayed or guessed token is rejected.
_CONFIRM_TTL_SEC = 300
_PENDING_CONFIRMS: dict[str, dict] = {}  # token -> {"key": ..., "expires": monotonic_deadline}


def _issue_confirm_token(domain: str, stage_id: str, scope: str, cost: float) -> str:
    import secrets
    import time
    token = secrets.token_urlsafe(24)
    _PENDING_CONFIRMS[token] = {
        "key": f"{domain}:{stage_id}:{scope}:{cost:.2f}",
        "expires": time.monotonic() + _CONFIRM_TTL_SEC,
    }
    # Opportunistically drop expired entries so the map can't grow unbounded on an exposed port.
    now = time.monotonic()
    for t in [t for t, v in _PENDING_CONFIRMS.items() if v["expires"] < now]:
        _PENDING_CONFIRMS.pop(t, None)
    return token


def _consume_confirm_token(token: str | None, domain: str, stage_id: str,
                           scope: str, cost: float) -> bool:
    """True iff `token` is a live, unexpired, single-use token this server issued for exactly this
    stage+cost. Consumes it (so it can't be replayed). Any mismatch/expiry/reuse returns False."""
    import time
    if not token:
        return False
    entry = _PENDING_CONFIRMS.pop(token, None)  # pop => single-use
    if entry is None:
        return False
    if entry["expires"] < time.monotonic():
        return False
    return entry["key"] == f"{domain}:{stage_id}:{scope}:{cost:.2f}"


@app.post("/api/batch/run-stage")
def batch_run_stage(req: BatchRunRequest,
                    x_amlguard_token: str | None = Header(default=None)) -> dict:
    """Run one batch stage live, behind the cost/lock/isolation rails documented above."""
    _check_auth(x_amlguard_token)
    stages = DOMAIN_STAGES.get(req.domain, _AML_STAGES)
    stage = next((s for s in stages if s["id"] == req.stage), None)
    if stage is None:
        raise HTTPException(404, f"unknown stage {req.stage!r} for domain {req.domain!r}")

    est = _stage_estimate(req.domain, stage, req.scope)

    # A PAID stage must be explicitly confirmed with a server-issued single-use token. On the first
    # call (or any call whose token isn't a live one we minted for exactly this stage+cost) we quote
    # the cost and issue a fresh token, running NOTHING. Only a call echoing that exact token — which
    # a human obtained by seeing the "$X — Confirm" box — proceeds.
    if est["paid"] and not _consume_confirm_token(
        req.confirm_token, req.domain, req.stage, req.scope, est["cost_usd"]
    ):
        token = _issue_confirm_token(req.domain, req.stage, req.scope, est["cost_usd"])
        return {"status": "confirm_required", "estimate": est, "confirm_token": token,
                "message": f"This stage runs a live LLM call (~${est['cost_usd']:.2f}). "
                           f"Re-submit with confirm_token to proceed."}

    # A fresh id per request so two live runs never share a _live/<id>/ dir and overwrite each
    # other's fixed-named artifacts. (persist.RUN_ID is a process-wide constant used for telemetry
    # correlation, not for filesystem isolation.)
    import uuid
    run_id = uuid.uuid4().hex[:12]

    # Acquire the run lock so a UI run never collides with a CLI run (or another UI run).
    from amlguard.persist import acquire_run_lock
    try:
        lock = acquire_run_lock(ROOT / "data")
    except RuntimeError:
        # acquire_run_lock raises RuntimeError on contention (persist.py); the CLI callers all
        # catch RuntimeError too. Map it to the documented 409 so a concurrent CLI/UI run gets a
        # clean "try again", not an unhandled 500.
        raise HTTPException(409, "another run holds the lock; try again when it finishes") from None
    try:
        out = _execute_stage(req.domain, stage, req.scope, run_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001, surface cleanly
        raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
    finally:
        import fcntl
        try:
            fcntl.flock(lock, fcntl.LOCK_UN); lock.close()
        except Exception:  # noqa: BLE001
            pass
    return {"status": "done", "run_id": run_id, "estimate": est, **out}


def _execute_stage(domain: str, stage: dict, scope: str, run_id: str) -> dict:
    """Run one stage for real, writing any artifact to the isolated _live/<run_id>/ dir."""
    live = _live_dir(run_id)
    run = stage.get("run")
    if run == "healthcare":
        import subprocess
        # The de-id script writes data/eval/healthcare/deidentify.json in place. It is deterministic
        # (same patient CSV -> identical output), but to GUARANTEE the committed artifact is never
        # altered by a live UI run, snapshot it first and restore it after copying the result to
        # _live. True isolation, not "it happens to be idempotent".
        src = ROOT / "data" / "eval" / "healthcare" / "deidentify.json"
        backup = src.read_text() if src.exists() else None
        subprocess.run([sys.executable, str(ROOT / "scripts" / "healthcare_deidentify.py")],
                       check=True, capture_output=True, env={**_os.environ, "AMLGUARD_NO_TRACKING": "1"})
        produced = src.read_text()
        (live / "healthcare_deidentify.json").write_text(produced)
        if backup is not None:
            src.write_text(backup)  # restore the committed artifact untouched
        return {"artifact": json.loads(produced), "wrote": "_live/%s/healthcare_deidentify.json" % run_id}
    if run == "support":
        import importlib.util
        spec = importlib.util.spec_from_file_location("sup", str(ROOT / "scripts" / "demo_support_gates.py"))
        sup = importlib.util.module_from_spec(spec); spec.loader.exec_module(sup)
        result = sup.run_dual_gate()
        (live / "support_dual_gate.json").write_text(json.dumps(result, indent=2))
        return {"artifact": result, "wrote": "_live/%s/support_dual_gate.json" % run_id}
    if run in ("embed", "present"):
        return {"artifact": None, "note": f"{stage['title']} is a local step; no artifact to write."}
    if run == "train":
        from amlguard.scopes import get_scope
        from amlguard.training import train_scope
        protected = ROOT / "data" / "protected" / get_scope(scope).slug
        if not (protected / "transactions.json").exists():
            raise HTTPException(400, f"scope {scope!r} not ingested; run Ingest first")
        r = train_scope(protected, ROOT / "data" / "corpus", scope)
        (live / f"train_{scope}.json").write_text(json.dumps(r.to_dict(), indent=2))
        return {"artifact": r.to_dict(), "wrote": f"_live/{run_id}/train_{scope}.json"}
    # ingest / retrieve / infer / egress are heavier; expose them as explicitly-not-wired here so
    # the endpoint never silently does something surprising. (Infer/egress are the paid ones and
    # would run one eval scope into _live; wiring them is the same pattern as train above.)
    raise HTTPException(501, f"live run for stage {stage['id']!r} is not wired yet in this build")
# No-cache on static assets: this is a demo/dev UI whose HTML/JS iterate, and a browser caching
# app.js by ETag serves a stale frontend after an edit (a real footgun during a live demo).
from starlette.staticfiles import StaticFiles as _StaticFiles  # noqa: E402


class _NoCacheStatic(_StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp


app.mount("/", _NoCacheStatic(directory=str(ROOT / "ui" / "web"), html=True), name="web")
