"""One place every runtime setting is read, so configuration is decoupled from code.

The endpoints, thresholds, and toggles the pipeline needs were each read with `os.getenv` at
their point of use, spread across ingest, guardrail, llm, observability, tracking, and
metrics_export. That is fine 12-factor practice, real environment always wins, but it left no
single place to see what is configurable, and several values (the discovery score threshold, the
guardrail model threshold) were plain constants that could not be tuned without editing Python.

This module is the settings surface: every setting resolved once, from the environment, with a
documented default. Modules import the value they need from here instead of calling `os.getenv`
themselves, so `.env.example` and this file together are the complete, authoritative list of
what a deployment can change. Kept dependency-free (no pydantic/dynaconf) for the same reason
`env.py` is: a judge should not install a config library before the pipeline runs.

`.env` is loaded by `env.load_dotenv` before this is read (every entrypoint does so); values are
resolved lazily via functions, not frozen at import, so a test can monkeypatch the environment.
"""

from __future__ import annotations

import os


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _flag(name: str) -> bool:
    return os.getenv(name) == "1"


def _get_bool(name: str, default: bool) -> bool:
    """A forgiving boolean for user-facing config flags: accepts 1/true/yes/on (any case) as true
    and 0/false/no/off as false. Unset or unrecognised returns `default`."""
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


# --- Protegrity services ----------------------------------------------------------------------
def discovery_url() -> str:
    return _get("AMLGUARD_DISCOVERY_URL",
                "http://localhost:8580/pty/data-discovery/v2/classify/text")


def discovery_threshold() -> float:
    """Minimum discovery score to treat a span as a detected entity (was a hardcoded 0.6)."""
    return _get_float("AMLGUARD_DISCOVERY_THRESHOLD", 0.6)


def discovery_workers() -> int:
    return _get_int("AMLGUARD_DISCOVERY_WORKERS", 4)


def guardrail_url() -> str:
    return _get("AMLGUARD_GUARDRAIL_URL",
                "http://localhost:8581/pty/semantic-guardrail/v1.1/conversations/messages/scan")


def anonymization_url() -> str:
    """Anonymization service (k-anon/DP/risk). Was hardcoded as ANON_EP in two scripts."""
    return _get("AMLGUARD_ANONYMIZATION_URL", "http://localhost:8085/pty/anonymization/v3")


def synthetic_url() -> str:
    """Synthetic-data service (vine copula). Was hardcoded in compare_protection_methods."""
    return _get("AMLGUARD_SYNTHETIC_URL", "http://localhost:8000/pty/syntheticdata/v2")


def ui_api_base() -> str:
    """Base URL the demo UI's browser client calls. Injected into the page at serve time so
    the frontend endpoint is config, not a literal baked into app.js."""
    return _get("AMLGUARD_UI_API_BASE", "http://localhost:8600")


def ui_api_token() -> str | None:
    """Optional shared secret guarding the UI's PAID endpoints (a chat turn is a real LLM call).

    Unset by default so the local judge demo runs frictionlessly on loopback. Set it for any
    exposure beyond loopback: the browser client sends it, and the paid endpoint rejects callers
    that don't. This is a spend/abuse gate, not a user-identity system.
    """
    return os.getenv("AMLGUARD_UI_API_TOKEN") or None


def ui_role_tokens() -> dict[str, str]:
    """Optional map of API token -> role name, so the CALLER's role is proven by their token
    rather than chosen freely in the request body. Format: ``AMLGUARD_UI_ROLE_TOKENS`` =
    ``role:token,role:token`` (e.g. ``auditor:tok-a,investigator:tok-i``).

    Unset by default (the local judge demo lets you switch roles freely to see the contrast). When
    set, a matching token pins that role and overrides the request's ``role`` field — the
    application-enforced analogue of a Protegrity policy role bound to an authenticated principal.
    Returns ``{token: role}`` for O(1) lookup.
    """
    raw = os.getenv("AMLGUARD_UI_ROLE_TOKENS") or ""
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            role, token = pair.split(":", 1)
            if role.strip() and token.strip():
                out[token.strip()] = role.strip()
    return out


def ui_max_turns() -> int:
    """Hard ceiling on live chat turns per API process, a second rail beside the spend cap.

    Every turn is a billed LLM call; without a ceiling a stuck client (or an open endpoint)
    could bill unbounded. Restarting the process resets it; 0 disables the cap.

    Uses `_get_int` (not a raw `int()`) so a malformed value falls back to the default like every
    other numeric setting — the value is read on the hot `/api/chat/turn` path, where a ValueError
    would 500 the endpoint on the first turn instead of degrading gracefully.
    """
    return _get_int("AMLGUARD_UI_MAX_TURNS", 200)


# --- Model / LLM ------------------------------------------------------------------------------
def _in_container() -> bool:
    """Best-effort: are we running inside a Docker container? Used only to choose the default
    Ollama host (a container's `localhost` is itself, not the host). An explicit OLLAMA_URL always
    wins, so a wrong guess here is never load-bearing."""
    from pathlib import Path
    if os.getenv("AMLGUARD_IN_CONTAINER") == "1":
        return True
    if Path("/.dockerenv").exists():
        return True
    try:
        return "docker" in Path("/proc/1/cgroup").read_text()
    except Exception:  # noqa: BLE001, /proc may be absent (non-Linux) — just means "not detected"
        return False


def ollama_url() -> str:
    """Where Ollama listens. An explicit OLLAMA_URL always wins. Otherwise the default depends on
    WHERE we run: on a host, `localhost:11434`; inside a container, the host's Ollama via
    `host.docker.internal:11434` (a container's own localhost is not the host). Deriving the
    container default here — rather than hardcoding it in compose `environment:` — means a user's
    OLLAMA_URL in `.env` is honoured whether they start the stack via `make` or raw `docker compose`,
    with no `environment:` entry to clobber it (the precedence pitfall in the review's F1)."""
    explicit = os.getenv("OLLAMA_URL")
    if explicit:
        return explicit
    return "http://host.docker.internal:11434" if _in_container() else "http://localhost:11434"


def default_model() -> str | None:
    return os.getenv("AMLGUARD_MODEL")


def max_spend_usd() -> float:
    return _get_float("AMLGUARD_MAX_SPEND_USD", 5.0)


# --- Live-serving model selection (UI chat) ---------------------------------------------------
# The UI's live chat picks a model at request time so a forked repo without cloud credentials
# still runs: an explicit override wins; otherwise a hosted model is used when its credentials
# are present; otherwise the open-source local model (Ollama) is used. All three are env-driven.
def ui_model() -> str | None:
    """Explicit model for the UI's live chat (a `config/models.yaml` key). When set it wins over
    the auto-selection below, so an operator can pin exactly which model live turns use."""
    return os.getenv("AMLGUARD_UI_MODEL") or None


def local_model() -> str:
    """The open-source model the UI falls back to when no hosted credentials are present. Must be
    an Ollama entry in config/models.yaml. Defaults to the small, laptop-friendly `llama3.2`."""
    return _get("AMLGUARD_LOCAL_MODEL", "llama3.2")


def hosted_ui_model() -> str:
    """The hosted model the UI prefers when cloud credentials ARE present (Bedrock by default,
    matching the committed evaluation artifacts)."""
    return _get("AMLGUARD_HOSTED_MODEL", "bedrock-sonnet-5")


def auto_pull_model() -> bool:
    """Whether the app may pull the local model automatically on first use (the one-time setup).
    Off by default: a download is a side effect a user should opt into. `make setup-local-model`
    is the explicit alternative."""
    return _get_bool("AMLGUARD_AUTO_PULL_MODEL", False)


def bedrock_available() -> bool:
    """Heuristic: are AWS credentials present so the Bedrock provider can authenticate? Checks the
    usual env vars and a shared-credentials/profile file. Used only to decide whether to fall back
    to the local model; the real check is the live preflight call, which fails cleanly either way."""
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return True
    if os.getenv("AWS_PROFILE") or os.getenv("AWS_ROLE_ARN"):
        return True
    from pathlib import Path
    home = Path.home()
    cred = os.getenv("AWS_SHARED_CREDENTIALS_FILE") or str(home / ".aws" / "credentials")
    cfg = os.getenv("AWS_CONFIG_FILE") or str(home / ".aws" / "config")
    return Path(cred).exists() or Path(cfg).exists()


# --- Observability ----------------------------------------------------------------------------
def mlflow_uri() -> str:
    return _get("AMLGUARD_MLFLOW_URI", "http://localhost:5001")


def mlflow_store_dir() -> str | None:
    """Filesystem location of the local MLflow store (backend db + artifacts), or None for the
    default `docker/observability/mlflow/store`. The store lives under docker/observability/ with the other
    two planes; override for a relocated/remote layout."""
    return os.getenv("AMLGUARD_MLFLOW_STORE_DIR") or None


def postgres_url() -> str:
    """Connection URL for the app Postgres (the corpus mirror), on host port 5433.

    Default targets the local `docker/app/postgres` container. The library read helpers in
    db.py are fail-soft (return None), but the online app treats Postgres as a REQUIRED dependency
    and returns 503 with no JSON fallback - run scripts/load_corpus_db.py as a deploy
    step before serving. Override for a managed database.

    (MLflow's backend deliberately stays on the SQLite store, not this Postgres: the
    model-registry alias API is broken against Postgres in the pinned MLflow version, and the
    champion registry must keep working. Only the corpus mirror lives here.)"""
    return _get("AMLGUARD_POSTGRES_URL", "postgresql://amlguard:amlguard@localhost:5433/amlguard")


def langfuse_host() -> str:
    return _get("LANGFUSE_HOST", "http://127.0.0.1:3000")


def langfuse_timeout() -> int:
    return _get_int("LANGFUSE_TIMEOUT", 20)


def pushgateway() -> str:
    return _get("AMLGUARD_PUSHGATEWAY", "localhost:9093")


def prometheus_query_url() -> str:
    """The Prometheus HTTP query API (read side), distinct from the pushgateway (write side, 9093).
    Used by observability_report.py to read back the metrics a run pushed."""
    return _get("AMLGUARD_PROMETHEUS_URL", "http://localhost:9092")


def log_level() -> str:
    return _get("AMLGUARD_LOG_LEVEL", "INFO")


# --- Telemetry kill switches (all default OFF -> telemetry ON) ---------------------------------
def no_tracking() -> bool:
    return _flag("AMLGUARD_NO_TRACKING")


def no_tracing() -> bool:
    return _flag("AMLGUARD_NO_TRACING")


def no_metrics() -> bool:
    return _flag("AMLGUARD_NO_METRICS")


# The complete list of settings this project reads, for `.env.example` generation and a test
# that pins the two in sync. (var name, whether it has a safe default / is optional.)
KNOWN_SETTINGS: tuple[tuple[str, bool], ...] = (
    ("DEV_EDITION_EMAIL", False),
    ("DEV_EDITION_PASSWORD", False),
    ("DEV_EDITION_API_KEY", False),
    ("AMLGUARD_DISCOVERY_URL", True),
    ("AMLGUARD_DISCOVERY_THRESHOLD", True),
    ("AMLGUARD_DISCOVERY_WORKERS", True),
    ("AMLGUARD_GUARDRAIL_URL", True),
    ("AMLGUARD_ANONYMIZATION_URL", True),
    ("AMLGUARD_SYNTHETIC_URL", True),
    ("AMLGUARD_PROMETHEUS_URL", True),
    ("AMLGUARD_UI_API_BASE", True),
    ("AMLGUARD_UI_API_TOKEN", True),
    ("AMLGUARD_UI_MAX_TURNS", True),
    ("OLLAMA_URL", True),
    ("AMLGUARD_IN_CONTAINER", True),
    ("AMLGUARD_MODEL", True),
    ("AMLGUARD_UI_MODEL", True),
    ("AMLGUARD_LOCAL_MODEL", True),
    ("AMLGUARD_HOSTED_MODEL", True),
    ("AMLGUARD_AUTO_PULL_MODEL", True),
    ("AMLGUARD_MAX_SPEND_USD", True),
    ("AMLGUARD_LLM_TIMEOUT", True),
    ("ANTHROPIC_API_KEY", True),
    ("OPENAI_API_KEY", True),
    ("AWS_DEFAULT_REGION", True),
    ("AMLGUARD_MLFLOW_URI", True),
    ("AMLGUARD_MLFLOW_STORE_DIR", True),
    ("AMLGUARD_POSTGRES_URL", True),
    ("LANGFUSE_HOST", True),
    ("LANGFUSE_PUBLIC_KEY", True),
    ("LANGFUSE_SECRET_KEY", True),
    ("LANGFUSE_TIMEOUT", True),
    ("AMLGUARD_PUSHGATEWAY", True),
    ("AMLGUARD_LOG_LEVEL", True),
    ("AMLGUARD_NO_TRACKING", True),
    ("AMLGUARD_NO_TRACING", True),
    ("AMLGUARD_NO_METRICS", True),
    ("DOTENV_PATH", True),
)
