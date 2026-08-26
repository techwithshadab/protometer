"""Rails on the demo API: the paid endpoint's auth gate + turn ceiling, and scope-correct plots.

These pin three round-3 review fixes so they can't regress:
  * /api/chat/turn is a BILLED LLM call; an exposed port must not be able to bill the account.
    A shared secret (when configured) and a per-process turn ceiling both gate it, and both are
    checked BEFORE any Protegrity/LLM work so a rejected call costs nothing.
  * /api/plot resolves a scope to its MLflow run(s) and serves only that scope's PNG, never the
    newest-by-mtime plot of some other scope.

No hosted calls: the turn is stubbed / rejected before the real session is built, and the plot
test only exercises the SQLite scope->run resolver against the committed store.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


def _load_app(monkeypatch, **env):
    """Import ui/api/app.py fresh with the given env, so module-level settings take effect."""
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # The app lives outside the importable package; load it by path with a fresh module each time
    # so the turn counter and settings are not shared across tests.
    import importlib.util
    spec = importlib.util.spec_from_file_location("_ui_app_under_test", ROOT / "ui" / "api" / "app.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_chat_turn_requires_token_when_configured(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_UI_API_TOKEN="s3cret", AMLGUARD_NO_TRACKING="1",
                    AMLGUARD_NO_TRACING="1")
    client = TestClient(mod.app)
    # No token -> 401, and crucially BEFORE any Protegrity login or LLM call (a wrong-credential
    # 401 here would otherwise be a billed round-trip).
    r = client.post("/api/chat/turn", json={"message": "hi"})
    assert r.status_code == 401


def test_chat_turn_open_on_loopback_when_no_token(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    # Auth is a no-op when unset; assert the gate itself does not reject (it will fail later on the
    # real session, which we don't exercise here — we only prove the gate is open).
    mod._check_auth(None)  # must not raise


def test_turn_ceiling_rejects_past_cap(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_UI_MAX_TURNS="2", AMLGUARD_NO_TRACKING="1",
                    AMLGUARD_NO_TRACING="1")
    mod._charge_turn()
    mod._charge_turn()
    with pytest.raises(Exception) as exc:  # HTTPException(429)
        mod._charge_turn()
    assert getattr(exc.value, "status_code", None) == 429


def test_turn_ceiling_disabled_when_zero(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_UI_MAX_TURNS="0", AMLGUARD_NO_TRACKING="1",
                    AMLGUARD_NO_TRACING="1")
    for _ in range(50):
        mod._charge_turn()  # never raises when the cap is disabled


def test_plot_resolver_is_scope_specific(monkeypatch):
    """_runs_for_scope must return only runs whose `scope` param matches, never a cross-scope set."""
    from amlguard.tracking import DEFAULT_TRACKING_DIR
    if not (DEFAULT_TRACKING_DIR / "mlflow.db").exists():
        pytest.skip("no local MLflow store to resolve against")
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    none_runs = mod._runs_for_scope("none")
    quasi_runs = mod._runs_for_scope("quasi")
    # Both should resolve to at least one run, and the two sets must be disjoint (a run has one
    # scope). A non-disjoint result would mean the resolver is scope-blind.
    if none_runs and quasi_runs:
        assert none_runs.isdisjoint(quasi_runs)
    assert mod._runs_for_scope("no-such-scope") == set()


def test_corpus_parties_503_when_postgres_down(monkeypatch):
    """Postgres is the app's source of truth: NO JSON fallback. When Postgres is
    unreachable, /api/corpus/parties returns 503, not file data - a fallback would mask a
    misconfigured deployment."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi_testclient.TestClient
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    from amlguard import db as _db
    monkeypatch.setattr(_db, "available", lambda: False)
    client = TestClient(mod.app)
    assert client.get("/api/corpus/parties?limit=3").status_code == 503


def test_corpus_parties_503_when_mirror_empty(monkeypatch):
    """An EMPTY corpus mirror (table exists but 0 rows, e.g. right after a truncate) returns 503,
    NOT 0 parties and NOT a file fallback: an unloaded corpus is a deploy error the operator must
    fix by running the loader, not something the app papers over."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi_testclient.TestClient
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    from amlguard import db as _db
    monkeypatch.setattr(_db, "available", lambda: True)
    monkeypatch.setattr(_db, "read_parties", lambda domain="aml": [])  # empty
    client = TestClient(mod.app)
    assert client.get("/api/corpus/parties?limit=2").status_code == 503


def test_corpus_parties_serves_from_postgres_when_loaded(monkeypatch):
    """When the mirror is populated, the endpoint serves it and reports source=postgres."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi_testclient.TestClient
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    from amlguard import db as _db
    fake = [{"party_id": f"P{i:05d}", "full_name": f"Party {i}"} for i in range(5)]
    monkeypatch.setattr(_db, "available", lambda: True)
    monkeypatch.setattr(_db, "read_parties", lambda domain="aml": fake)
    client = TestClient(mod.app)
    r = client.get("/api/corpus/parties?limit=3")
    assert r.status_code == 200
    j = r.json()
    assert j["source"] == "postgres" and j["count"] == 5 and len(j["parties"]) == 3


def test_pipeline_is_domain_specific(monkeypatch):
    """Each domain returns its own stage list: AML 7-stage, healthcare/support smaller."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi_testclient.TestClient
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    c = TestClient(mod.app)
    aml = c.get("/api/pipeline?domain=aml").json()
    hc = c.get("/api/pipeline?domain=healthcare").json()
    sup = c.get("/api/pipeline?domain=customer-support").json()
    assert [s["id"] for s in aml["stages"]][:2] == ["ingest", "train"]
    assert [s["id"] for s in hc["stages"]] == ["safe_harbor", "expert_determination"]
    assert [s["id"] for s in sup["stages"]] == ["gate1", "gate2"]
    # infer/egress are the paid AML stages
    assert any(s["id"] == "infer" and s["paid"] for s in aml["stages"])


def test_batch_run_paid_stage_requires_confirm(monkeypatch):
    """A paid live-batch stage returns an estimate + confirm_token and runs NOTHING until confirmed
    (the live-batch cost rail). No billing happens in this test."""
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi_testclient.TestClient
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    c = TestClient(mod.app)
    r = c.post("/api/batch/run-stage", json={"domain": "aml", "stage": "infer", "scope": "none"})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "confirm_required" and j["confirm_token"] and j["estimate"]["paid"]
    assert j["estimate"]["cost_usd"] >= 0


def test_batch_run_unknown_stage_404(monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    TestClient = fastapi_testclient.TestClient
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    c = TestClient(mod.app)
    assert c.post("/api/batch/run-stage", json={"domain": "aml", "stage": "nope"}).status_code == 404


def test_confirm_token_is_not_client_forgeable(monkeypatch):
    """The confirm token gate must not be bypassable by a client that computes the token from the
    public (domain, stage, scope, cost) inputs: it is a server-minted single-use nonce, and a
    forged/guessed value re-triggers confirm_required (runs nothing), never execution."""
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    c = TestClient(mod.app)
    # A client submits a made-up token for a paid stage.
    r = c.post("/api/batch/run-stage",
               json={"domain": "aml", "stage": "infer", "scope": "none",
                     "confirm_token": "deadbeefdeadbeef"})
    assert r.status_code == 200
    # It is rejected as unrecognised: we re-quote and issue a FRESH token, nothing is executed.
    assert r.json()["status"] == "confirm_required"


def test_confirm_token_is_single_use(monkeypatch):
    """A server-issued token authorises exactly ONE execution attempt; a replay is rejected. We
    intercept _execute_stage so the test never bills, and assert the second use no longer runs."""
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    calls = {"n": 0}
    monkeypatch.setattr(mod, "_execute_stage", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or {"ok": True}))
    # Neutralise the run lock so the test exercises the token, not fcntl.
    monkeypatch.setattr(mod, "acquire_run_lock", lambda *a, **k: __import__("io").StringIO(), raising=False)
    c = TestClient(mod.app)
    tok = c.post("/api/batch/run-stage",
                 json={"domain": "aml", "stage": "infer", "scope": "none"}).json()["confirm_token"]
    first = c.post("/api/batch/run-stage",
                   json={"domain": "aml", "stage": "infer", "scope": "none", "confirm_token": tok})
    assert first.json()["status"] == "done" and calls["n"] == 1
    # Replaying the SAME token must not run the stage again.
    second = c.post("/api/batch/run-stage",
                    json={"domain": "aml", "stage": "infer", "scope": "none", "confirm_token": tok})
    assert second.json()["status"] == "confirm_required" and calls["n"] == 1


def test_corpus_parties_never_emits_clear_sensitive_fields(monkeypatch):
    """The parties preview endpoint must NEVER return clear ssn/credit_card/dob/account/etc — the
    whole pipeline exists to protect exactly those. Even with a fully-populated row, only the safe
    allow-list columns come back."""
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    from amlguard import db as _db
    hot = [{"party_id": "P00001", "full_name": "Jane Doe", "party_type": "individual",
            "jurisdiction": "US", "risk_rating": "high", "is_pep": False,
            "ssn": "123-45-6789", "credit_card": "4111111111111111",
            "date_of_birth": "1980-01-01", "account_number": "999", "email": "j@x.com",
            "phone": "555", "address": "1 St", "tax_id": "T1", "bank_account": "B1"}]
    monkeypatch.setattr(_db, "available", lambda: True)
    monkeypatch.setattr(_db, "read_parties", lambda domain="aml": hot)
    c = TestClient(mod.app)
    row = c.get("/api/corpus/parties?limit=1").json()["parties"][0]
    for leaked in ("ssn", "credit_card", "date_of_birth", "account_number", "email",
                   "phone", "address", "tax_id", "bank_account"):
        assert leaked not in row, f"{leaked} leaked through the parties preview endpoint"
    assert row["party_id"] == "P00001" and row["full_name"] == "Jane Doe"


def test_corpus_parties_requires_token_when_configured(monkeypatch):
    """When a shared secret is set, the parties endpoint refuses an unauthenticated request (it
    carries the same gate as the billed endpoints)."""
    mod = _load_app(monkeypatch, AMLGUARD_UI_API_TOKEN="s3cret",
                    AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    c = TestClient(mod.app)
    assert c.get("/api/corpus/parties").status_code == 401


# ── Live-chat model selection: a fork runs with or without cloud credentials ──────────
# resolve_ui_model() must pick the hosted model when AWS creds are present, the local open-source
# model otherwise, and honour an explicit override — reporting the choice, never silently swapping.

def _no_aws(monkeypatch):
    """Force 'no cloud credentials' regardless of the runner's real ~/.aws / env."""
    for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE", "AWS_ROLE_ARN"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent/credentials")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent/config")


def test_resolve_ui_model_prefers_hosted_when_aws_present(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    from amlguard import settings
    monkeypatch.setattr(settings, "bedrock_available", lambda: True)
    monkeypatch.delenv("AMLGUARD_UI_MODEL", raising=False)
    model, reason = mod.resolve_ui_model()
    assert model == "bedrock-sonnet-5"
    assert "AWS credentials" in reason


def test_resolve_ui_model_falls_back_to_local_without_aws(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    from amlguard import llm
    _no_aws(monkeypatch)
    monkeypatch.delenv("AMLGUARD_UI_MODEL", raising=False)
    # Pretend Ollama is up and the model is pulled, so no network call is made.
    monkeypatch.setattr(llm, "ensure_ollama_model", lambda *a, **k: True)
    monkeypatch.setattr(llm, "ollama_reachable", lambda *a, **k: True)
    model, reason = mod.resolve_ui_model()
    assert model == "llama3.2"
    assert "local model" in reason


def test_resolve_ui_model_reports_missing_local_model(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    from amlguard import llm
    _no_aws(monkeypatch)
    monkeypatch.delenv("AMLGUARD_UI_MODEL", raising=False)
    # Ollama reachable but the model is not pulled and auto-pull is off.
    monkeypatch.setattr(llm, "ensure_ollama_model", lambda *a, **k: False)
    monkeypatch.setattr(llm, "ollama_reachable", lambda *a, **k: True)
    model, reason = mod.resolve_ui_model()
    assert model == "llama3.2"
    assert "not pulled" in reason and "setup-local-model" in reason


def test_resolve_ui_model_reports_ollama_down(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    from amlguard import llm
    _no_aws(monkeypatch)
    monkeypatch.delenv("AMLGUARD_UI_MODEL", raising=False)
    monkeypatch.setattr(llm, "ensure_ollama_model", lambda *a, **k: False)
    monkeypatch.setattr(llm, "ollama_reachable", lambda *a, **k: False)
    model, reason = mod.resolve_ui_model()
    assert model == "llama3.2"
    assert "not reachable" in reason


def test_resolve_ui_model_honours_explicit_override(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_UI_MODEL="gpt-4o-mini",
                    AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    from amlguard import settings
    # Override wins even when AWS creds are present.
    monkeypatch.setattr(settings, "bedrock_available", lambda: True)
    model, reason = mod.resolve_ui_model()
    assert model == "gpt-4o-mini"
    assert "AMLGUARD_UI_MODEL=gpt-4o-mini" in reason


def test_resolve_ui_model_flags_invalid_override(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_UI_MODEL="no-such-model",
                    AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    model, reason = mod.resolve_ui_model()
    assert model == "no-such-model"
    assert "invalid" in reason.lower()


def test_health_reports_live_chat_model(monkeypatch):
    mod = _load_app(monkeypatch, AMLGUARD_NO_TRACKING="1", AMLGUARD_NO_TRACING="1")
    from amlguard import llm
    _no_aws(monkeypatch)
    monkeypatch.delenv("AMLGUARD_UI_MODEL", raising=False)
    monkeypatch.setattr(llm, "ensure_ollama_model", lambda *a, **k: True)
    monkeypatch.setattr(llm, "ollama_reachable", lambda *a, **k: True)
    monkeypatch.setattr(llm, "ollama_has_model", lambda *a, **k: True)
    c = TestClient(mod.app)
    body = c.get("/api/health").json()
    assert body["ok"] is True
    live = body["live_chat"]
    assert live["model"] == "llama3.2" and live["provider"] == "ollama"
    assert live["ready"] is True
