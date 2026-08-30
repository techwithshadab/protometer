"""The central settings surface: one place to read config, and .env.example stays in sync.

Pins the config-decoupling invariants: every setting the code reads has a documented default
in settings.py, and .env.example documents every KNOWN_SETTINGS var so a deployment can see the
whole config surface without grepping the source (the gap this fixed: 15 vars were undocumented).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard import settings


def test_env_example_documents_every_known_setting():
    env = (ROOT / ".env.example").read_text()
    missing = [name for name, _ in settings.KNOWN_SETTINGS if name not in env]
    assert not missing, f".env.example is missing documented settings: {missing}"


def test_defaults_resolve_without_env(monkeypatch):
    # With nothing set, every defaulted accessor returns its documented default, no crash.
    for name, _ in settings.KNOWN_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    assert settings.discovery_threshold() == 0.6
    assert settings.max_spend_usd() == 5.0
    assert "8580" in settings.discovery_url()
    assert "8581" in settings.guardrail_url()
    assert settings.langfuse_timeout() == 20
    assert not settings.no_tracing()


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("AMLGUARD_DISCOVERY_THRESHOLD", "0.8")
    assert settings.discovery_threshold() == 0.8
    monkeypatch.setenv("AMLGUARD_NO_TRACING", "1")
    assert settings.no_tracing()


def test_malformed_numeric_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AMLGUARD_DISCOVERY_THRESHOLD", "not-a-number")
    assert settings.discovery_threshold() == 0.6  # not a crash


def test_ui_max_turns_malformed_falls_back(monkeypatch):
    # Read on the hot /api/chat/turn path: a malformed value must degrade to the default, not
    # raise (a raw int() would 500 the endpoint on the first turn).
    monkeypatch.setenv("AMLGUARD_UI_MAX_TURNS", "unlimited")
    assert settings.ui_max_turns() == 200


# ── Live-chat model selection settings ────────────────────────────────────────────────
def test_get_bool_accepts_common_truthy_and_falsy(monkeypatch):
    for v in ("1", "true", "TRUE", "Yes", "on"):
        monkeypatch.setenv("AMLGUARD_AUTO_PULL_MODEL", v)
        assert settings.auto_pull_model() is True, v
    for v in ("0", "false", "No", "off"):
        monkeypatch.setenv("AMLGUARD_AUTO_PULL_MODEL", v)
        assert settings.auto_pull_model() is False, v
    monkeypatch.setenv("AMLGUARD_AUTO_PULL_MODEL", "garbage")
    assert settings.auto_pull_model() is False  # unrecognised -> the default (False)
    monkeypatch.delenv("AMLGUARD_AUTO_PULL_MODEL", raising=False)
    assert settings.auto_pull_model() is False


def test_model_selection_defaults(monkeypatch):
    for v in ("AMLGUARD_LOCAL_MODEL", "AMLGUARD_HOSTED_MODEL", "AMLGUARD_UI_MODEL"):
        monkeypatch.delenv(v, raising=False)
    assert settings.local_model() == "llama3.2"
    assert settings.hosted_ui_model() == "bedrock-sonnet-5"
    assert settings.ui_model() is None


def test_bedrock_available_true_with_env_keys(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA...")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    assert settings.bedrock_available() is True


def test_bedrock_available_false_without_creds(monkeypatch):
    for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE", "AWS_ROLE_ARN"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent/credentials")
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent/config")
    assert settings.bedrock_available() is False


def test_ollama_url_explicit_wins(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://ollama:11434")
    monkeypatch.setenv("AMLGUARD_IN_CONTAINER", "1")  # would otherwise force host.docker.internal
    assert settings.ollama_url() == "http://ollama:11434"


def test_ollama_url_container_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.setenv("AMLGUARD_IN_CONTAINER", "1")
    assert settings.ollama_url() == "http://host.docker.internal:11434"


def test_ollama_url_host_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("AMLGUARD_IN_CONTAINER", raising=False)
    monkeypatch.setattr(settings, "_in_container", lambda: False)
    assert settings.ollama_url() == "http://localhost:11434"


def test_ui_role_tokens_parses_and_binds(monkeypatch):
    monkeypatch.setenv("AMLGUARD_UI_ROLE_TOKENS", "auditor:tok-a, investigator:tok-i")
    m = settings.ui_role_tokens()
    assert m == {"tok-a": "auditor", "tok-i": "investigator"}


def test_ui_role_tokens_empty_by_default(monkeypatch):
    monkeypatch.delenv("AMLGUARD_UI_ROLE_TOKENS", raising=False)
    assert settings.ui_role_tokens() == {}


def test_ui_role_tokens_ignores_malformed(monkeypatch):
    monkeypatch.setenv("AMLGUARD_UI_ROLE_TOKENS", "garbage,:notoken,role:,ok:tok")
    assert settings.ui_role_tokens() == {"tok": "ok"}
