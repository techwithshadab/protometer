"""Provider request-shaping: the decode-control config must actually reach the API.

Pins the finding that `thinking:disabled` was forwarded on the Bedrock path but silently
dropped on the Anthropic path, so a Claude-5 model declaring it in config would still reason
adaptively and could return zero text (the measured max_tokens/reasoning failure). Both paths
must forward `additional_request_fields`; these tests drive each provider with a fake client
and assert the fields land in the outgoing request.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.llm import ModelSpec

THINKING = {"thinking": {"type": "disabled"}}


def test_anthropic_forwards_additional_request_fields(monkeypatch):
    from protometer import llm

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)

            class _Msg:
                content = [type("B", (), {"type": "text", "text": "{}"})()]
                usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()
            return _Msg()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    prov = llm.AnthropicProvider.__new__(llm.AnthropicProvider)
    prov._client = FakeClient()
    spec = ModelSpec(name="claude", provider="anthropic", model_id="claude-opus-5",
                     max_output_tokens=100, additional_request_fields=THINKING)
    prov.generate(spec, "sys", "prompt", 100)
    assert captured.get("extra_body") == THINKING, (
        "Anthropic path dropped additional_request_fields (thinking:disabled would be ignored)"
    )


def test_anthropic_omits_extra_body_when_no_fields(monkeypatch):
    from protometer import llm

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)

            class _Msg:
                content = [type("B", (), {"type": "text", "text": "{}"})()]
                usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()
            return _Msg()

    class FakeClient:
        messages = FakeMessages()

    prov = llm.AnthropicProvider.__new__(llm.AnthropicProvider)
    prov._client = FakeClient()
    spec = ModelSpec(name="c", provider="anthropic", model_id="m", max_output_tokens=100)
    prov.generate(spec, "sys", "prompt", 100)
    assert "extra_body" not in captured, "empty extra_body should not be sent"
