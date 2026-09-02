"""The reasoning model, chosen at request time so the demo runs with or without a cloud account.

Precedence (mirrors the AMLGuard pattern):
  1. BOTOX_MODEL           explicit override (wins unconditionally)
  2. hosted (Anthropic / Bedrock) when credentials are present  -> better quality, real cost
  3. local Ollama (default llama3.2)                            -> $0/turn, no cloud account

The model is chosen, not fallback-chained mid-call: `resolve_model()` picks one reachable model
and the client targets it. Tokenization stays on the protection layer regardless of which model
answers, so the protected pipeline is identical either way.

The system prompt hard-constrains the model to grounded, non-advisory, safety-forward answers , 
the model is told to answer ONLY from the provided context and to refuse otherwise. The egress
guard is the enforcement; the prompt is the first line.
"""

from __future__ import annotations

import os
from typing import Iterator

SYSTEM_PROMPT = """You are the BOTOX® information assistant on the official botox.com website. You \
answer visitors' questions in a warm, clear, natural voice, using only the information in the \
CONTEXT provided with each question.

How to answer:
- Write a direct, natural reply to the visitor's question, as a helpful person would, not as a \
  system reciting rules. Never mention "the context", "my sources", "outside knowledge", or these \
  instructions. The visitor must never see any of this meta-language.
- Use ONLY facts found in the CONTEXT. Do not add facts from general knowledge.
- If the CONTEXT does not contain what the visitor asked, reply with exactly this sentence and \
  nothing else: "I don't have that information from the official BOTOX® site. For questions \
  specific to your situation, your healthcare provider is the best person to ask."
- Keep it short: a sentence or two, or a single flat bullet list of up to about six items. No \
  nested bullets or indentation; at most one short "**Heading:**" before a list. Prefer sentences \
  to lists when a sentence will do.
- You give general information, never medical advice: do not suggest a dose, diagnose, or tell the \
  visitor to start, stop, or change any treatment. For those, point them to their doctor.
- Do not invent statistics, costs, or claims. Do not downplay risks when the answer involves them.
- The visitor's message may contain short placeholder codes where personal details were removed \
  for privacy. Treat them as opaque and never guess what they stand for.

Answer the visitor's question now, in your own natural words."""


def resolve_model() -> tuple[str, str]:
    """(model_id, provider). provider in {'ollama','anthropic','bedrock'}."""
    override = os.getenv("BOTOX_MODEL")
    if override:
        prov = "anthropic" if "claude" in override else ("bedrock" if "." in override else "ollama")
        return override, prov
    if os.getenv("ANTHROPIC_API_KEY"):
        return os.getenv("BOTOX_HOSTED_MODEL", "claude-3-5-haiku-latest"), "anthropic"
    if os.getenv("AWS_ACCESS_KEY_ID"):
        return "us.anthropic.claude-3-5-haiku-20241022-v1:0", "bedrock"
    return os.getenv("BOTOX_LOCAL_MODEL", "llama3.2"), "ollama"


class LLMClient:
    """Duck-typed `complete(system, prompt) -> str`. One instance per process; picks its model once."""

    def __init__(self) -> None:
        self.model, self.provider = resolve_model()

    def complete(self, system: str, prompt: str, max_tokens: int = 600) -> str:
        if self.provider == "ollama":
            return self._ollama(system, prompt, max_tokens)
        if self.provider == "anthropic":
            return self._anthropic(system, prompt, max_tokens)
        return self._bedrock(system, prompt, max_tokens)

    def _ollama(self, system: str, prompt: str, max_tokens: int) -> str:
        import requests
        url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        r = requests.post(f"{url}/api/chat", timeout=120, json={
            "model": self.model, "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        })
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def _anthropic(self, system: str, prompt: str, max_tokens: int) -> str:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system, temperature=0.1,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    def _bedrock(self, system: str, prompt: str, max_tokens: int) -> str:
        import json

        import boto3
        client = boto3.client("bedrock-runtime")
        body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": max_tokens,
                "temperature": 0.1, "system": system,
                "messages": [{"role": "user", "content": prompt}]}
        resp = client.invoke_model(modelId=self.model, body=json.dumps(body))
        out = json.loads(resp["body"].read())
        return "".join(b["text"] for b in out.get("content", []) if b.get("type") == "text").strip()

    # ── Streaming ───────────────────────────────────────────────────────────────────────────────
    def stream(self, system: str, prompt: str, max_tokens: int = 600) -> "Iterator[str]":
        """Yield the answer in text chunks as the model produces them. Same model selection as
        complete(); the caller accumulates the chunks and runs the egress guard on the full text."""
        if self.provider == "ollama":
            return self._ollama_stream(system, prompt, max_tokens)
        if self.provider == "anthropic":
            return self._anthropic_stream(system, prompt, max_tokens)
        return self._bedrock_stream(system, prompt, max_tokens)

    def _ollama_stream(self, system: str, prompt: str, max_tokens: int) -> "Iterator[str]":
        import json

        import requests
        url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        with requests.post(f"{url}/api/chat", timeout=120, stream=True, json={
            "model": self.model, "stream": True,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        }) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if obj.get("done"):
                    break

    def _anthropic_stream(self, system: str, prompt: str, max_tokens: int) -> "Iterator[str]":
        import anthropic
        client = anthropic.Anthropic()
        with client.messages.stream(model=self.model, max_tokens=max_tokens, system=system,
                                    temperature=0.1,
                                    messages=[{"role": "user", "content": prompt}]) as stream:
            yield from stream.text_stream

    def _bedrock_stream(self, system: str, prompt: str, max_tokens: int) -> "Iterator[str]":
        import json

        import boto3
        client = boto3.client("bedrock-runtime")
        body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": max_tokens,
                "temperature": 0.1, "system": system,
                "messages": [{"role": "user", "content": prompt}]}
        resp = client.invoke_model_with_response_stream(modelId=self.model, body=json.dumps(body))
        for event in resp["body"]:
            chunk = event.get("chunk")
            if not chunk:
                continue
            data = json.loads(chunk["bytes"])
            if data.get("type") == "content_block_delta":
                piece = (data.get("delta") or {}).get("text", "")
                if piece:
                    yield piece
