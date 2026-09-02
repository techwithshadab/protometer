"""LLM layer, configuration-driven, provider-agnostic, instrumented.

Every model the pipeline can use is declared in `config/models.yaml`, never in code. Swapping
an open-source model on a workstation for a hosted frontier model is a configuration change,
which matters for two reasons: the architecture's claims must not depend on any single
vendor, and a reviewer must be able to reproduce the findings on whatever they have access to.

What this layer provides beyond a bare API call:

  * **Provider abstraction**, Ollama, Anthropic and OpenAI behind one `complete()` surface.
  * **Cost and latency accounting**, per-call tokens, USD, and p50/p95 latency, because
    "protection is affordable" is a claim this project has to substantiate with numbers.
  * **Deterministic decoding**, temperature 0 and a fixed seed. The evaluation attributes
    output differences to protection scope, so sampling noise would contaminate the result.
  * **Response caching**, keyed on the full request. Evaluation runs repeat identical
    prompts across scopes; without caching that is wasted latency and spend.
  * **Retries with backoff**, transient failures are retried, permanent ones fail fast.
  * **Fallback chain**, if the configured model is unreachable the next is tried, and the
    model that actually served each call is recorded so results are never misattributed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
from amlguard import settings as _settings  # noqa: E402

OLLAMA_URL = _settings.ollama_url()

# Deterministic decoding. Differences between runs must be attributable to protection scope.
TEMPERATURE = 0.0
SEED = 20260811

# Process-wide spend ledger, shared by every `LLMClient` in this interpreter.
#
# `max_spend_usd` was enforced against each client's own `stats`, so the documented "$5 cap"
# multiplied by however many clients a run happened to construct, `eval/runner.py` builds one
# per scope plus a judge and a baseline, giving a real ceiling of $45-85. A single mutable cell
# behind one lock keeps the ceiling the number the operator set.
_SPEND_LOCK = threading.Lock()
_PROCESS_SPEND_USD = [0.0]


def process_spend_usd() -> float:
    """Total hosted-model spend across every client in this process."""
    with _SPEND_LOCK:
        return _PROCESS_SPEND_USD[0]

# Errors worth retrying: transient transport and server-side capacity problems.
RETRYABLE_MARKERS = (
    "timeout", "timed out", "connection", "429", "rate limit", "overloaded",
    "500", "502", "503", "504", "unavailable", "econnreset",
    # Bedrock signals throttling by exception name rather than status code, and its wording
    # matched none of the markers above, so throttled calls failed immediately instead of
    # backing off. Measured: 42 of 50 rapid rationale calls were lost this way while the run
    # still reported success, because each failure degrades to a placeholder rather than
    # raising.
    "throttlingexception", "too many requests", "toomanyrequests",
    "serviceunavailable", "modelnotready", "modeltimeout",
)


class LLMError(RuntimeError):
    """Raised when a completion cannot be obtained."""


class LLMConfigError(LLMError):
    """Raised when the requested model is not declared in config/models.yaml."""


class SpendCapExceeded(LLMError):
    """Raised when a call would take the process ledger past the operator's ceiling.

    A distinct type because `complete()` must treat it differently from every other
    `LLMError`: a provider failure may legitimately try the fallback chain, but a breached
    cap must stop the run. Routing it through fallback either continued a capped run on a
    free local model (a cap that quietly downgrades is not a cap) or surfaced as "all
    configured models failed", masking the actual cause from the operator.
    """


@dataclass
class ModelSpec:
    """One entry from `config/models.yaml`."""

    name: str
    provider: str
    model_id: str
    context_window: int = 8192
    max_output_tokens: int = 2048
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    supports_prompt_caching: bool = False
    # Provider-specific request fields, verbatim. Exists because Claude 5 models on Bedrock
    # reason *adaptively by default*: a hard prompt can spend the entire `maxTokens` budget
    # inside a reasoningContent block and return zero text, measured live, 19 eval tasks
    # failed exactly this way (stopReason=max_tokens, block types [reasoningContent],
    # text len 0). The evaluation's premise is a fixed decode budget spent on the answer,
    # so those models declare {"thinking": {"type": "disabled"}} here, in config, where a
    # reviewer can see the measurement condition instead of inferring it.
    additional_request_fields: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def is_local(self) -> bool:
        return self.provider == "ollama"

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1_000_000 * self.cost_per_1m_input
            + output_tokens / 1_000_000 * self.cost_per_1m_output
        )


@dataclass
class ModelRegistry:
    """The declared model catalogue."""

    specs: dict[str, ModelSpec]
    default: str
    fallback_order: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "ModelRegistry":
        if not path.exists():
            raise LLMConfigError(f"Model config not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        specs = {
            name: ModelSpec(name=name, **entry)
            for name, entry in (raw.get("models") or {}).items()
        }
        if not specs:
            raise LLMConfigError(f"No models declared in {path}")
        return cls(
            specs=specs,
            default=raw.get("default") or next(iter(specs)),
            fallback_order=list(raw.get("fallback_order") or []),
        )

    def get(self, name: str) -> ModelSpec:
        try:
            return self.specs[name]
        except KeyError:
            raise LLMConfigError(
                f"Unknown model {name!r}. Declared: {', '.join(sorted(self.specs))}"
            ) from None


@dataclass
class CallRecord:
    """One completed call. Kept individually so percentiles are real, not estimated."""

    model: str
    seconds: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False
    attempts: int = 1


@dataclass
class LLMStats:
    """Aggregate accounting across a run."""

    records: list[CallRecord] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return len(self.records)

    @property
    def billed_calls(self) -> int:
        return sum(1 for r in self.records if not r.cached)

    @property
    def cache_hits(self) -> int:
        return sum(1 for r in self.records if r.cached)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def total_seconds(self) -> float:
        return sum(r.seconds for r in self.records)

    @property
    def input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.records)

    @property
    def output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    def latency_percentile(self, pct: float) -> float:
        """Latency over billed calls only, cache hits would flatter the number."""
        live = sorted(r.seconds for r in self.records if not r.cached)
        if not live:
            return 0.0
        index = min(int(len(live) * pct / 100), len(live) - 1)
        return live[index]

    def summary(self) -> str:
        return (
            f"calls={self.calls} (billed={self.billed_calls} cached={self.cache_hits}) "
            f"tokens={self.input_tokens}in/{self.output_tokens}out "
            f"cost=${self.total_cost_usd:.4f} "
            f"p50={self.latency_percentile(50):.1f}s p95={self.latency_percentile(95):.1f}s"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "billed_calls": self.billed_calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_seconds": round(self.total_seconds, 2),
            "latency_p50": round(self.latency_percentile(50), 2),
            "latency_p90": round(self.latency_percentile(90), 2),
            "latency_p95": round(self.latency_percentile(95), 2),
            "latency_p99": round(self.latency_percentile(99), 2),
            "latency_max": round(self.latency_percentile(100), 2),
        }


class Provider(Protocol):
    """What every backend must implement. Returns (text, input_tokens, output_tokens)."""

    def generate(
        self, spec: ModelSpec, system: str, prompt: str, max_tokens: int
    ) -> tuple[str, int, int]: ...


def _estimate_tokens(text: str) -> int:
    """Rough token count for providers that do not report usage.

    Deliberately approximate, it feeds cost estimates for local models, which are zero-cost
    anyway, so precision here would buy nothing.
    """
    return max(1, len(text) // 4)


def ollama_reachable(timeout: float = 2.0) -> bool:
    """True if an Ollama server answers at OLLAMA_URL. Cheap, non-fatal probe used to decide
    whether the local open-source fallback is usable before committing to it."""
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout).raise_for_status()
        return True
    except Exception:  # noqa: BLE001, any failure just means "not usable right now"
        return False


def ollama_has_model(model_id: str, timeout: float = 5.0) -> bool:
    """True if `model_id` is already pulled on the reachable Ollama server. Matches with and
    without an explicit `:latest` tag, since `ollama list` and a bare name refer to the same model."""
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout).json().get("models", [])
    except Exception:  # noqa: BLE001
        return False
    names = {m.get("name", "") for m in tags}
    wanted = {model_id, f"{model_id}:latest", model_id.removesuffix(":latest")}
    return bool(names & wanted)


def ollama_pull(model_id: str, progress: "Callable[[str], None] | None" = None,
                timeout: float = 3600.0) -> None:
    """Pull `model_id` on the reachable Ollama server (the one-time setup). Streams pull status
    lines to `progress` (default: print), so a first-time user sees the download happening rather
    than a silent multi-minute stall. Raises LLMConfigError if the server is unreachable."""
    log = progress or (lambda s: print(s, flush=True))
    if not ollama_reachable():
        raise LLMConfigError(
            f"Ollama is not reachable at {OLLAMA_URL}. Install it (https://ollama.com/download) and "
            f"start it, or set OLLAMA_URL to a running server."
        )
    log(f"Pulling local model '{model_id}' (one-time download)...")
    with requests.post(f"{OLLAMA_URL}/api/pull", json={"model": model_id, "stream": True},
                       stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        last = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                evt = json.loads(line)
            except ValueError:
                continue
            status = evt.get("status", "")
            if status and status != last:
                log(f"  {status}")
                last = status
            if evt.get("error"):
                raise LLMConfigError(f"Ollama pull failed: {evt['error']}")
    log(f"Model '{model_id}' is ready.")


def ensure_ollama_model(model_id: str, auto_pull: bool = False,
                        progress: "Callable[[str], None] | None" = None) -> bool:
    """Make sure `model_id` is available locally. Returns True if it is (already present, or pulled
    just now when `auto_pull`), False if it is missing and auto-pull is off. The one-time-setup seam:
    the UI calls this with `auto_pull` from AMLGUARD_AUTO_PULL_MODEL; the Make target calls it True."""
    if not ollama_reachable():
        return False
    if ollama_has_model(model_id):
        return True
    if not auto_pull:
        return False
    ollama_pull(model_id, progress=progress)
    return ollama_has_model(model_id)


class OllamaProvider:
    """Local inference. No key, no egress, no per-token cost."""

    def generate(
        self, spec: ModelSpec, system: str, prompt: str, max_tokens: int
    ) -> tuple[str, int, int]:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": spec.model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "num_predict": max_tokens,
                    "num_ctx": spec.context_window,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        content = (payload.get("message") or {}).get("content", "")
        # Ollama reports real counts; fall back to estimation only if absent.
        return (
            content,
            int(payload.get("prompt_eval_count") or _estimate_tokens(system + prompt)),
            int(payload.get("eval_count") or _estimate_tokens(content)),
        )


class AnthropicProvider:
    """Hosted inference via the Anthropic API.

    Note what crosses the network here: tokens, never identifiers. That is the architecture's
    claim, and routing through a hosted model is a demonstration of it rather than an
    exception to it.
    """

    def __init__(self) -> None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMConfigError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:
            raise LLMConfigError("pip install anthropic") from exc
        # Bound the request so an interactive serving turn cannot hang on the SDK's ~10-minute
        # default. Overridable via AMLGUARD_LLM_TIMEOUT for a batch run that wants more slack.
        self._client = anthropic.Anthropic(
            timeout=float(os.getenv("AMLGUARD_LLM_TIMEOUT", "120"))
        )

    def generate(
        self, spec: ModelSpec, system: str, prompt: str, max_tokens: int
    ) -> tuple[str, int, int]:
        # The system prompt is identical across every call in a run, so caching it removes
        # it from repeat input cost.
        system_block: Any = system
        if spec.supports_prompt_caching:
            system_block = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]

        # Forward the same decode-control config the Bedrock path forwards. Claude 5 reasons
        # adaptively by default and can spend the whole max_tokens budget inside a thinking
        # block, returning zero text (measured on Bedrock, notes above); a model that declares
        # {"thinking": {"type": "disabled"}} in config must have that honoured on the Anthropic
        # path too, or the same silent zero-text failure reappears when a reviewer points
        # `--model` at the hosted-Anthropic build. Passed through `extra_body`, which the SDK
        # forwards verbatim to the API, so this needs no SDK-version-specific typed param and
        # carries any other declared field unchanged.
        extra: dict[str, Any] = {}
        if spec.additional_request_fields:
            extra["extra_body"] = dict(spec.additional_request_fields)

        message = self._client.messages.create(
            model=spec.model_id,
            max_tokens=min(max_tokens, spec.max_output_tokens),
            temperature=TEMPERATURE,
            system=system_block,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )
        content = "".join(b.text for b in message.content if b.type == "text")
        return content, message.usage.input_tokens, message.usage.output_tokens


class OpenAIProvider:
    """Hosted inference via the OpenAI API, present so findings can be shown to hold
    across vendors rather than being an artefact of one."""

    def __init__(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMConfigError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMConfigError("pip install openai") from exc
        # Bounded like the Anthropic client, so a serving turn cannot hang on the SDK default.
        self._client = OpenAI(timeout=float(os.getenv("AMLGUARD_LLM_TIMEOUT", "120")))

    def generate(
        self, spec: ModelSpec, system: str, prompt: str, max_tokens: int
    ) -> tuple[str, int, int]:
        completion = self._client.chat.completions.create(
            model=spec.model_id,
            max_tokens=min(max_tokens, spec.max_output_tokens),
            temperature=TEMPERATURE,
            seed=SEED,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        content = completion.choices[0].message.content or ""
        usage = completion.usage
        return content, usage.prompt_tokens, usage.completion_tokens


class BedrockProvider:
    """Inference via AWS Bedrock, using the account's existing AWS credentials.

    Preferred over the direct API for this project: spend lands on the AWS bill, access is
    governed by IAM rather than a loose API key, and the data path stays inside a boundary an
    enterprise already controls, which is the deployment a bank would actually accept.

    What crosses that boundary is still tokens, never identifiers.
    """

    def __init__(self) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise LLMConfigError("pip install boto3") from exc

        region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def generate(
        self, spec: ModelSpec, system: str, prompt: str, max_tokens: int
    ) -> tuple[str, int, int]:
        # Bedrock's Converse API is model-agnostic and reports usage uniformly, so the same
        # call shape works across vendors on Bedrock rather than only for Anthropic models.
        # `temperature` is deprecated on newer Claude models and Bedrock rejects the whole
        # request when it is supplied, which, because this client has a fallback chain,
        # silently routed every "Bedrock" call to the local model. Results were attributed to
        # Sonnet that a 14B local model actually produced.
        #
        # Determinism is unaffected: these models are deterministic at their default setting,
        # which is what the parameter was pinning anyway.
        config: dict[str, Any] = {"maxTokens": min(max_tokens, spec.max_output_tokens)}

        # The system prompt is byte-identical across every call in a run, the rationale
        # instructions, the judge rubric, the investigation brief, so it is paid for on every
        # request unless it is cached. The Anthropic provider below already marks it
        # `ephemeral`; Bedrock takes the same idea as a `cachePoint` block appended to the
        # system array, and this path was sending it uncached.
        #
        # Measured on the 189-token rationale prompt: $0.3697 -> $0.0375 across a 652-call
        # curve, since cache reads bill at roughly a tenth of input. Small in absolute terms
        # and free to claim, and it grows with every prompt that gets longer.
        #
        # Guarded on the model spec rather than assumed: Bedrock rejects `cachePoint` for
        # models that do not support it, and a rejected request would fall through the
        # fallback chain exactly the way the `temperature` incident did.
        system_blocks: list[dict[str, Any]] = [{"text": system}]
        if spec.supports_prompt_caching:
            system_blocks.append({"cachePoint": {"type": "default"}})

        def _converse(blocks: list[dict[str, Any]], inference: dict[str, Any]):
            extra: dict[str, Any] = {}
            if spec.additional_request_fields:
                extra["additionalModelRequestFields"] = spec.additional_request_fields
            return self._client.converse(
                modelId=spec.model_id,
                system=blocks,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig=inference,
                **extra,
            )

        try:
            response = _converse(system_blocks, {**config, "temperature": TEMPERATURE})
        except Exception as exc:  # noqa: BLE001, botocore raises a generic ClientError
            message = str(exc)
            # Retry without the caching hint before giving up on it, so an unsupported
            # `cachePoint` degrades to a working uncached call rather than to the local model.
            if "cachePoint" in message or "cache" in message.lower():
                system_blocks = [{"text": system}]
                try:
                    response = _converse(system_blocks, {**config, "temperature": TEMPERATURE})
                except Exception as retry_exc:  # noqa: BLE001
                    if "temperature" not in str(retry_exc):
                        raise
                    response = _converse(system_blocks, config)
            elif "temperature" not in message:
                raise
            else:
                response = _converse(system_blocks, config)
        content = "".join(
            block.get("text", "")
            for block in response["output"]["message"]["content"]
        )
        usage = response.get("usage", {})
        return content, int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0))


PROVIDERS: dict[str, type] = {
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "bedrock": BedrockProvider,
}


@dataclass
class LLMClient:
    """The pipeline's single entry point to any model.

    Caching is on by default because evaluation runs repeat identical prompts across
    protection scopes; without it those repeats cost latency and money for no new information.
    The cache is keyed on the full request including model id, so a scope or model change
    correctly misses.
    """

    model: str | None = None
    registry: ModelRegistry = field(default_factory=ModelRegistry.load)
    max_retries: int = 4
    enable_cache: bool = True
    cache_dir: Path | None = None
    allow_fallback: bool = True
    # Hard spend ceiling in USD for this client's lifetime. Exceeding it raises rather than
    # continuing: an unattended evaluation loop is exactly the shape of program that quietly
    # runs up a bill, and a cap that warns is not a cap. Local models are always free, so this
    # only ever binds on hosted providers.
    max_spend_usd: float = float(os.getenv("AMLGUARD_MAX_SPEND_USD", "5.0"))
    # Opaque tag folded into every cache key. Changing it invalidates the cache wholesale.
    #
    # Prompt text alone is not a sufficient key: a change to scoring, detection, or task
    # construction alters what an answer *means* without necessarily altering the prompt, and a
    # stale hit would then be served as a fresh measurement. Callers set this to something that
    # changes when the run should be considered new, a scope slug, a corpus fingerprint, a
    # code version, and `--no-cache` bypasses it entirely.
    cache_namespace: str = ""
    # Label under which this client's calls appear in Langfuse ("eval", "hybrid", "judge").
    # Purely observational; empty means the generic "llm".
    trace_component: str = ""
    # Optional Langfuse session key. A batch run leaves this empty and every call groups under
    # the process RUN_ID; a serving layer sets it to a conversation id so each conversation is
    # its own session in the UI. Purely observational.
    trace_session: str = ""
    # Optional Langfuse Prompt-Management name of the SYSTEM prompt this client is running. When set,
    # record_generation resolves the managed prompt object and LINKS the generation to its version
    # (UI lineage: "this generation used prompt vN"). Empty -> no link. Purely observational.
    trace_prompt_name: str = ""
    # Optional Langfuse PROJECT this client's generations trace into (an AMLGuard domain's own
    # project: aml/healthcare/support). Empty -> the default project. Purely observational.
    trace_project: str = ""
    stats: LLMStats = field(default_factory=LLMStats)

    _provider: Any = field(default=None, repr=False)
    _spec: ModelSpec | None = field(default=None, repr=False)
    _memory_cache: dict[str, str] = field(default_factory=dict, repr=False)
    # Guards the spend check and the stats append, which together are a read-modify-write.
    # Without it, concurrent callers can each observe the same pre-call total and collectively
    # overshoot the cap by up to (workers x per-call cost), turning a hard ceiling into an
    # advisory one, which is the opposite of what a spend cap is for.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        requested = self.model or os.getenv("AMLGUARD_MODEL") or self.registry.default
        self._spec = self.registry.get(requested)
        self._provider = self._make_provider(self._spec)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_provider(spec: ModelSpec) -> Any:
        factory = PROVIDERS.get(spec.provider)
        if factory is None:
            raise LLMConfigError(
                f"Unknown provider {spec.provider!r}. Known: {', '.join(PROVIDERS)}"
            )
        return factory()

    @property
    def spec(self) -> ModelSpec:
        assert self._spec is not None
        return self._spec

    @property
    def name(self) -> str:
        return f"{self.spec.provider}/{self.spec.model_id}"

    # -- caching -----------------------------------------------------------------

    def _cache_key(
        self, system: str, prompt: str, max_tokens: int,
        spec_override: ModelSpec | None = None,
    ) -> str:
        spec = spec_override or self.spec
        payload = json.dumps(
            {
                "namespace": self.cache_namespace,
                "model": spec.model_id,
                "provider": spec.provider,
                "system": system,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": TEMPERATURE,
                "seed": SEED,
                # Request-shaping fields change what a completion *is* (reasoning on vs
                # off); serving a completion produced under a different decode config would
                # be silent contamination wearing a cache hit.
                "request_fields": spec.additional_request_fields,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, key: str) -> str | None:
        if not self.enable_cache:
            return None
        if key in self._memory_cache:
            return self._memory_cache[key]
        if self.cache_dir:
            path = self.cache_dir / f"{key}.json"
            if path.exists():
                # A torn entry (disk full, kill mid-write) must be a cache MISS, not a
                # permanent crash: the unguarded form raised JSONDecodeError out of
                # complete() on every later run touching the same prompt, the one spot
                # where corruption did not self-heal.
                try:
                    content = json.loads(path.read_text())["completion"]
                except (json.JSONDecodeError, KeyError, OSError):
                    path.unlink(missing_ok=True)
                    return None
                self._memory_cache[key] = content
                return content
        return None

    def _cache_put(self, key: str, completion: str) -> None:
        if not self.enable_cache:
            return
        self._memory_cache[key] = completion
        if self.cache_dir:
            from amlguard.persist import atomic_write_json

            try:
                atomic_write_json(
                    self.cache_dir / f"{key}.json",
                    {"model": self.spec.model_id, "completion": completion},
                    indent=0,
                )
            except OSError:
                pass  # a failed cache write must not fail the completed call

    # -- generation ---------------------------------------------------------------

    def is_cached(self, system: str, prompt: str, max_tokens: int | None = None) -> bool:
        """Whether `complete` with these arguments would be served without a network call.

        Exists so callers that pace themselves (the rationale loop sleeps between calls for
        the provider's burst limit) can skip the pause on hits, a re-run over a warm cache
        was paying 0.4s x 50 sleeps to make zero requests.
        """
        if not self.enable_cache:
            return False
        # Same budget defaulting as `complete`, or the probe and the call compute
        # different keys and every peek on a default-budget call reports a miss.
        key = self._cache_key(system, prompt, max_tokens or self.spec.max_output_tokens)
        if key in self._memory_cache:
            return True
        return bool(self.cache_dir) and (self.cache_dir / f"{key}.json").exists()

    def would_bill(self, system: str, prompt: str, max_tokens: int | None = None) -> bool:
        """Whether `complete` with these arguments would make a live provider call.

        The pacing seam: callers that sleep between calls to respect a provider's burst
        limit ask the client, instead of re-deriving its cache-key semantics. A caller once
        hardcoded the token budget in a second file and the two drifted; this method is the
        one place that knowledge lives.
        """
        try:
            return not self.is_cached(system, prompt, max_tokens)
        except Exception:  # noqa: BLE001, a broken peek must never block a call
            return True

    def complete(self, system: str, prompt: str, max_tokens: int | None = None) -> str:
        """Generate a completion, retrying transient failures and falling back if needed."""
        budget = max_tokens or self.spec.max_output_tokens
        key = self._cache_key(system, prompt, budget)

        cached = self._cache_get(key)
        if cached is not None:
            with self._lock:
                self.stats.records.append(CallRecord(self.name, 0.0, 0, 0, 0.0, cached=True))
            from amlguard.observability import record_generation

            record_generation(
                component=self.trace_component or "llm",
                model=self.name, system=system, prompt=prompt, completion=cached,
                input_tokens=0, output_tokens=0, cost_usd=0.0, latency_s=0.0,
                cached=True, prompt_name=self.trace_prompt_name or None,
                project=self.trace_project or None,
                metadata={"namespace": self.cache_namespace,
                          **({"langfuse_session_id": self.trace_session}
                             if self.trace_session else {})},
            )
            return cached

        # Spend enforcement lives in ONE place: the reservation inside
        # `_generate_with_retry`, which claims the projected cost under the ledger lock and
        # reconciles to the actual bill. A second check-only copy used to sit here; two
        # enforcement points with one semantics is how they drift, and the fallback path
        # bypassed this one entirely.
        try:
            return self._generate_with_retry(system, prompt, budget, key)
        except SpendCapExceeded:
            raise  # a breached cap stops the run, fallback would defeat its purpose
        except LLMError:
            if not self.allow_fallback:
                raise
            return self._generate_with_fallback(system, prompt, budget, key)


    def _generate_with_retry(
        self,
        system: str,
        prompt: str,
        budget: int,
        key: str,
        spec: ModelSpec | None = None,
        provider: Any = None,
    ) -> str:
        """One model's retry loop.

        `spec`/`provider` are **parameters**, not client state. The fallback path used to
        swap `self._spec`/`self._provider` in place and restore them in a `finally`, correct
        single-threaded, and a race under the evaluation's five-worker pool: another thread
        mid-`complete()` could read the swapped spec, computing the wrong cache key and
        attributing the wrong cost to the wrong model. Passing them down removes the shared
        mutation instead of locking around it, which would have serialised every call behind
        the slowest fallback.
        """
        spec = spec or self.spec
        provider = provider or self._provider
        delay = 1.0
        last: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            # Reserve the projected cost *inside* the ledger lock, reconcile after.
            #
            # Check-then-act let N workers each observe the same pre-call total and
            # collectively overshoot the cap by N x per-call cost, the precise race the
            # cap's own docstring claimed the lock prevented. With reservation, a concurrent
            # caller sees the budget already claimed; residual overshoot is bounded by
            # N x (actual - projected), i.e. the projection's estimation error, not the full
            # call cost. Verified under an 8-way concurrent test: exactly the affordable
            # number of calls complete and the ledger equals the cap.
            projected = 0.0
            if not spec.is_local:
                projected = spec.cost_usd(len(system + prompt) // 4, budget)
                with _SPEND_LOCK:
                    if _PROCESS_SPEND_USD[0] + projected > self.max_spend_usd:
                        spent = _PROCESS_SPEND_USD[0]
                        raise SpendCapExceeded(
                            f"Spend cap reached: ${spent:.4f} spent across all clients, "
                            f"${projected:.4f} projected for this call on {spec.name}, "
                            f"cap ${self.max_spend_usd:.2f}."
                        )
                    _PROCESS_SPEND_USD[0] += projected
            started = time.monotonic()
            try:
                content, input_tokens, output_tokens = provider.generate(
                    spec, system, prompt, budget
                )
                elapsed = time.monotonic() - started
                call_cost = spec.cost_usd(input_tokens, output_tokens)
                # Reconcile BEFORE the empty-content check: an empty completion still
                # billed its input tokens, and the except path below releases whatever is
                # left in `projected`, reconciling first (and zeroing the reservation)
                # keeps the ledger equal to what the provider actually charged.
                if not spec.is_local:
                    with _SPEND_LOCK:
                        _PROCESS_SPEND_USD[0] += call_cost - projected
                    projected = 0.0
                if not content.strip():
                    raise LLMError("empty completion")

                with self._lock:
                    self.stats.records.append(
                        CallRecord(
                            # provider/model_id, matching the cached-record format, the two
                            # paths once recorded different name shapes for the same model,
                            # which made "which models answered this run" ungroupable.
                            model=f"{spec.provider}/{spec.model_id}",
                            seconds=elapsed,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cost_usd=call_cost,
                            attempts=attempt,
                        )
                    )
                self._cache_put(key, content)
                from amlguard.observability import record_generation

                record_generation(
                    component=self.trace_component or "llm",
                    model=f"{spec.provider}/{spec.model_id}",
                    system=system, prompt=prompt, completion=content,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    cost_usd=call_cost, latency_s=elapsed, cached=False,
                    attempts=attempt, prompt_name=self.trace_prompt_name or None,
                    project=self.trace_project or None,
                    metadata={"namespace": self.cache_namespace,
                          **({"langfuse_session_id": self.trace_session}
                             if self.trace_session else {})},
                )
                return content

            except Exception as exc:  # noqa: BLE001, provider SDKs raise varied types
                # A failed call spent nothing (or will be re-reserved on retry): release.
                if not spec.is_local and projected:
                    with _SPEND_LOCK:
                        _PROCESS_SPEND_USD[0] -= projected
                last = exc
                message = str(exc).lower()
                retryable = any(marker in message for marker in RETRYABLE_MARKERS)
                if not retryable or attempt == self.max_retries:
                    break
                # Jitter avoids synchronised retries when calls are issued in parallel.
                time.sleep(delay + random.uniform(0, delay * 0.25))
                delay = min(delay * 2, 30.0)

        raise LLMError(f"{spec.name} failed after {self.max_retries} attempts: {last}")

    def _generate_with_fallback(
        self, system: str, prompt: str, budget: int, key: str
    ) -> str:
        """Try the configured fallback chain, recording which model actually answered."""
        for candidate in self.registry.fallback_order:
            if candidate == self.spec.name:
                continue
            try:
                spec = self.registry.get(candidate)
                provider = self._make_provider(spec)
            except LLMConfigError:
                continue  # not installed or not credentialed, try the next

            # The candidate spec travels as a parameter, never installed on the client. The
            # previous swap-and-restore was correct single-threaded and a race under the
            # five-worker pool; and a swap left in place after success once attributed an
            # entire run to a model that never produced it (a sibling of the mixed-model incident).
            try:
                # Cached under the *fallback's* key. Writing it under the primary's key would
                # serve the fallback's answer on a later run where the primary was healthy.
                fallback_key = self._cache_key(system, prompt, budget, spec_override=spec)
                content = self._generate_with_retry(
                    system, prompt, budget, fallback_key, spec=spec, provider=provider
                )
                return content
            except SpendCapExceeded:
                raise  # trying the next candidate would spend past a breached cap
            except LLMError:
                continue

        raise LLMError("all configured models failed, including fallbacks")


def get_llm(model: str | None = None, **kwargs: Any) -> LLMClient:
    """Construct a client for `model`, or the configured default."""
    return LLMClient(model=model, **kwargs)


def preflight(model: str | None = None) -> str:
    """One live call proving the *resolved* provider answers, before a long or paid run.

    The failure this prevents happened once: a dead Ollama or a rejected Bedrock
    model id entered the fallback chain silently, and an entire curve labelled "Sonnet 5" was
    served by a local 14B model. Discovery and the guardrail both have preflights; the LLM -
    the component with the recorded history of silent substitution, did not.

    Three properties matter and each is deliberate:
      * **fallback disabled**, a probe that may fall back validates nothing;
      * **cache disabled**, a cached "OK" proves the disk works, not the provider;
      * **the resolved spec is compared to the request**, the same guard `run_hybrid` applies.

    Returns the resolved model name. Raises LLMError with the operator's next step otherwise.
    Cost: ~8 output tokens on hosted providers; free locally.
    """
    client = LLMClient(model=model, enable_cache=False, allow_fallback=False)
    if model and client.spec.name != model:
        raise LLMError(
            f"requested {model!r} but registry resolved {client.spec.name!r}, "
            f"check config/models.yaml"
        )
    try:
        reply = client.complete("Reply with exactly one word.", "Say OK.", max_tokens=8)
    except Exception as exc:  # noqa: BLE001, every failure gets the same instruction shape
        hint = (
            "start Ollama and pull the model (`ollama pull ...`)"
            if client.spec.is_local
            else "check AWS credentials/region and that the model id is enabled in Bedrock"
        )
        raise LLMError(
            f"Preflight failed for {client.spec.name!r} "
            f"({type(exc).__name__}: {str(exc)[:120]}). "
            f"Nothing has been spent; {hint}."
        ) from exc
    if not reply.strip():
        raise LLMError(f"Preflight got an empty completion from {client.spec.name!r}.")
    return client.spec.name


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a completion.

    Models wrap JSON in prose or fences despite instructions. Treating that as a wrong answer
    would confound output-format compliance with the reasoning degradation the evaluation
    exists to measure, so parsing is deliberately tolerant.
    """
    candidates = [text]
    for part in text.split("```"):
        candidates.append(part[4:] if part.startswith("json") else part)

    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
    # Salvage a truncated object rather than losing the answer.
    #
    # A response cut off by the token ceiling still contains complete leading fields, the
    # content is there, only the closing brace is missing. Discarding it reported "no JSON
    # found" for answers that were substantively fine, and 13 of 50 rationales were lost this
    # way even after raising the limit, because a thorough model simply writes long.
    start = text.find("{")
    if start != -1:
        fragment = text[start:]
        # Trim back to the last complete "key": "value" pair and close the object.
        last_pair = max(fragment.rfind('",'), fragment.rfind('"}'))
        if last_pair > 0:
            try:
                return json.loads(fragment[: last_pair + 1] + "}")
            except json.JSONDecodeError:
                pass

    raise LLMError(f"No JSON object found in completion: {text[:200]!r}")
