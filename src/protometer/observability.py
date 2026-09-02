"""LLM observability, every prompt, completion, and verdict traceable in Langfuse.

MLflow answers "which run produced this number" at experiment grain; it does not show the
prompts. For a pipeline whose output is *language*, rationales an analyst reads, judged
answers a score depends on, the reviewable unit is the individual generation: what went in,
what came out, what it cost, and what the guards said about it. Langfuse (self-hosted,
`vendor/langfuse`, loopback-only) is that record; MLflow remains the experiment ledger. The
two deliberately do not overlap: metrics and models to MLflow, generations and their
per-item scores here.

Design mirrors `tracking.Tracker`, for the same reason: telemetry must never become a
dependency. If the SDK is missing, the keys are unset, or the server is down, every function
here is a no-op and the pipeline runs unchanged.

**Data at rest, stated plainly:** Langfuse stores prompts and completions in its database.
At scope `none` those contain clear synthetic narrative text, which is why the stack binds
to 127.0.0.1 only, its volumes stay out of git, and a real deployment would point
LANGFUSE_HOST at an instance inside the same boundary as the model.

Instrumentation lives at the one seam every call crosses (`LLMClient._generate_with_retry`
and the cache-hit path in `complete`), so evaluation, hybrid rationales, judge grading and
preflight are all traced without any caller changing.
"""

from __future__ import annotations

import atexit
import os
from typing import Any

from protometer import settings as _settings
from protometer.log import get_logger

_log = get_logger("observability")

# One Langfuse client PER PROJECT (keyed by project id), so each Protometer domain traces into its OWN
# Langfuse project (aml / healthcare / support) for full isolation, like Botox has its own. The
# default project (LANGFUSE_PUBLIC_KEY/SECRET_KEY, no suffix) is used when no project is named or a
# domain's own keys are unset. Cached so the health probe + import cost are paid once per project.
_clients: dict[str, Any] = {}
_atexit_registered = False

# Protometer domain name -> Langfuse project suffix. The env keys are LANGFUSE_PUBLIC_KEY_<SUFFIX> /
# LANGFUSE_SECRET_KEY_<SUFFIX> (uppercased). A domain with no keys set falls back to the default
# project, so this is safe to add before the healthcare/support projects exist.
_DOMAIN_PROJECT = {"aml": "aml", "healthcare": "healthcare", "customer-support": "support"}


def domain_project(domain: str | None) -> str | None:
    """The Langfuse project id for an Protometer domain (aml/healthcare/support), or None for the
    default project. Used to route each domain's traces into its own project."""
    if not domain:
        return None
    return _DOMAIN_PROJECT.get(domain)

# Clear values that must be scrubbed from generation bodies before they are written to
# Langfuse. At scope `none` (and any partially-protecting scope) the prompt and completion
# carry real identifiers straight from the clear narratives; Langfuse persists bodies at rest
# in its database, so "keep it loopback-bound" is not sufficient protection for a project whose
# thesis is that sensitive data stays controlled. A run installs the corpus's forbidden values
# once via `set_trace_redaction`, and `record_generation` redacts them from system/prompt/
# completion before export. Empty set (the default, e.g. a fully-protected scope) is a no-op.
_REDACT_VALUES: frozenset[str] = frozenset()
_REDACT_TOKEN = "[REDACTED-PII]"


def set_trace_redaction(values: "frozenset[str] | set[str] | None") -> None:
    """ADD clear values to scrub from every exported generation body (union, process-wide).

    Called once per domain by an entry point that knows that corpus's clear identifiers (the same
    forbidden-value set the egress guard uses). Because ONE process now serves all three Protometer
    domains (aml/healthcare/support), this ACCUMULATES a UNION rather than replacing: otherwise the
    last domain to build its guardrail would clobber the others' forbidden values, leaving a
    partial-scope trace of an earlier-loaded domain unredacted. Passing None or an empty set is a
    no-op that keeps the accumulated set (use reset_trace_redaction() to clear, e.g. in tests).
    Defence in depth *in addition to* the loopback bind, not a replacement for it.
    """
    global _REDACT_VALUES
    incoming = frozenset(v for v in (values or ()) if v)
    if incoming:
        _REDACT_VALUES = _REDACT_VALUES | incoming


def reset_trace_redaction() -> None:
    """Clear the accumulated redaction set (tests / a fresh run)."""
    global _REDACT_VALUES
    _REDACT_VALUES = frozenset()


def _redact(text: str) -> str:
    """Replace any installed forbidden clear value with a marker. No-op when none installed.

    Matching is normalized the SAME way the egress guard's leak-check normalizes (NFKC, casefold,
    ignorable-stripped), so a case variant / fullwidth homoglyph / zero-width-joined rendering of
    a forbidden value cannot slip past the trace redactor while the egress guard would have caught
    it. Because normalization is not length-preserving, we redact on the normalized text and export
    that canonical form — this is at-rest telemetry, not user-facing, so a canonicalized body is
    acceptable and, crucially, leaks no clear variant.
    """
    if not _REDACT_VALUES or not text:
        return text
    from protometer.guardrail import _normalize_for_match

    norm = _normalize_for_match(text)
    # Longest-first so a value that contains another is masked whole, not left as a fragment.
    for value in sorted(_REDACT_VALUES, key=len, reverse=True):
        nv = _normalize_for_match(value)
        if nv and nv in norm:
            norm = norm.replace(nv, _REDACT_TOKEN)
    # If nothing matched, the normalized text equals the (casefolded) original with no marker;
    # return the ORIGINAL unchanged so we don't needlessly canonicalize clean bodies.
    return norm if _REDACT_TOKEN in norm else text


def _project_keys(project: str | None) -> tuple[str | None, str | None]:
    """(public, secret) keys for a project. A named project uses LANGFUSE_PUBLIC_KEY_<SUFFIX> /
    LANGFUSE_SECRET_KEY_<SUFFIX>; if those are unset it FALLS BACK to the default keys, so adding a
    domain before its dedicated project exists still traces (into the default project)."""
    default_pub, default_sec = os.getenv("LANGFUSE_PUBLIC_KEY"), os.getenv("LANGFUSE_SECRET_KEY")
    if not project:
        return default_pub, default_sec
    suffix = project.upper().replace("-", "_")
    pub = os.getenv(f"LANGFUSE_PUBLIC_KEY_{suffix}") or default_pub
    sec = os.getenv(f"LANGFUSE_SECRET_KEY_{suffix}") or default_sec
    return pub, sec


def _get_client(project: str | None = None) -> Any:
    """The Langfuse client for a project (or the default project), or None when observability is off.

    One client per distinct key pair, memoized: import cost and the health probe are paid once per
    project, and only by processes that actually trace. `project` routes an Protometer domain's traces
    into its own Langfuse project (aml/healthcare/support); None uses the default project.
    """
    global _atexit_registered
    if os.getenv("PROTOMETER_NO_TRACING") == "1":
        return None

    public, secret = _project_keys(project)
    if not public or not secret:
        return None
    # Cache key is the public key, so two domains that resolve to the SAME keys (e.g. both falling
    # back to the default) share ONE client rather than opening duplicates.
    cache_key = public
    if cache_key in _clients:
        return _clients[cache_key]

    client = None
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=public,
            secret_key=secret,
            host=_settings.langfuse_host(),
            # The SDK default export timeout is 5s; a cold self-hosted stack's first
            # ClickHouse insert regularly exceeds that, and each timeout silently drops a
            # batch of generations. 20s costs nothing on the happy path.
            timeout=_settings.langfuse_timeout(),
        )
        if not _atexit_registered:
            atexit.register(flush)
            _atexit_registered = True
    except Exception as exc:  # noqa: BLE001, observability must never break the run
        _log.warning("disabled: %s: %s", type(exc).__name__, str(exc)[:100])
        client = None
    _clients[cache_key] = client
    return client


def record_generation(
    *,
    component: str,
    model: str,
    system: str,
    prompt: str,
    completion: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    latency_s: float,
    cached: bool,
    attempts: int = 1,
    prompt_name: str | None = None,
    project: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """One completed LLM call, as a Langfuse generation.

    `project` routes this generation into a specific Langfuse project (an Protometer domain's own
    project); None uses the default project. When `prompt_name` is a managed Prompt-Management
    prompt, the generation is LINKED to that prompt's version (UI lineage), resolved from that
    project's client cache so there's no extra round-trip in the hot path.
    """
    client = _get_client(project)
    if client is None:
        return
    try:
        from protometer.persist import RUN_ID

        # Scrub clear identifiers from the bodies before they leave for Langfuse's at-rest
        # store. A no-op unless a run installed forbidden values via `set_trace_redaction`.
        system, prompt, completion = _redact(system), _redact(prompt), _redact(completion)

        # Resolve the managed prompt object to link the generation to its version. Best-effort:
        # a miss (unmanaged prompt, registry hiccup) just means no link, never a failed trace.
        prompt_obj = None
        if prompt_name:
            try:
                prompt_obj = client.get_prompt(prompt_name)
            except Exception:  # noqa: BLE001, no link is fine
                prompt_obj = None

        start_kwargs: dict[str, Any] = {
            "name": f"{component}:{model}",
            "as_type": "generation",
            "model": model,
            "input": {"system": system, "prompt": prompt},
        }
        if prompt_obj is not None:
            start_kwargs["prompt"] = prompt_obj
        obs = client.start_observation(
            **start_kwargs,
            metadata={
                "component": component,
                "run_id": RUN_ID,
                "cached": cached,
                "attempts": attempts,
                "latency_s": round(latency_s, 3),
                # Session groups every call of one process-run into one Langfuse session,
                # so an operator sees "this eval run" or "this hybrid run" as a unit rather
                # than hundreds of orphan traces. The user is the component (eval / judge /
                # hybrid / preflight), so the UI can filter by which stage made the call.
                # These two keys are Langfuse's OTEL-attribute convention for session/user.
                "langfuse_session_id": RUN_ID,
                "langfuse_user_id": component,
                **(metadata or {}),
            },
        )
        obs.update(
            output=completion,
            usage_details={"input": int(input_tokens), "output": int(output_tokens)},
            cost_details={"total": float(cost_usd)},
        )
        obs.end()
    except Exception as exc:  # noqa: BLE001
        _log.warning("generation not recorded: %s: %s", type(exc).__name__, str(exc)[:80])


def record_score(name: str, value: float, comment: str = "", project: str | None = None) -> None:
    """A run-level quality verdict, groundedness flags, egress outcomes, queue precision.

    `project` routes the score into a specific Langfuse project (a domain's own); None = default.

    This self-hosted Langfuse runs in v4 `events_only` mode, where the legacy score-ingestion
    route (`client.create_score`, backing `/api/public/scores`) is DISABLED and returns a
    server-side "Bad request" for every call — the SDK logs it from its background flush, so it
    is silent noise rather than a raised exception, but no comment lands and the path is wrong.
    (Confirmed: `GET /api/public/scores` 404s with "not available ... in v4 events_only mode".)

    The working v4 path is the OTel-event route: attach the score to a real observation via
    `obs.score()` / `obs.score_trace()`, which emits it as a span event (the ingestion channel
    events_only DOES accept) and lands the full value + comment in the `scores` table. So: open
    an anchor span, score its own trace, end it. No standalone `create_score`, no `session_id`
    (session grouping comes from the span metadata, as with every other span here).
    """
    client = _get_client(project)
    if client is None:
        return
    try:
        from protometer.persist import RUN_ID

        obs = client.start_observation(
            name=name, as_type="span",
            metadata={"run_id": RUN_ID, "langfuse_session_id": RUN_ID},
        )
        obs.score_trace(
            name=name,
            value=float(value),
            data_type="NUMERIC",
            comment=comment or None,
        )
        obs.end()
    except Exception as exc:  # noqa: BLE001
        _log.warning("score not recorded: %s: %s", type(exc).__name__, str(exc)[:80])


def managed_prompt(name: str, label: str | None = None, project: str | None = None) -> str:
    """Resolve a prompt: Langfuse's latest version, else the file-backed fallback on disk.

    `project` selects which Langfuse project the prompt is read from / seeded into, so each
    Protometer domain's prompts are versioned in ITS OWN project (aml/healthcare/support); None uses
    the default project.

    The durable source of truth is `config/prompts/<name>.txt` (see `protometer.prompts`), which
    ships with the code and needs no telemetry to read. Langfuse is the *editing surface*: a
    prompt is versioned and UI-editable there, and this fetches the latest (or a `label`ed)
    version so a wording change needs no code deploy.

    The two are kept in step. When Langfuse returns a version whose text differs from the
    file, that text is written back to the file (`prompts.save_prompt`), so the on-disk
    fallback is always the latest version the code has *seen*, not a constant frozen at deploy
    time. Then, if Langfuse is later unreachable, the pipeline falls back to that last-synced
    latest text. On first use the file text is seeded into the registry as v1 so the UI has
    something to edit. Which version served a run is provenance (stamped on linked generations).

    Telemetry is never a dependency: no client, an unseeded prompt, or a registry error all
    resolve to the file. The only hard error is a genuinely missing prompt *file*, which is a
    packaging bug, not a runtime condition.
    """
    from protometer.prompts import load_prompt, save_prompt

    fallback = load_prompt(name)  # disk is the floor; raises only if the file is missing
    client = _get_client(project)
    if client is None:
        return fallback
    try:
        try:
            prompt = client.get_prompt(name, label=label) if label else client.get_prompt(name)
            text = prompt.compile()
            # Keep the durable fallback current with the version we just served.
            save_prompt(name, text)
            return text
        except Exception:  # noqa: BLE001, not-found -> seed the file text as v1
            created = client.create_prompt(
                name=name, prompt=fallback, labels=["production"], type="text",
                commit_message="seeded from config/prompts file",
            )
            # Return the compiled seeded prompt, not the raw file text, so the first run uses
            # exactly what every later run will fetch (identical for our variable-free prompts,
            # but this keeps the seed and steady-state paths from ever diverging).
            try:
                return created.compile()
            except Exception:  # noqa: BLE001, SDK shape varies -> file text is the same content
                return fallback
    except Exception as exc:  # noqa: BLE001, registry unreachable -> file fallback
        _log.warning("prompt %s not managed: %s: %s", name, type(exc).__name__, str(exc)[:60])
        return fallback


def sync_eval_dataset(dataset_name: str, items: list[dict]) -> None:
    """Mirror the eval task set into a Langfuse dataset (idempotent, create-if-missing).

    The evaluation *is* a dataset-and-experiment: fixed tasks with expected answers, run
    across scopes, scored per checkpoint. Representing it as a Langfuse dataset lets the UI
    compare runs item-by-item across scopes and models, which is exactly the thing this
    project measures. `items` is [{id, input, expected}]; created once, reused across runs.
    """
    client = _get_client()
    if client is None:
        return
    try:
        try:
            client.get_dataset(dataset_name)
            return  # already exists; items are stable across runs
        except Exception:  # noqa: BLE001, not-found means create it
            client.create_dataset(
                name=dataset_name,
                description="Protometer investigation tasks: one item per task, expected "
                            "answers per checkpoint. Runs = protection scopes / models.",
            )
        for item in items:
            client.create_dataset_item(
                dataset_name=dataset_name,
                input=item["input"],
                expected_output=item.get("expected"),
                metadata={"task_id": item["id"]},
            )
    except Exception as exc:  # noqa: BLE001
        _log.warning("dataset sync failed: %s: %s", type(exc).__name__, str(exc)[:80])


def record_dataset_run_score(
    dataset_name: str, run_name: str, task_id: str, score_name: str, value: float
) -> None:
    """A per-item score for one experiment run (one scope) over the eval dataset.

    This is what turns the Langfuse dataset into a comparable experiment: each scope is a
    run, each task an item, and the checkpoint score is attached so the UI can diff scopes
    per task. Best-effort and session-grouped like every other score.
    """
    client = _get_client()
    if client is None:
        return
    try:
        from protometer.persist import RUN_ID

        # v4 events_only disables `create_score`; attach the per-item score to a real span via
        # the OTel-event route instead (see record_score). The task/run context that the old
        # standalone score carried in `metadata` now lives on the anchor span's metadata.
        obs = client.start_observation(
            name=f"{run_name}:{task_id}", as_type="span",
            metadata={
                "run_id": RUN_ID, "langfuse_session_id": RUN_ID,
                "dataset": dataset_name, "run": run_name, "task_id": task_id,
            },
        )
        obs.score_trace(name=score_name, value=float(value), data_type="NUMERIC")
        obs.end()
    except Exception as exc:  # noqa: BLE001
        _log.warning("dataset-run score failed: %s: %s", type(exc).__name__, str(exc)[:80])


def flush() -> None:
    """Drain the buffers of EVERY per-project client, required in short-lived scripts, harmless
    elsewhere."""
    for client in list(_clients.values()):
        if client is not None:
            try:
                client.flush()
            except Exception:  # noqa: BLE001
                pass
