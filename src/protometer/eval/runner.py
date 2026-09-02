"""Evaluation runner, the same tasks across every protection scope, unattended.

The evaluation design requires this to run all scopes in a single command: a harness needing supervision
turns five runs into five days, while one that runs unattended costs a single night.

The design rule that makes results trustworthy is that **only the corpus changes between
runs**. Same tasks, same prompts, same model, same seed, same temperature, so any difference
in score is attributable to protection scope and nothing else.

Results are written incrementally. A run that dies partway leaves completed scopes intact and
resumes rather than repaying their cost.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from protometer.eval.baseline import run_baseline
from protometer.eval.scoring import CheckpointScore, TaskScore, score_checkpoint
from protometer.eval.tasks import InvestigationTask, Stratum, build_tasks
from protometer.llm import LLMClient, get_llm
from protometer.persist import RUN_ID, atomic_write_json
from protometer.pipeline import InvestigationPipeline
from protometer.retrieval import NarrativeIndex
from protometer.scopes import CURVE_ORDER, ProtectionScope, get_scope
from protometer.tracking import Tracker


@dataclass
class ScopeResult:
    """Every task's score under one protection scope, plus what it cost."""

    scope_name: str
    model: str
    tasks: list[TaskScore] = field(default_factory=list)
    llm_stats: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    # Every model name that appears in the client's call records for this scope. With
    # fallback disabled this must be a single entry equal to `model`; the field exists so
    # the artifact carries the evidence rather than the claim.
    models_used: list[str] = field(default_factory=list)
    # Egress-scan summary over model answers for this scope. Best-effort: the guardrail is a
    # local sidecar, and a measurement harness must run without it, but when it is up, model
    # output leaves the eval unscanned in no path.
    egress: dict[str, int] = field(default_factory=dict)

    @property
    def checkpoint_scores(self) -> list[CheckpointScore]:
        return [c for task in self.tasks for c in task.checkpoints]

    @property
    def mean_checkpoint_score(self) -> float:
        """The headline figure: mean score across every checkpoint.

        Carries the granularity that reveals the *shape* of degradation, which strict task
        completion cannot.
        """
        scores = self.checkpoint_scores
        return sum(c.score for c in scores) / len(scores) if scores else 0.0

    @property
    def task_completion_rate(self) -> float:
        """Fraction of tasks where every checkpoint passed. Realism-facing, deliberately strict."""
        return (
            sum(1 for t in self.tasks if t.completed) / len(self.tasks) if self.tasks else 0.0
        )

    def stratum_scores(self) -> dict[str, float]:
        """Mean score per stratum. Mandatory: a blended average measures the task mix."""
        buckets: dict[str, list[float]] = {}
        for checkpoint in self.checkpoint_scores:
            buckets.setdefault(checkpoint.stratum.value, []).append(checkpoint.score)
        return {k: sum(v) / len(v) for k, v in sorted(buckets.items()) if v}

    def verifiable_score(self) -> float:
        """Mean over exact-match strata only, excluding judged narrative checkpoints.

        Reported separately so the headline number stays independently verifiable by a
        reviewer who does not trust an LLM judge.
        """
        scores = [
            c.score for c in self.checkpoint_scores if c.stratum is not Stratum.NARRATIVE
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict:
        return {
            "scope": self.scope_name,
            "model": self.model,
            "mean_checkpoint_score": round(self.mean_checkpoint_score, 4),
            "verifiable_score": round(self.verifiable_score(), 4),
            "task_completion_rate": round(self.task_completion_rate, 4),
            "stratum_scores": {k: round(v, 4) for k, v in self.stratum_scores().items()},
            "seconds": round(self.seconds, 1),
            "models_used": self.models_used,
            "egress": self.egress,
            "llm_stats": self.llm_stats,
            "tasks": [t.to_dict() for t in self.tasks],
        }


def _narrative_fingerprint(scope_slug: str, protected_root: Path) -> str:
    """This scope's protected-narrative fingerprint (the value the index staleness guard uses).

    Distinct from the source-corpus fingerprint: this ties an INDEX to the protected narratives it
    was built from. Logged to MLflow (as `index_fingerprint`) and written into the eval artifact so
    index drift is a queryable fact, not something only discovered when a run raises StaleIndexError.
    Empty string when the scope's narratives are absent.
    """
    from protometer.retrieval import corpus_fingerprint as _narr_fingerprint

    narr_path = protected_root / scope_slug / "narratives.json"
    if not narr_path.exists():
        return ""
    return _narr_fingerprint(json.loads(narr_path.read_text()))


def _eval_index(scope_slug: str, index_root: Path, protected_root: Path) -> NarrativeIndex:
    """Open a scope's narrative index for the EVAL READ path, fingerprint-validated.

    The build path checks staleness; the read path did not, so a non-empty index left over from
    a superseded corpus was queried silently (contaminating retrieved notes). Compute this
    scope's narrative fingerprint, hand it to the index, and validate once — raising
    StaleIndexError on mismatch rather than serving the previous corpus's chunks.
    """
    fp = _narrative_fingerprint(scope_slug, protected_root)
    index = NarrativeIndex(scope_slug, index_root / scope_slug, fp)
    index.validate_fingerprint()
    return index


def run_scope(
    scope: ProtectionScope,
    tasks: list[InvestigationTask],
    protected_root: Path,
    index_root: Path,
    llm: LLMClient,
    judge: LLMClient | None = None,
    progress: bool = True,
    max_workers: int = 5,
    detection_dir: Path | None = None,
) -> ScopeResult:
    """Run every task against one protected corpus."""
    started = time.monotonic()

    # Install Langfuse trace-redaction UNCONDITIONALLY, before the first generation and
    # independent of whether the egress guardrail is built. Redaction was previously a side
    # effect of Guardrail.for_corpus, which the block below skips for `none` (empty entities) —
    # yet `none` is the ONE scope whose prompts/completions carry full clear identifiers, so it
    # was exactly the scope exported to Langfuse unredacted. The egress SCAN stays none-exempt
    # (its clear exposure is deliberate); the at-rest REDACTION must cover every scope.
    try:
        from protometer.guardrail import forbidden_values_from_parties
        from protometer.observability import set_trace_redaction

        _parties = json.loads(
            (protected_root.parent / "corpus" / "parties.json").read_text()
        )
        set_trace_redaction(forbidden_values_from_parties(_parties))
    except Exception:  # noqa: BLE001, redaction is a backstop; never break the eval over it
        if progress:
            print(f"  [{scope.name}] trace-redaction not installed (parties unreadable)")

    # Egress guard over model answers, best-effort. The hybrid path scans every rationale an
    # analyst sees; this path was the remaining unscanned model output. `none` is exempt by
    # design, it is the clear baseline, and flagging its deliberate exposure as leaks would
    # bury real findings in expected ones.
    guardrail = None
    if scope.entities:
        try:
            from protometer.guardrail import Guardrail

            guardrail = Guardrail.for_corpus(
                protected_root.parent / "corpus" / "parties.json"
            )
        except Exception:  # noqa: BLE001, the harness must run without the sidecar
            if progress:
                print(f"  [{scope.name}] guardrail unavailable, eval answers unscanned")
    pipeline = InvestigationPipeline(
        protected_dir=protected_root / scope.slug,
        index=_eval_index(scope.slug, index_root, protected_root),
        llm=llm,
        scope_name=scope.name,
        detection_dir=detection_dir,
    )

    result = ScopeResult(scope_name=scope.name, model=llm.name)
    _egress_lock = threading.Lock()

    def run_task(task: InvestigationTask) -> TaskScore:
        task_started = time.monotonic()
        investigation = pipeline.investigate(
            task_id=task.task_id,
            question=task.question,
            response_shape=task.response_shape,
            party_id=task.party_id,
            top_k=task.top_k,
        )

        task_score = TaskScore(
            task_id=task.task_id, error=investigation.error, model=llm.name
        )
        if guardrail is not None and investigation.succeeded:
            # Per-task best-effort, explicitly. The guardrail client itself fails closed
            # (correct for the analyst-facing path), but in a *measurement* harness a
            # sidecar timeout must not zero the task's utility score, that records an
            # infrastructure hiccup as model degradation. Measured before this guard: 7
            # tasks across two scopes scored 0.00 because the local sidecar was starved
            # for 30s. Failures are counted in their own bucket, never silently.
            try:
                # ensure_ascii=False: default JSON escaping turns non-ASCII identifiers
                # into \uXXXX sequences the substring leak-check can never match, a
                # silent false-PASS class (latent on this corpus, structural elsewhere).
                verdict = guardrail.scan_response(
                    json.dumps(investigation.answer, ensure_ascii=False)
                )
            except Exception:  # noqa: BLE001, scan failure is a counted outcome
                with _egress_lock:
                    result.egress["scan_failed"] = result.egress.get("scan_failed", 0) + 1
                verdict = None
            if verdict is not None:
                key = (
                    "blocked" if verdict.blocked
                    else "discounted" if verdict.discounted
                    else "clean"
                )
                with _egress_lock:
                    result.egress[key] = result.egress.get(key, 0) + 1
                if verdict.blocked and verdict.leaked_values:
                    # A real corpus value in a model answer is the failure this system
                    # exists to prevent, surfaced as the task's error, not buried. The
                    # error reports the COUNT, never the values: result files are
                    # committable, and an error message repeating the leak is a leak.
                    task_score.error = (
                        f"egress: {len(verdict.leaked_values)} forbidden value(s) "
                        f"detected in the model answer"
                    )
        if investigation.succeeded:
            task_score.checkpoints = [
                score_checkpoint(checkpoint, investigation.answer, judge)
                for checkpoint in task.checkpoints
            ]
        else:
            # A failed call still scores every checkpoint as zero rather than dropping the
            # task, so failures depress the score instead of vanishing from the denominator.
            task_score.checkpoints = [
                score_checkpoint(checkpoint, {}, None) for checkpoint in task.checkpoints
            ]
        task_score.seconds = time.monotonic() - task_started
        return task_score

    # Tasks within a scope run concurrently.
    #
    # Safe because each prompt is built before its call and depends on nothing from any other
    # task, there is no cross-task state, so results are order-independent. Decoding stays
    # deterministic (temperature 0, fixed seed), and the spend cap is guarded by a lock in the
    # client, without which concurrent callers could each observe the same pre-call total and
    # collectively overshoot it.
    #
    # Scopes remain sequential so per-scope result writing and resume semantics are untouched,
    # and so latency and cost stay attributable to one scope at a time.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_task, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            try:
                task_score = future.result()
            except Exception as exc:  # noqa: BLE001, one task must not abort the scope
                task_score = TaskScore(task_id=task.task_id, error=f"runner: {exc}")
                task_score.checkpoints = [
                    score_checkpoint(checkpoint, {}, None) for checkpoint in task.checkpoints
                ]
            result.tasks.append(task_score)
            completed += 1

            if progress:
                marker = "OK " if task_score.completed else "   "
                print(
                    f"  [{scope.name}] {completed:>2}/{len(tasks)} {marker}"
                    f"{task.task_id:<26} score={task_score.mean_score:.2f}"
                    + (f"  ({task_score.error})" if task_score.error else "")
                )

    # Restore task order. Completion order is nondeterministic under concurrency, and the
    # results file is compared across runs, an unstable ordering would show as a spurious diff.
    order = {task.task_id: i for i, task in enumerate(tasks)}
    result.tasks.sort(key=lambda t: order.get(t.task_id, 0))

    result.seconds = time.monotonic() - started
    result.llm_stats = llm.stats.to_dict()
    result.models_used = sorted({r.model for r in llm.stats.records})
    return result


def run_evaluation(
    corpus_dir: Path,
    protected_root: Path,
    index_root: Path,
    output_dir: Path,
    scopes: list[str] | None = None,
    model: str | None = None,
    judge_model: str | None = None,
    max_typologies: int = 8,
    resume: bool = True,
    progress: bool = True,
    no_cache: bool = False,
    detection_dir: Path | None = None,
    # Concurrency per scope. Exposed so a throttling provider can be handled with a flag
    # rather than an edit, hosted throughput varies by account and time of day.
    max_workers: int = 5,
) -> dict[str, ScopeResult]:
    """Run the full evaluation across scopes and write results incrementally."""
    scope_names = scopes or list(CURVE_ORDER)
    tracker = Tracker("protometer-llm")

    # Two fingerprints with two jobs, and the distinction is load-bearing.
    #
    # The *prompt* fingerprint covers only the files whose bytes reach prompts
    # (transactions, narratives). It namespaces the LLM cache: a cached completion is valid
    # exactly when the model, the prompt, and the corpus text behind it are unchanged -
    # ground-truth edits do not invalidate a completion for an unchanged prompt.
    #
    # The *corpus* fingerprint covers every corpus file, including the ones that shape tasks
    # and grading (alerts, ground_truth, parties). It guards resume: a stored result graded
    # against superseded ground truth must re-evaluate. The original single fingerprint
    # covered only the prompt files, so alerts/ground_truth/parties changed underneath a
    # committed result and resume kept serving it as current.
    prompt_fingerprint = hashlib.sha256(
        (corpus_dir / "transactions.json").read_bytes()
        + (corpus_dir / "narratives.json").read_bytes()
    ).hexdigest()[:12]
    corpus_fingerprint = hashlib.sha256(
        b"".join(
            (corpus_dir / name).read_bytes()
            for name in (
                "transactions.json", "narratives.json",
                "alerts.json", "ground_truth.json", "parties.json",
            )
        )
    ).hexdigest()[:12]

    # Entity-resolution expectations must match the ledger the prompt actually carries, so
    # tasks are built against the *unprotected* pipeline's network view. Using the baseline
    # keeps a single task set across every scope, the tasks are the constant, and only the
    # corpus varies, while still reflecting what a prompt contains.
    baseline = InvestigationPipeline(
        protected_dir=protected_root / "none",
        index=_eval_index("none", index_root, protected_root),
        llm=get_llm(model),
        scope_name="none",
    )
    # Detection runs on the clear ledger unless told otherwise, so every scope receives
    # identical candidate input and the curve isolates the model.
    # None -> clear ledger (isolates the model). The sentinel path means "each scope's own
    # protected ledger", i.e. end-to-end deployment behaviour where detectors degrade too.
    detection_per_scope = detection_dir is not None and detection_dir.name == "__per_scope__"
    detection_source = None if detection_per_scope else (detection_dir or protected_root / "none")
    tasks = build_tasks(
        corpus_dir,
        max_typologies=max_typologies,
        visible_transactions=lambda pid: baseline.transaction_network(pid, hops=2, limit=120),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    atomic_write_json(output_dir / "tasks.json", [t.to_dict() for t in tasks])

    # Mirror the task set into a Langfuse dataset once, so each scope becomes a comparable
    # experiment run over the same items (best-effort; no-op when tracing is off).
    from protometer.observability import sync_eval_dataset

    sync_eval_dataset(
        "protometer-investigation-tasks",
        [{"id": t.task_id, "input": t.question,
          "expected": {c.answer_key: str(c.expected) for c in t.checkpoints}}
         for t in tasks],
    )

    checkpoint_count = sum(len(t.checkpoints) for t in tasks)
    if progress:
        print(f"{len(tasks)} tasks, {checkpoint_count} checkpoints, {len(scope_names)} scopes\n")

    results: dict[str, ScopeResult] = {}

    for name in scope_names:
        scope = get_scope(name)
        result_path = output_dir / f"{scope.slug}.json"

        if resume and result_path.exists():
            # Resume only when the stored result came from *this* corpus. Keying on the
            # filename alone republished results produced by a different corpus: 11 of 26
            # committed result files carried `corpus_fingerprint: null`, including one for a
            # scope that had never been ingested at all, and `generate_results.py` then
            # globbed them into `docs/results.md` as legitimate measurements.
            #
            # The fingerprint was already folded into the LLM cache key for exactly this
            # reason; the resume check simply never consulted it.
            try:
                stored = json.loads(result_path.read_text()).get("corpus_fingerprint")
            except (OSError, json.JSONDecodeError):
                stored = None
            if stored == corpus_fingerprint:
                if progress:
                    print(f"  [{scope.name}] already evaluated, skipping")
                continue
            if progress:
                print(
                    f"  [{scope.name}] stored result is from a different corpus "
                    f"(stored {stored}, current {corpus_fingerprint}), re-evaluating"
                )

        if not (protected_root / scope.slug / "narratives.json").exists():
            if progress:
                print(f"  [{scope.name}] not ingested, skipping")
            continue

        # Staleness guard: the protected ledger must derive from the SAME clear corpus the
        # tasks were just built from. A regenerated corpus with a stale data/protected would
        # otherwise run fresh tasks against old tokens silently, a failure that happened once
        # happened once. The ingest report stamps its source fingerprint; compare it.
        report_path = protected_root / scope.slug / "ingestion_report.json"
        if report_path.exists():
            try:
                stamped = json.loads(report_path.read_text()).get("source_fingerprint")
            except (OSError, json.JSONDecodeError):
                stamped = None
            if stamped and stamped != corpus_fingerprint:
                raise RuntimeError(
                    f"[{scope.name}] protected ledger derives from corpus {stamped} but the "
                    f"current corpus is {corpus_fingerprint}. Re-ingest this scope "
                    f"(`python scripts/ingest_all.py {scope.slug}`) before evaluating; "
                    f"running would score fresh tasks against stale tokens."
                )

        # A fresh client per scope keeps latency and cost attributable to that scope, and
        # prevents the response cache from serving one scope's answers to another.
        # Cache directories are per scope. A shared cache would serve one scope's answer to
        # another whenever their prompts coincide, which happens for tasks with no party_id,
        # whose prompts derive only from retrieved chunks, misattributing that scope's latency
        # and cost to a call it never made.
        # Namespaced by scope *and* corpus fingerprint, so regenerating the corpus or
        # changing detection/scoring produces a fresh key rather than a stale hit served as a
        # new measurement.
        namespace = f"{scope.slug}:{prompt_fingerprint}"
        # Fallback is disabled for evaluation clients. The curve's entire claim is "same
        # model, only the corpus changes", a fallback that silently substitutes a local
        # model mid-run (as Bedrock throttling once caused, contaminating every scope with a
        # three-model mix) breaks the invariant invisibly. Under this posture a throttled
        # call retries with backoff and then fails the task *visibly* as a task error.
        llm = get_llm(
            model,
            cache_dir=None if no_cache else output_dir / scope.slug / "llm_cache",
            cache_namespace=namespace,
            enable_cache=not no_cache,
            allow_fallback=False,
            trace_component="eval",
        )
        judge = (
            get_llm(
                judge_model,
                cache_dir=None if no_cache else output_dir / scope.slug / "judge_cache",
                cache_namespace=namespace,
                enable_cache=not no_cache,
                allow_fallback=False,
                trace_component="judge",
            )
            if judge_model
            else llm
        )

        result = run_scope(
            scope, tasks, protected_root, index_root, llm, judge,
            progress=progress,
            max_workers=max_workers,
            detection_dir=None if detection_per_scope else detection_source,
        )
        # Per-task scores for this scope's experiment run over the dataset, so the Langfuse
        # UI can diff scopes item by item. Best-effort; no-op when tracing is off.
        from protometer.observability import record_dataset_run_score

        for t in result.tasks:
            record_dataset_run_score(
                "protometer-investigation-tasks", scope.name, t.task_id,
                f"score:{scope.name}", t.mean_score,
            )
        # Copy-the-detector baseline over the same tasks: the model's score minus this is the
        # part attributable to reasoning rather than transcription.
        baseline_scores = run_baseline(
            InvestigationPipeline(
                protected_dir=protected_root / scope.slug,
                index=_eval_index(scope.slug, index_root, protected_root),
                llm=llm,
                scope_name=scope.name,
                detection_dir=None if detection_per_scope else detection_source,
            ),
            tasks,
        )
        payload = result.to_dict()
        payload["copy_baseline"] = {
            "mean_checkpoint_score": round(
                sum(c.score for t in baseline_scores for c in t.checkpoints)
                / max(sum(len(t.checkpoints) for t in baseline_scores), 1),
                4,
            ),
            "tasks": [t.to_dict() for t in baseline_scores],
        }
        payload["corpus_fingerprint"] = corpus_fingerprint
        # The per-scope index fingerprint (protected narratives), written to the artifact AND
        # logged to MLflow below, so index drift is trackable/queryable rather than surfacing only
        # as a StaleIndexError mid-run. Distinct from corpus_fingerprint (the source-corpus hash).
        index_fingerprint = _narrative_fingerprint(scope.slug, protected_root)
        payload["index_fingerprint"] = index_fingerprint
        payload["run_id"] = RUN_ID
        payload["detection_ledger"] = (
            "protected (per scope)" if detection_per_scope else "clear"
        )

        # Tracked with the parameters that determine whether two runs are comparable at all:
        # model, scope, corpus fingerprint, index fingerprint, detection ledger, cache state.
        with tracker.run(
            f"llm:{name}",
            {
                "scope": name,
                "model": result.model,
                "index_fingerprint": index_fingerprint,
                "detection_ledger": payload["detection_ledger"],
                "cache": "disabled" if no_cache else "enabled",
                "tasks": len(tasks),
            },
        ):
            tracker.log_metrics({
                "mean_checkpoint_score": result.mean_checkpoint_score,
                "verifiable_score": result.verifiable_score(),
                "task_completion_rate": result.task_completion_rate,
                "copy_baseline": payload["copy_baseline"]["mean_checkpoint_score"],
                # The quantity the whole exercise is for: what the model adds over copying.
                "marginal_over_baseline": (
                    result.mean_checkpoint_score
                    - payload["copy_baseline"]["mean_checkpoint_score"]
                ),
                # Failures aggregated rather than buried per-task inside the scope JSON -
                # a scope where 6 of 18 tasks errored is a different measurement from one
                # where none did, and previously nothing surfaced the difference.
                "task_errors": sum(1 for t in result.tasks if t.error),
                **{f"egress.{k}": v for k, v in result.egress.items()},
                **{f"stratum.{k}": v for k, v in result.stratum_scores().items()},
                # LLM cost and latency are NOT logged here: they are per-generation facts
                # that live in Langfuse (prompt, completion, tokens, cost, latency at the
                # LLMClient seam). MLflow keeps only what defines the experiment, the scores
                # and the parameters that make two runs comparable. One fact, one home.
            })
            tracker.log_artifact_json(f"{name}.json", payload)

        results[name] = result
        atomic_write_json(result_path, payload)

        if progress:
            print(
                f"  [{scope.name}] mean={result.mean_checkpoint_score:.3f} "
                f"verifiable={result.verifiable_score():.3f} "
                f"completion={result.task_completion_rate:.1%} "
                f"({result.seconds:.0f}s)\n"
            )

    return results
