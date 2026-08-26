"""Measure the hybrid: classifier ranks the queue, the model reasons over its head.

The question is not "does the LLM improve the ranking", it does not touch the ranking. It is
whether model-assisted review produces a **defensible rationale** for the items an analyst will
actually open, at a bounded and predictable cost.

    python scripts/run_hybrid.py --capacity 25
    python scripts/run_hybrid.py --capacity 25 --model bedrock-sonnet-5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# README step 2 is `cp .env.example .env`; make that instruction true.
from amlguard.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from amlguard.alert_queue import AlertQueueResult, rank_alerts
from amlguard.hybrid import HybridResult, rank_queue
from amlguard.llm import get_llm
from amlguard.tracking import Tracker
from amlguard.training import build_classifier


def main() -> int:
    run_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="none")
    parser.add_argument("--capacity", type=int, default=25,
                        help="review depth, a staffing parameter, not a tuned threshold")
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-llm", action="store_true", help="rank only, skip rationales")
    parser.add_argument("--yes", action="store_true",
                        help="skip the cost confirmation (for unattended runs)")
    parser.add_argument("--grain", choices=("alert", "transaction"), default="alert",
                        help="alert grain is what an analyst actually reviews; "
                             "transaction grain is retained for comparison")
    parser.add_argument("--no-guardrail", action="store_true",
                        help="skip the Semantic Guardrail egress check on rationales")
    args = parser.parse_args()

    from amlguard.persist import acquire_run_lock

    try:
        _lock = acquire_run_lock(ROOT / "data")  # held for process lifetime  # noqa: F841
    except RuntimeError as exc:
        sys.exit(str(exc))

    corpus = ROOT / "data" / "corpus"
    protected = ROOT / "data" / "protected" / args.scope

    # One construction path, shared with `train_scope`. This script previously duplicated the
    # split/feature/illicit-set logic inline, and the copy went stale: transaction-level
    # split, whole-ledger feature fitting, no dual-membership exclusion, every leak
    # `training.py` had eliminated was still live here, and the published queue precision was
    # computed on it. The bundle makes that drift impossible: there is nothing to assemble.
    bundle = build_classifier(protected, corpus)
    transactions = bundle.transactions
    model, features, names = bundle.model, bundle.features, bundle.feature_names
    test_idx = bundle.test_idx
    scores = bundle.test_scores
    item_ids = [transactions[i]["transaction_id"] for i in test_idx]
    truth = {
        transactions[i]["transaction_id"]: int(bundle.labels[i]) for i in test_idx
    }

    if args.grain == "alert":
        # The queue an analyst actually works: alerts on subjects, not raw transactions.
        # At transaction grain 50 queued items collapsed to 17 distinct subjects, one party
        # appearing ten times, an analyst would open the same case ten times.
        alerts = json.loads((ROOT / "data" / "corpus" / "alerts.json").read_text())
        escalating = {a["alert_id"] for a in alerts if a.get("escalated")}
        result = AlertQueueResult(review_capacity=args.capacity)
        # The scoring window is the temporal test fold; the queue is the alerts that fired in
        # it. See rank_alerts.window_start for why anything else is a category error.
        #
        # Dates come from the CLEAR ledger, not the protected one: at any scope tokenizing
        # DATETIME, `value_date` is a token and min() over tokens is a random cutoff, the
        # filter silently stopped filtering (measured: quasi queued 1,607 alerts against
        # none's 728, making the two scopes incomparable). Identical bug class to the
        # temporal split, fixed the same way; the window is experiment metadata, not a
        # feature the model sees.
        clear_dates = {
            t["transaction_id"]: t["value_date"]
            for t in json.loads((corpus / "transactions.json").read_text())
        }
        window_start = min(
            clear_dates[transactions[i]["transaction_id"]] for i in test_idx
        )
        result.decisions = rank_alerts(
            alerts, transactions, item_ids, scores, args.capacity,
            window_start=window_start,
        )
        precision = result.precision_at_capacity(escalating)
        result.precision_at_capacity_value = precision
        result.classifier_hash = bundle.model_hash
        print(f"queue {len(result.decisions)} alerts | capacity {args.capacity}")
        print(f"precision@capacity: {precision:.3f}")
        print(
            f"distinct subjects in head: {result.distinct_subjects()}/{args.capacity}"
            f" | overdue: {sum(1 for d in result.escalated if d.overdue)}"
        )
    else:
        result = HybridResult(review_capacity=args.capacity)
        result.decisions = rank_queue(item_ids, scores, args.capacity)
        print(f"queue {len(result.decisions)} items | capacity {args.capacity}")
        print(f"precision@capacity: {result.precision_at_capacity(truth):.3f}")

    blocked = discounted = ungrounded_count = 0
    from amlguard.review import ReviewOutcome

    outcome = ReviewOutcome()  # default so --no-llm runs don't NameError in the tracker block

    if not args.no_llm:
        # Explicit model, never the auto-resolved default: a run launched with
        # --model bedrock-sonnet-5 that silently fell back to the local model produced
        # rationales at $0.00 and attributed them to the wrong system.
        from amlguard.llm import preflight

        # Preflight is the real guard against silent model substitution: it makes
        # one live call with fallback disabled and compares the resolved spec to the request.
        # A name-equality check that used to sit here was tautological, `spec.name` IS the
        # registry key, and unknown keys already raise in the registry, with a comment that
        # described the opposite of what the code did.
        preflight(args.model)
        # Disk-backed cache, like the eval runner's. Without a cache_dir the client caches
        # in memory only, so every re-invocation re-billed all fifty rationales (~$0.70) -
        # measured three times in one afternoon before anyone noticed, because each run is
        # individually cheap. Keyed per scope; the prompt text itself (evidence + SHAP
        # attributions) already changes whenever the inputs do.
        llm = get_llm(
            args.model,
            trace_component="hybrid",
            cache_dir=ROOT / "data" / "llm_cache" / f"hybrid_{args.scope}",
            cache_namespace=f"hybrid:{args.scope}",
            allow_fallback=False,  # a throttle must fail visibly, not silently swap models
        )
        print(f"model: {llm.name}")

        # Estimated and confirmed before the first billed call, as `run_eval.py` does. This
        # path previously went straight to a loop of paid calls with no estimate and no
        # prompt, its only backstop the spend cap, which raises mid-run, after partial spend,
        # and discards the work already paid for.
        if not llm.spec.is_local:
            n_calls = len(result.escalated)
            # ~1.2k prompt tokens per rationale (evidence row + five attributions + system),
            # ~180 out under the word-count instruction that fixed the truncation problem.
            projected = llm.spec.cost_usd(1_200 * n_calls, 180 * n_calls)
            print(
                f"\nPlanned: {n_calls} rationale calls on {llm.name}\n"
                f"Estimated cost: ${projected:.2f}\n"
            )
            if projected > 0 and not args.yes:
                reply = input("Proceed? [y/N] ").strip().lower()
                if reply not in ("y", "yes"):
                    print("Aborted before any call was made.")
                    return 0
        # Egress guard over every rationale, seeded with the clear corpus's real values so a
        # plaintext identifier is caught even when the vendor's classifier scores it 0.0 -
        # which it does for names outside its training distribution. Fails closed here:
        # this is the analyst path, and a rationale that cannot be scanned must not ship.
        from amlguard.guardrail import Guardrail, GuardrailUnavailable

        guardrail = None
        if not args.no_guardrail:
            try:
                guardrail = Guardrail.for_corpus(ROOT / "data" / "corpus" / "parties.json")
            except GuardrailUnavailable as exc:
                raise SystemExit(
                    f"{exc}\n\nRun with --no-guardrail to skip the egress check."
                ) from exc

        # The rationale/groundedness/egress loop is a library interface, not script inline:
        # decisions are annotated in place, the outcome carries the accounting, and the
        # invariants (basis equals what the model was shown, egress verdicts counted) are
        # pinned by tests behind that seam.
        from amlguard.review import review_head

        try:
            outcome = review_head(
                decisions=result.escalated,
                bundle=bundle,
                item_ids=item_ids,
                llm=llm,
                guardrail=guardrail,
                grain=args.grain,
            )
        except GuardrailUnavailable as exc:
            # The sidecar died MID-queue. Fail closed with the documented exit message,
            # not a raw traceback; already-paid calls are safe in the disk cache.
            raise SystemExit(
                f"{exc}\n\nEgress guard lost mid-run; completed rationales are cached. "
                f"Restart the guardrail and re-run, or use --no-guardrail."
            ) from exc
        result.llm_calls = outcome.llm_calls
        blocked, discounted = outcome.blocked, outcome.discounted
        ungrounded_count, explain_failures = outcome.ungrounded, outcome.explain_failures

        result.llm_cost_usd = llm.stats.total_cost_usd
        print(f"rationales generated: {result.llm_calls} | cost ${result.llm_cost_usd:.4f}")
        if outcome.skipped or outcome.generation_failures:
            # An item shipping without a rationale is a fact the operator must see, not
            # infer from a count mismatch.
            print(
                f"  {outcome.skipped} escalated item(s) had no in-window evidence; "
                f"{outcome.generation_failures} generation(s) degraded to a placeholder"
            )
        if guardrail is not None:
            # Persisted so the claim is checkable from the artifact, not only the log.
            # Unconditional: both grains' result types carry the fields now, the old
            # hasattr guard silently logged 0 for the transaction grain.
            result.egress_blocked = blocked
            result.egress_discounted = discounted
            print(
                f"egress: {blocked} withheld, {discounted} surrogate-key false positives "
                f"discounted (of {result.llm_calls} scanned)"
            )
        print(
            f"groundedness: {ungrounded_count}/{result.llm_calls} rationales carry at least "
            f"one assertion with no basis in their evidence (marked per decision)"
        )
        if explain_failures:
            print(
                f"  {explain_failures}/{result.llm_calls} rationales were generated without "
                f"SHAP attributions, their evidence is the transaction row alone"
            )
        example = next((d for d in result.escalated if d.rationale), None)
        if example:
            print(f"\nexample (rank {example.rank}, score {example.score:.2f}):")
            print(f"  {example.rationale[:320]}")

    tracker = Tracker("amlguard-hybrid")
    with tracker.run(
        f"hybrid:{args.scope}:{args.grain}:cap{args.capacity}",
        {
            "scope": args.scope,
            "grain": args.grain,
            "capacity": args.capacity,
            "model": args.model or "default",
            "classifier_hash": bundle.model_hash,
        },
    ):
        # Grain-correct precision. The previous call passed the transaction-id truth dict to
        # `AlertQueueResult.precision_at_capacity`, which matches *alert* ids, so the
        # MLflow-tracked precision was identically 0.0 at the default grain while the printed
        # number was right. Same method name, incompatible argument semantics; resolved by
        # computing the right one per grain here.
        if args.grain == "alert":
            tracked_precision = result.precision_at_capacity(escalating)
        else:
            tracked_precision = result.precision_at_capacity(truth)
        # One fact, one home. MLflow keeps the EXPERIMENT view: the queue-quality metrics
        # that are comparable across scope/capacity/model (precision, distinct subjects,
        # counts). The per-generation quality VERDICTS (grounded rate, egress blocks) are
        # Langfuse scores attached to the generations they judge, below, not duplicated here.
        # Cost/latency are Langfuse's (per-generation), not MLflow's.
        tracker.log_metrics({
            "precision_at_capacity": tracked_precision,
            "distinct_subjects": getattr(result, "distinct_subjects", lambda: 0)()
            if args.grain == "alert" else 0,
            "llm_calls": result.llm_calls,
            # A throttled run that degrades every rationale must not record as healthy in the
            # experiment ledger; these counts make the masquerade visible in comparison.
            "escalated_without_evidence": outcome.skipped,
            "generation_failures": outcome.generation_failures,
            "wall_seconds": time.monotonic() - run_started,
        })
        # Global SHAP importances for the classifier this queue was scored by, so the run in
        # the UI shows *why* the model ranks as it does, not only how well.
        try:
            from amlguard.explain import global_importance

            shap_importance = global_importance(
                model, features[test_idx], names
            )
            tracker.log_metrics(
                {f"shap.{k}": v for k, v in list(shap_importance.items())[:15]}
            )
        except Exception as exc:  # noqa: BLE001, telemetry, never fatal
            print(f"[tracking] SHAP logging failed: {type(exc).__name__}")
        tracker.log_artifact_json("hybrid.json", result.to_dict())

        # Run-level quality verdicts to Langfuse, beside the generations they judge. The
        # division of labour: MLflow keeps the experiment metrics above; Langfuse keeps the
        # prompt-level record, and these scores make its view queryable by outcome.
        from amlguard.observability import record_score

        if result.llm_calls:
            record_score(
                f"hybrid:{args.scope}:grounded_rate",
                1.0 - ungrounded_count / result.llm_calls,
                comment=f"{ungrounded_count} of {result.llm_calls} rationales carry "
                        f"ungrounded assertions",
            )
            record_score(
                f"hybrid:{args.scope}:egress_blocked",
                float(getattr(result, "egress_blocked", 0)),
            )
            record_score(
                f"hybrid:{args.scope}:precision_at_capacity", float(tracked_precision),
            )

    # Grain is in the filename: the two grains answer different questions and must not
    # overwrite each other. Alert grain keeps the plain name because it is the deployable
    # queue; transaction grain is retained for the comparison that motivated the change.
    suffix = "" if args.grain == "alert" else f"-{args.grain}"
    out = ROOT / "data" / "eval" / f"hybrid_{args.scope}{suffix}.json"

    # A rank-only run must never clobber a paid artifact. This exact overwrite has happened
    # three times in this project's history, a --no-llm smoke test silently replacing fifty
    # billed rationales with an empty queue, because the cheap run and the expensive run
    # shared a filename. Rank-only output goes to its own file; the paid artifact is only
    # ever replaced by another paid run.
    if args.no_llm:
        out = out.with_name(out.stem + "-ranking" + out.suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    from amlguard.persist import RUN_ID, atomic_write_json

    payload = result.to_dict()
    payload["run_id"] = RUN_ID
    atomic_write_json(out, payload)
    print(f"\nwritten to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
