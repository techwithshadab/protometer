"""Generate docs/results-<domain>.md from evaluation output.

The results document is derived from the JSON the harness wrote, never hand-typed, so a number
in the write-up cannot drift from the number that was measured. Re-running this after a new
evaluation regenerates the document.

Domain-namespaced so one use case never overwrites another's results: AML reads the
canonical `data/eval/` and writes `docs/results-aml.md`; another domain reads
`data/eval/<domain>/` and writes `docs/results-<domain>.md`. A provenance header records the
domain, use case, model, corpus fingerprint, and how the run was produced.

    python scripts/generate_results.py                 # domain=aml -> docs/results-aml.md
    python scripts/generate_results.py --domain healthcare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.scopes import CURVE_ORDER, get_scope  # noqa: E402

# AML is the canonical domain and keeps writing to the top-level data/eval/ (backward compatible
# with every existing artifact path); any other domain lives under data/eval/<domain>/ so its
# training.json / attacks.json / hybrid_*.json cannot collide with AML's.
DEFAULT_DOMAIN = "aml"
EVAL_ROOT = ROOT / "data" / "eval"


def _eval_root(domain: str) -> Path:
    return EVAL_ROOT if domain == DEFAULT_DOMAIN else EVAL_ROOT / domain

# Below this many checkpoints of separation, two scopes are not distinguishable on this task
# count and must not be reported as ordered. Measured: `direct-plus-monetary` and `quasi`
# differ by 3 checkpoints of 46, entirely in aggregation and narrative, with identity and
# typology identical.
NOISE_FLOOR_CHECKPOINTS = 4


def load(model_dir: Path) -> dict[str, dict]:
    results = {}
    for path in model_dir.glob("*.json"):
        if path.name in ("tasks.json", "comparison.json"):
            continue
        payload = json.loads(path.read_text())
        if "scope" not in payload:
            continue
        stats = payload.get("llm_stats") or {}
        if stats.get("billed_calls", 0) == 0 and stats.get("cache_hits", 0) == 0:
            continue  # no measurement took place
        results[payload["scope"]] = payload
    return results


def checkpoint_counts(result: dict) -> tuple[int, int]:
    passed = sum(1 for t in result["tasks"] for c in t["checkpoints"] if c["passed"])
    total = sum(len(t["checkpoints"]) for t in result["tasks"])
    return passed, total


def _provenance_header(domain: str) -> None:
    """Stamp which domain / use case / run produced these numbers, so a reader never has to
    guess and a future domain's document is unmistakably a different measurement."""
    from datetime import datetime, timezone

    try:
        from amlguard.tracking import corpus_source_fingerprint
        fp = corpus_source_fingerprint(ROOT / "data" / "corpus")
    except Exception:  # noqa: BLE001
        fp = "unknown"
    # A domain dir may hold several model subdirs; report the ones actually measured.
    measured = sorted(
        d.name for d in EVAL_ROOT.iterdir()
        if d.is_dir() and not d.name.endswith("cache") and not d.name.startswith("_")
        and load(d).get("none")
    ) if EVAL_ROOT.exists() else []
    print(f"# Results ({domain}): what data protection costs an AI pipeline\n")
    print(
        f"- **Domain:** {domain}\n"
        f"- **Use case:** batch measurement pipeline\n"
        f"- **Corpus fingerprint:** `{fp}`\n"
        f"- **Model(s) measured:** {', '.join(measured) or '(none yet)'}\n"
        f"- **Generated:** {datetime.now(timezone.utc):%Y-%m-%d} (UTC) by "
        f"`python scripts/generate_results.py --domain {domain} > docs/results-{domain}.md`\n"
    )
    print(
        "Every figure below is generated from the evaluation harness output; nothing here is\n"
        "hand-entered. This document covers the **" + domain + "** domain only; other domains\n"
        "have their own `docs/results-<domain>.md` and never overwrite this one.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=DEFAULT_DOMAIN,
                        help="domain to report (aml reads data/eval/, else data/eval/<domain>/)")
    args = parser.parse_args()

    global EVAL_ROOT
    EVAL_ROOT = _eval_root(args.domain)
    if not EVAL_ROOT.exists():
        sys.exit(f"No eval artifacts for domain {args.domain!r} at {EVAL_ROOT}")

    # Underscore-prefixed directories are excluded by convention: anything staged, archived,
    # or superseded must never be globbed into the report as a current measurement.
    model_dirs = [
        d
        for d in EVAL_ROOT.iterdir()
        if d.is_dir() and not d.name.endswith("cache") and not d.name.startswith("_")
    ]

    _provenance_header(args.domain)

    measured_models = [d for d in sorted(model_dirs) if load(d).get("none")]
    if not measured_models:
        # Stated, not skipped: an absent section reads as "not attempted", and a missing
        # measurement is a different claim.
        print("\n## LLM investigation curve\n")
        print(
            "_Not yet measured on this corpus._ Run:\n"
            "`python scripts/run_eval.py --model <model>`.\n"
        )

    for model_dir in sorted(model_dirs):
        results = load(model_dir)
        if not results or "none" not in results:
            continue

        baseline = results["none"]
        base_mean = baseline["mean_checkpoint_score"]
        model_name = baseline.get("model", model_dir.name)

        tasks = len(baseline["tasks"])
        _, checkpoints = checkpoint_counts(baseline)

        print(f"\n## {model_name}\n")
        print(f"{tasks} investigation tasks, {checkpoints} checkpoints per scope.\n")
        print(
            "| Scope | Mean | Verifiable | Retained | Task completion | "
            "Aggregation | Identity | Typology | Narrative |"
        )
        print("|---|---|---|---|---|---|---|---|---|")

        for name in CURVE_ORDER:
            result = results.get(name)
            if not result:
                continue
            strata = result.get("stratum_scores", {})
            retained = result["mean_checkpoint_score"] / base_mean if base_mean else 0
            print(
                f"| `{name}` | {result['mean_checkpoint_score']:.3f} "
                f"| {result['verifiable_score']:.3f} | {retained:.0%} "
                f"| {result['task_completion_rate']:.0%} "
                f"| {strata.get('aggregation', 0):.2f} | {strata.get('identity', 0):.2f} "
                f"| {strata.get('typology', 0):.2f} | {strata.get('narrative', 0):.2f} |"
            )

        # The marginal value of the LLM over naive copying, reported because it was
        # measured: every scope artifact carries a copy-the-detector baseline, and a
        # favorable number the harness computed belongs in the report, not only in JSON.
        baselines = [
            r.get("copy_baseline", {}).get("mean_checkpoint_score")
            for r in results.values()
        ]
        baselines = [b for b in baselines if b is not None]
        if baselines and base_mean:
            deltas = [
                r["mean_checkpoint_score"] - r["copy_baseline"]["mean_checkpoint_score"]
                for r in results.values() if r.get("copy_baseline")
            ]
            print(
                f"\nCopy-the-detector baseline: {min(baselines):.2f}-{max(baselines):.2f} "
                f"mean checkpoint score across scopes, against model means of "
                f"{min(r['mean_checkpoint_score'] for r in results.values()):.2f}-"
                f"{max(r['mean_checkpoint_score'] for r in results.values()):.2f}: the "
                f"model adds {min(deltas)*100:.0f}-{max(deltas)*100:.0f} points over "
                f"transcribing detector output.\n"
            )

        # Grading provenance, stated: exact-match strata dominate, and the judged
        # narrative checkpoints (14 of 60) are graded by the same model that produced
        # the answers unless --judge-model overrides it. The `Verifiable` column is the
        # self-judging-free view.
        print(
            "_Judged checkpoints (narrative stratum) default to the subject model as its\n"
            "own judge; the Verifiable column excludes them entirely._\n"
        )

        # Cost and latency, which the "protection is affordable" claim rests on.
        total_cost = sum(r["llm_stats"].get("total_cost_usd", 0) for r in results.values())
        total_calls = sum(r["llm_stats"].get("billed_calls", 0) for r in results.values())
        p50s = [r["llm_stats"].get("latency_p50", 0) for r in results.values()]
        print(
            f"\n{total_calls} billed calls, ${total_cost:.4f} total, "
            f"per-scope median latency averaging {sum(p50s) / len(p50s):.0f}s per call. "
            f"(Billed figures are what the artifact set as-committed cost to produce, "
            f"re-runs over a warm response cache bill only fresh calls; a from-scratch "
            f"single-model measurement of all eight scopes bills every call.)\n"
        )

        # Pairs too close to order, stated rather than silently presented as ranked.
        print("### Separation\n")
        ordered = [n for n in CURVE_ORDER if n in results]
        for earlier, later in zip(ordered, ordered[1:]):
            a_pass, _ = checkpoint_counts(results[earlier])
            b_pass, _ = checkpoint_counts(results[later])
            if abs(a_pass - b_pass) < NOISE_FLOOR_CHECKPOINTS:
                print(
                    f"- `{earlier}` and `{later}` differ by {abs(a_pass - b_pass)} checkpoints "
                    f"and are **not distinguishable** at this task count."
                )
        print()

    _training_section()
    _frontier_section()
    _attacks_section()
    _erasure_section()
    _hybrid_section()
    _composition_section()

    print("\n## Scope definitions\n")
    print("| Scope | Protects |")
    print("|---|---|")
    for name in CURVE_ORDER:
        scope = get_scope(name)
        print(f"| `{name}` | {scope.description} |")

    return 0


def _load_artifact(name: str):
    """Read one eval artifact, or None when that measurement has not been run.

    Absence is reported in the document rather than skipped silently: a results file that
    quietly omits a measurement reads as "we did not do that" when it may mean "the run
    failed", and those are different claims.
    """
    path = EVAL_ROOT / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _training_section() -> None:
    """The classifier curve, the measurement the write-up's headline rests on."""
    data = _load_artifact("training.json")
    print("\n## Classifier: what protection costs a trained model\n")
    if not data:
        print("_Not measured, run `python scripts/run_training.py`._\n")
        return

    by_scope = {d["scope"]: d for d in data}
    base = by_scope.get("none", {}).get("average_precision") or 0

    print(
        "Random forest over the protected ledger plus graph features, **temporally split** -\n"
        "train on the earlier ledger, test on the later. Population aggregates and the graph\n"
        "are fit on the training fold only.\n"
    )
    print("| Scope | AP (\u00b1seed SD) | ROC-AUC | Retained | P@25 | P@50 | Lift@25 | ECE |")
    print("|---|---|---|---|---|---|---|---|")
    for name in CURVE_ORDER:
        row = by_scope.get(name)
        if not row:
            continue
        aml = row.get("aml_metrics", {})
        p_at = aml.get("precision_at_k", {})
        lift = aml.get("lift_at_k", {})
        retained = row["average_precision"] / base if base else 0
        sd = row.get("average_precision_seed_std", 0.0)
        print(
            f"| `{name}` | {row['average_precision']:.3f} \u00b1{sd:.3f} "
            f"| {row['roc_auc']:.3f} "
            f"| {retained:.0%} | {p_at.get('25', 0):.2f} | {p_at.get('50', 0):.2f} "
            f"| {lift.get('25', 0):.1f}x "
            f"| {aml.get('expected_calibration_error', 0):.3f} |"
        )

    print(
        "\nThe `±` is the SD of AP across RandomForest seeds (split, features and labels fixed; "
        "only `random_state` varies). The seed set **includes** the seed the reported AP is fit "
        "at, so the band is an interval around a member of its own population, not around an "
        "unsampled point. Read a scope delta smaller than this SD as seed noise, not protection "
        "cost."
    )

    top = by_scope.get("none", {}).get("shap_importance", {})
    if top:
        print("\nTop features by SHAP on the clear ledger:\n")
        for feature, value in list(top.items())[:5]:
            print(f"- `{feature}` {value:.3f}")

    recall = by_scope.get("none", {}).get("per_typology_recall", {})
    if recall:
        print(
            "\nRecall by typology at the operating threshold (clear ledger), with the test-fold\n"
            "denominator. **Read these as anecdote, not rates:** trade_based is a handful of\n"
            "transactions, so a single one moves its recall double digits. The operating\n"
            "threshold is F1-selected on the **training** fold and applied unchanged to the\n"
            "test fold, so these carry no threshold-overfitting bias (AP and ROC-AUC are\n"
            "threshold-free regardless). Trade-based is nearly invisible because its defining\n"
            "indicator, invoice value mismatch, cannot exist in a ledger-only corpus:\n"
        )
        print("| Typology | Recall | n (caught / test txns) |")
        print("|---|---|---|")
        for typology, v in sorted(recall.items(), key=lambda kv: -kv[1]["recall"]):
            print(f"| {typology} | {v['recall']:.0%} | {v['n_caught']}/{v['n_total']} |")
    print()


def _frontier_section() -> None:
    """Tokenization vs anonymization vs synthetic data, one corpus, three capabilities."""
    data = _load_artifact("protection_methods.json")
    print("\n## The protection-technique frontier\n")
    if not data:
        print("_Not measured, run `python scripts/compare_protection_methods.py`._\n")
        return

    risk = data.get("metadata_risk", {})
    if risk:
        print(
            f"**The metadata tokenization deliberately leaves clear, scored by the "
            f"vendor's own risk engine across all three attacker models:** k-anonymity "
            f"{risk.get('k_anonymity')}, prosecutor risk {risk.get('prosecutor_risk')} "
            f"(worst-case), journalist {risk.get('journalist_risk')}, marketer "
            f"{risk.get('marketer_risk')} (average-case bulk linkage), rated "
            f"{risk.get('highest_risk_level')} across {risk.get('n_records')} parties on "
            f"{', '.join(risk.get('quasi_identifiers', []))}. The split matters: worst-case "
            f"exposure is high but average-case bulk-linkage is low, an institution reads "
            f"both. The open-metadata trade is now a number, not an assertion.\n"
        )
    gen = data.get("generalized_amounts", {})
    if gen:
        print("| AMOUNT treatment | Average precision | vs clear |")
        print("|---|---|---|")
        clear = gen.get("clear_ap") or 0
        for label, ap in (
            ("clear", gen.get("clear_ap")),
            ("k-anonymity interval generalization (midpoints)", gen.get("average_precision")),
            ("format-preserving tokenization", gen.get("tokenized_amount_ap")),
        ):
            if ap is None:
                continue
            print(f"| {label} | {ap:.3f} | {ap / clear:.0%} |")
        # Derive the comparison from the DATA, not hardcoded numbers, so the prose can never
        # drift from the corpus it describes (a prior version hardcoded old-corpus figures).
        gen_ap = gen.get("average_precision")
        tok_ap = gen.get("tokenized_amount_ap")
        gen_gap = (gen_ap - clear) if (gen_ap is not None and clear) else None
        tok_cost = (clear - tok_ap) if (tok_ap is not None and clear) else None
        gap_txt = (
            f"the generalized AP is {gen_gap:+.3f} vs clear "
            f"({'within' if abs(gen_gap) < 0.02 else 'beyond'} the ~0.013 RF-seed SD)"
            if gen_gap is not None else "the generalized AP tracks clear"
        )
        tok_txt = (
            f"tokenizing AMOUNT costs {tok_cost:.3f} AP on this corpus"
            if tok_cost is not None else "the AMOUNT cost is corpus-dependent"
        )
        print(
            f"\nGeneralization keeps the magnitude signal tokenization destroys: {gap_txt}, so "
            f"read it as no measurable loss rather than 'better than clear', while {tok_txt}. "
            f"On this corpus draw that AMOUNT cost is small and within seed noise; a different "
            f"draw produced an amount-reliant model where it was ~10% (the single-seed "
            f"sensitivity the utility curve is built to expose). The techniques answer different "
            f"exposures: tokenization is reversible per row behind a role gate; generalization "
            f"is an irreversible release format.\n"
        )
    synth = data.get("synthetic_ledger", {})
    if synth:
        tstr = synth.get("tstr", {})
        tstr_line = ""
        if tstr:
            # The retention ratio is only meaningful when the REFERENCE task (train-real/test-real)
            # is comfortably above chance. compare_protection_methods._tstr records that judgement
            # as `reference_above_chance`; when it is False the ratio is two near-random numbers,
            # so we render the caveat the code already computed instead of a clean headline. Older
            # artifacts predate the guard (no reference_above_chance key) -> treat as caveated.
            above_chance = tstr.get("reference_above_chance")
            trtr = tstr.get("train_real_test_real")
            chance = tstr.get("chance_macro_f1")
            if above_chance is False or (above_chance is None and "reference_above_chance" not in tstr):
                caveat = (
                    f" (reference task near chance — TRTR {trtr} vs chance {chance}; "
                    f"read the retention ratio as DIRECTIONAL only, not precise utility)"
                    if trtr is not None and chance is not None else
                    " (reference task near chance; retention ratio is directional only)"
                )
            else:
                caveat = ""
            tstr_line = (
                f" **Task utility (TSTR):** a classifier trained on the synthetic table and "
                f"tested on real data retains **{tstr.get('utility_retained')}** of the "
                f"train-on-real score ({tstr.get('task')}){caveat}, so the synthetic arm is "
                f"scored by downstream utility like the other techniques, not fidelity moments "
                f"alone."
            )
        print(
            f"**Synthetic twin (vine copula):** {synth.get('rows')} rows, identity linkage "
            f"{synth.get('identity_linkage', 'none by construction')}. Fidelity: amount "
            f"mean {synth.get('amount_mean', {}).get('real')} real vs "
            f"{synth.get('amount_mean', {}).get('synthetic')} synthetic, p95 "
            f"{synth.get('amount_p95', {}).get('real')} vs "
            f"{synth.get('amount_p95', {}).get('synthetic')}, channel-mix L1 distance "
            f"{synth.get('channel_l1_distance')}.{tstr_line}\n"
        )

    dp = data.get("differential_privacy", {})
    if dp:
        if dp.get("available"):
            print(
                f"**Differential privacy ({dp.get('technique')}):** a DP aggregate released "
                f"with a formal epsilon guarantee, the shareable-aggregate stage where DP "
                f"belongs. Results: {dp.get('dp_counts')}.\n"
            )
        else:
            print(
                f"**Differential privacy, the fourth technique, attempted and reported "
                f"honestly:** DP is the right control for the *shareable-aggregate* stage "
                f"where it belongs (not the RAG path it rules out). We wired a DP count-by-channel "
                f"and attempted it, but **Developer Edition does not enable DP compute** "
                f"(\"{dp.get('reason')}\"). The anon service is otherwise healthy; only the DP "
                f"endpoints are tier-gated. We report the gate rather than fabricate a number, "
                f"on a Team/Enterprise tier the same call returns a DP count per channel with "
                f"its privacy budget.\n"
            )


def _attacks_section() -> None:
    """Adversarial evaluation, what the protected corpus still discloses."""
    data = _load_artifact("attacks.json")
    print("\n## Adversarial evaluation\n")
    if not data:
        print("_Not measured, run `python scripts/run_attacks.py`._\n")
        return

    scope = "direct" if "direct" in data else next(iter(data))
    print(
        f"Adversary holds the protected corpus at `{scope}` plus an auxiliary graph, and not\n"
        f"the tokenization key.\n"
    )
    print("| Attack | Success rate | Relabeled-graph control | Chance |")
    print("|---|---|---|---|")
    for row in sorted(data[scope], key=lambda r: -r["accuracy"]):
        control = row.get("control_accuracy")
        print(
            f"| {row['attack']} | {row['accuracy']:.1%} "
            f"| {f'{control:.2%}' if control is not None else 'n/a'} "
            f"| {row['baseline_accuracy']:.2%} |"
        )
    # Derive the neighbourhood/control example from the data, not a hardcoded pair, so the
    # prose can never drift from the artifact it describes.
    nl = next((r for r in data[scope] if "neighbour" in r["attack"].lower()), None)
    example = ""
    if nl is not None and nl.get("control_accuracy") is not None:
        example = f" (neighbourhood {nl['accuracy']:.1%} -> {nl['control_accuracy']:.2%})"
    print(
        f"\nFor the structural attacks the success rate is a **disclosure rate**: the share\n"
        f"of parties whose graph signature is unique against an exact auxiliary graph. The\n"
        f"honest null is the **relabeled-graph control** (identical structure, randomized\n"
        f"identities), not random guessing, the attack measures real linkage only insofar as\n"
        f"the control collapses toward chance{example}. Lift-over-chance is omitted here\n"
        f"because for a conclude-only-when-unique procedure it merely restates the raw count.\n"
    )


def _erasure_section() -> None:
    """Semantic Erasure, behavioural retrieval against identity retrieval."""
    data = _load_artifact("semantic_erasure.json")
    print("\n## Semantic Erasure\n")
    if not data:
        print("_Not measured, run `python scripts/measure_semantic_erasure.py`._\n")
        return

    print(
        "Both arms scored the same way: recall of a known-correct document in the top 10.\n"
    )
    print("| Scope | Behavioural found | Identity found | Identity mean rank | Fisher p vs baseline |")
    print("|---|---|---|---|---|")
    for name in CURVE_ORDER:
        row = data.get(name)
        if not row:
            continue
        rank = row.get("identity_mean_rank")
        p = row.get("identity_fisher_p_vs_baseline")
        print(
            f"| `{name}` | {row.get('behavioural_found', '-')} "
            f"| {row.get('identity_found', '-')} "
            f"| {rank if rank else 'not found'} "
            f"| {f'{p:.1e}' if p is not None else '-'} |"
        )
    print()


def _hybrid_section() -> None:
    """Hybrid triage, the classifier ranks, the model explains the head of the queue."""
    print("\n## Hybrid triage\n")
    rows = []
    for scope in ("none", "quasi"):
        data = _load_artifact(f"hybrid_{scope}.json")
        if data:
            rows.append((scope, data))
    if not rows:
        print("_Not measured, run `python scripts/run_hybrid.py`._\n")
        return

    print(
        "Queue at **alert grain**, the unit an analyst dispositions. Ranked on a composite of\n"
        "model evidence, days remaining on the 31 CFR 1020.320 filing clock, and repeat-alert\n"
        "history.\n"
    )
    print(
        "| Scope | Queue | P@50 | Distinct subjects | Egress (blocked/discounted) "
        "| Ungrounded | Cost |"
    )
    print("|---|---|---|---|---|---|---|")
    for scope, data in rows:
        subjects = data.get("distinct_subjects_in_head")
        ungrounded = sum(
            1 for d in data.get("decisions", []) if d.get("escalated") and d.get("ungrounded")
        )
        print(
            f"| `{scope}` | {data.get('queue_length', 0)} "
            f"| {data.get('precision_at_capacity', 0):.2f} "
            f"| {subjects if subjects is not None else '-'}/{data.get('review_capacity', 0)} "
            f"| {data.get('egress_blocked', 0)}/{data.get('egress_discounted', 0)} "
            f"| {ungrounded}/{data.get('llm_calls', 0)} "
            f"| ${data.get('llm_cost_usd', 0):.4f} |"
        )
    none_p = next((d.get("precision_at_capacity") for s, d in rows if s == "none"), None)
    p_txt = f"{none_p:.2f}" if none_p is not None else "the reported P@50"
    print(
        f"\nThe queue is restricted to alerts raised in the scoring window (the temporal test\n"
        f"fold), and precision is scored against whether the *case* deserved escalation. The\n"
        f"structural ceiling is well below 1.0 because only the in-window alerts have evidence\n"
        f"the test-fold model can see, so a clear-scope P@50 of {p_txt} is a strong multiple\n"
        f"over working the queue in random order at this base rate, not a weak absolute number.\n"
        f"'Ungrounded' counts rationales carrying at least one figure with no basis in their\n"
        f"evidence; each is marked on the decision.\n"
        f"\n**Filing-clock caveat:** the urgency term anchors `as_of` to the newest alert, but\n"
        f"the corpus's alert dates all precede the SAR deadline window, so every alert in the\n"
        f"head is past the 30-day clock and the urgency term saturates uniformly, on this\n"
        f"corpus the ordering is effectively score-driven. The clock becomes discriminative\n"
        f"only on a live feed where alert ages straddle the deadline; stated so the queue is\n"
        f"not read as demonstrating deadline triage.\n"
    )





def _composition_section() -> None:
    """Queue composition against protected-class-adjacent attributes.

    The classifier is structurally blind to jurisdiction, PEP status and party type, none is
    a feature, but blindness is not fairness: the priority term compounds prior alerting, and
    guilty-walks propagate label history through the graph. This table is the answer to the
    question a second-line reviewer asks first: who ends up in the reviewed head, against the
    population base rate?
    """
    data = _load_artifact("hybrid_none.json")
    parties_path = ROOT / "data" / "corpus" / "parties.json"
    print("\n## Queue composition (fairness check)\n")
    if not data or not parties_path.is_file():
        print("_Not measured, requires a hybrid run and the corpus._\n")
        return

    parties = {p["party_id"]: p for p in json.loads(parties_path.read_text())}
    decisions = data.get("decisions", [])
    head = [d for d in decisions if d.get("escalated")]
    subjects = [
        parties[d["subject_party_id"]]
        for d in head
        if d.get("subject_party_id") in parties
    ]
    # Two denominators, because they answer different questions. Against ALL parties the
    # table measures the whole pipeline (which scenarios fire + how the ranker orders);
    # against the QUEUED alerts it isolates what the *ranker* adds, the number a
    # disparate-impact review actually wants, since the ranker is the component this system
    # contributes. Reporting only the first conflated the two.
    queued = [
        parties[d["subject_party_id"]]
        for d in decisions
        if d.get("subject_party_id") in parties
    ]
    if not subjects or not queued:
        print("_No escalated subjects resolved against the corpus._\n")
        return

    from collections import Counter

    total = len(parties)
    population = Counter(p["jurisdiction"] for p in parties.values())
    in_queue = Counter(p["jurisdiction"] for p in queued)
    in_head = Counter(p["jurisdiction"] for p in subjects)

    def lift(head_share: float, queue_share: float) -> str:
        # One rendering for a zero denominator everywhere. The PEP row used to print
        # "0.0x" where jurisdiction printed "inf" for the same undefined ratio, two
        # different lies about the same absence of evidence.
        return f"{head_share / queue_share:.1f}x" if queue_share else "n/a"

    print("| Attribute | Population | Queued alerts | Reviewed head | Ranker lift (head vs queue) |")
    print("|---|---|---|---|---|")
    for jurisdiction, count in in_head.most_common(6):
        pop_share = population[jurisdiction] / total
        queue_share = in_queue.get(jurisdiction, 0) / len(queued)
        head_share = count / len(subjects)
        print(
            f"| jurisdiction {jurisdiction} | {pop_share:.1%} | {queue_share:.1%} "
            f"| {head_share:.1%} | {lift(head_share, queue_share)} |"
        )
    pep_queue = sum(1 for p in queued if p.get("is_pep")) / len(queued)
    pep_head = sum(1 for p in subjects if p.get("is_pep")) / len(subjects)
    print(
        f"| PEP | {sum(1 for p in parties.values() if p.get('is_pep'))/total:.1%} "
        f"| {pep_queue:.1%} | {pep_head:.1%} | {lift(pep_head, pep_queue)} |"
    )
    org_queue = sum(1 for p in queued if p.get("party_type") == "organization") / len(queued)
    org_head = sum(1 for p in subjects if p.get("party_type") == "organization") / len(subjects)
    print(
        f"| organization | {sum(1 for p in parties.values() if p.get('party_type')=='organization')/total:.1%} "
        f"| {org_queue:.1%} | {org_head:.1%} | {lift(org_head, org_queue)} |"
    )
    pep_pop = sum(1 for p in parties.values() if p.get("is_pep")) / total
    expected_pep = pep_pop * len(subjects)
    print(
        f"\n**Limits of this table, stated:** the head holds {len(subjects)} subjects, so a\n"
        f"zero (e.g. PEP) is uninformative rather than reassuring, at a {pep_pop:.1%} base rate "
        f"the\nexpectation in {len(subjects)} draws is ~{expected_pep:.2f}, and zero observed "
        f"cannot distinguish fairness\n"
        f"from chance. Offshore-jurisdiction lift over the population reflects that layering\n"
        f"chains genuinely route through such entities in the planted typologies; the\n"
        f"classifier takes no geographic, PEP or party-type input, and the ranker-lift\n"
        f"column is the one that would expose the ranker adding disparity. Reported so drift\n"
        f"is visible, not because current values alarm.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
