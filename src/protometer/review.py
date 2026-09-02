"""The review head: rationale, groundedness marking, and egress for every escalated item.

This is the layer an analyst actually receives, and it used to live inline in
`scripts/run_hybrid.py` as ~100 lines of a `main()` no test could reach. Three real
defects incubated there for exactly that reason: a groundedness basis that omitted part
of the prompt, egress counts a `hasattr` guard silently dropped, and a client that
re-billed every invocation. Behind one interface, the invariants are pinnable by tests
instead of review rounds.

The interface is deliberately small: decisions in, counters out, everything else inside.
Decisions are annotated in place (rationale, attributions, ungrounded terms, provenance),
because they are the artifact; the returned `ReviewOutcome` is the accounting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from protometer.explain import explain_prediction
from protometer.hybrid import generate_rationale, ungrounded_terms
from protometer.log import get_logger

_log = get_logger("review")


@dataclass
class ReviewOutcome:
    """What happened across one pass over the review head."""

    llm_calls: int = 0
    blocked: int = 0
    discounted: int = 0
    ungrounded: int = 0
    explain_failures: int = 0
    # Escalated items that received no rationale because none of their evidence falls in
    # the scoring window. Previously invisible: the item shipped bare and only a careful
    # reader of "rationales generated < capacity" would notice.
    skipped: int = 0
    # Rationale calls that degraded to an error placeholder. Split out so a throttled run
    # cannot masquerade as a healthy one in the counters.
    generation_failures: int = 0


def review_head(
    *,
    decisions: list[Any],
    bundle: Any,
    item_ids: list[str],
    llm: Any,
    guardrail: Any = None,
    grain: str = "alert",
    progress: bool = True,
) -> ReviewOutcome:
    """Generate, ground-check, and egress-scan a rationale for every escalated decision.

    `bundle` is the `ClassifierBundle` the queue was ranked from, the same object, so
    attributions are computed by the exact model that produced the scores. `guardrail`
    of None skips the egress scan (a visible caller decision); a scan failure propagates,
    because this path fails closed: a rationale that cannot be scanned must not ship.
    """
    outcome = ReviewOutcome()
    by_id = {t["transaction_id"]: t for t in bundle.transactions}
    position = {item: i for i, item in enumerate(item_ids)}

    for decision in decisions:
        if not getattr(decision, "escalated", False):
            continue

        # An alert's evidence is its subject's strongest-scoring transaction; a
        # transaction-grain decision *is* that transaction. Both resolve to one row so
        # the attribution path is shared.
        if grain == "alert":
            evidence_ids = [
                i for i in decision.evidence_transaction_ids if i in position
            ]
            if not evidence_ids:
                outcome.skipped += 1
                continue
            anchor_id = evidence_ids[0]
        else:
            anchor_id = decision.item_id
        row = bundle.features[bundle.test_idx[position[anchor_id]]]

        try:
            explanation = explain_prediction(
                bundle.model, row, bundle.feature_names,
                bundle.features[bundle.train_idx],
            )
            attributions = [
                {"feature": a.feature, "value": a.value,
                 "contribution": a.contribution, "direction": a.direction}
                for a in explanation.top(5)
            ]
        except Exception as exc:  # noqa: BLE001, one row must not abort the queue
            # Counted and reported. A bare `except: attributions = []` meant a SHAP
            # failure on every row still produced a rationale for every item, still
            # billed, and looked identical to a healthy run.
            attributions = []
            outcome.explain_failures += 1
            if progress and outcome.explain_failures == 1:
                _log.warning("SHAP failed (%s: %s); rationales for affected rows carry no "
                             "attributions", type(exc).__name__, exc)

        txn = by_id[anchor_id]
        row_text = (
            f"{txn['value_date']} | {txn['origin_party_id']} -> "
            f"{txn['beneficiary_party_id']} | {txn['amount']} | "
            f"{txn['channel']} | {txn.get('memo') or '-'}"
        )
        if grain == "alert":
            # The case-file header an analyst opens with: which scenario fired, on whom,
            # how long is left on the filing clock, and what history the subject carries.
            evidence = (
                f"ALERT {decision.alert_id} | scenario {decision.scenario_id} | "
                f"subject {decision.subject_party_id}\n"
                f"raised {decision.raised_on}, {decision.days_remaining} days remaining "
                f"on the 30-day filing clock\n"
                f"prior alerts on this subject: {decision.prior_alerts} "
                f"({decision.linked_alerts} linked)\n"
                f"{decision.transaction_count} transactions in the lookback window; "
                f"strongest:\n{row_text}"
            )
        else:
            evidence = row_text

        generate_rationale(llm, decision, evidence, attributions)
        outcome.llm_calls += 1

        # A degraded generation carries an error placeholder, not model output. Running
        # the groundedness gate over "(rationale unavailable: ThrottlingException ...)"
        # flagged every integer in the exception text as an ungrounded assertion, so a
        # throttled run reported a depressed grounded-rate on top of its failures.
        if (decision.rationale or "").startswith("(rationale unavailable"):
            outcome.generation_failures += 1
            continue

        # Groundedness: what did the model assert that its inputs do not contain? The
        # basis must equal what the model was SHOWN, evidence, attributions, and the
        # prompt header's score and rank. A rationale citing "model score (0.51)" was
        # once flagged because the basis omitted the score the prompt itself supplied.
        basis = (
            evidence + " " + json.dumps(attributions)
            + f" model score {decision.score:.2f} rank {decision.rank}"
        )
        decision.ungrounded = ungrounded_terms(decision.rationale or "", basis)
        if decision.ungrounded:
            outcome.ungrounded += 1

        # Egress: the rationale is the only text this system puts in front of a human,
        # so it is the only place a leak could actually reach one.
        if guardrail is not None:
            verdict = guardrail.scan_response(decision.rationale or "")
            if verdict.blocked:
                outcome.blocked += 1
                # The withheld notice must never contain what was withheld. An earlier
                # version echoed the caught clear values into the replacement text, the
                # one string the system promises is identifier-free, visible to any role
                # and serialized into committable artifacts. Count and reason only.
                reason = (
                    f"{len(verdict.leaked_values)} forbidden value(s) detected"
                    if verdict.leaked_values else verdict.outcome
                )
                decision.rationale = f"[withheld: response failed the egress check, {reason}]"
                # Ungrounded terms were extracted from the pre-block text; disclosing
                # fragments of an unshippable rationale defeats the withholding.
                decision.ungrounded = []
            elif verdict.discounted:
                outcome.discounted += 1

    return outcome
