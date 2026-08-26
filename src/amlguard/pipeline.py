"""The investigation pipeline, retrieval, inference over protected values, re-identification.

This is where the architecture's claim becomes observable. An investigation runs in three
stages, and the ordering is the whole point:

  1. **Retrieve** token-bearing chunks from the protected index and protected transactions.
  2. **Reason** over those tokens. The prompt sent to the model contains no real identifier,
     and `InvestigationResult.prompt` keeps a verbatim copy so that claim can be inspected
     rather than trusted.
  3. **Re-identify** only the model's output, gated by the viewer's role.

Nothing unprotects before step 3. If it did, the system would be protecting data at rest
while exposing it at the moment of highest risk, the failure mode the hackathon's "Protect
Data In Use" track exists to exclude.

The model is told the tokens are stable pseudonyms. That instruction matters: without it,
models tend to treat opaque strings as noise and refuse to reason. With it, they use tokens
as identity anchors, which is exactly what deterministic tokenization makes possible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from amlguard.llm import LLMClient, extract_json
from amlguard.protect import Protector
from amlguard.reidentify import AUDITOR, Role, reidentify, strip_tags
from amlguard.retrieval import NarrativeIndex, RetrievedChunk

# SYSTEM_PROMPT moved to config/prompts/amlguard-investigation-system.txt (loaded via observability.managed_prompt).


@dataclass
class InvestigationResult:
    """One completed investigation, with every stage kept inspectable."""

    task_id: str
    question: str
    # Verbatim payload sent to the model. This is the evidence for the central claim, and
    # the demo shows it on screen unedited.
    prompt: str
    raw_completion: str
    answer: dict[str, Any]
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    # Populated only when a role re-identifies the output.
    analyst_view: str = ""
    reidentified: int = 0
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.error


def format_transactions(transactions: list[dict], limit: int = 40) -> str:
    """Render transactions as a compact table.

    Amounts and dates are included as they appear in the protected corpus, tokenized at
    wider scopes, so the model sees exactly what protection left behind.
    """
    if not transactions:
        return "(no transactions found)"

    rows = ["date | origin | beneficiary | amount | channel | memo"]
    for txn in transactions[:limit]:
        rows.append(
            " | ".join(
                (
                    str(txn.get("value_date", "")),
                    str(txn.get("origin_party_id", "")),
                    str(txn.get("beneficiary_party_id", "")),
                    str(txn.get("amount", "")),
                    str(txn.get("channel", "")),
                    str(txn.get("memo", "")) or "-",
                )
            )
        )
    if len(transactions) > limit:
        rows.append(f"... and {len(transactions) - limit} more transactions")
    return "\n".join(rows)


def format_chunks(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no case notes retrieved)"
    return "\n\n".join(
        f"[{chunk.document_id}] {strip_tags(chunk.text)}" for chunk in chunks
    )


@dataclass
class InvestigationPipeline:
    """Runs investigations against one protected corpus.

    One pipeline per [[Protection Scope]]: the corpus, the index and therefore the answers
    all differ by scope, and that difference is what the evaluation measures.
    """

    protected_dir: Path
    index: NarrativeIndex
    llm: LLMClient
    scope_name: str

    # Ledger the deterministic detectors read. Supplying the **clear** corpus here separates
    # detector degradation from model degradation, which is otherwise confounded.
    #
    # The detectors parse amounts with `Decimal`; once AMOUNT is tokenized every rule returns
    # nothing and the candidate section disappears from the prompt entirely, measured, exact
    # detection falls from 26/28 to 11/28 at `quasi`. Since three of five typology checkpoints
    # score figures the prompt instructs the model to copy from those candidates, a curve run
    # this way measures the detectors at least as much as the model.
    #
    # With a clear detection ledger, every scope receives *identical* candidate input and the
    # only thing varying is what the model sees. That isolates the LLM's marginal contribution,
    # which is the quantity this project claims to measure.
    #
    # Left as None, detection runs on the protected ledger, the end-to-end configuration a
    # real deployment would have. Both are legitimate; they answer different questions, and the
    # results must say which was used.
    detection_dir: Path | None = None

    # The use-case this pipeline serves. Selects the system prompt (and, for a serving wrapper,
    # the guardrail model). Defaults to the AML domain, so the measurement path is unchanged.
    domain: "Any | None" = None

    _parties: list[dict] = field(default_factory=list, repr=False)
    _transactions: list[dict] = field(default_factory=list, repr=False)
    _detection_transactions: list[dict] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._parties = json.loads((self.protected_dir / "parties.json").read_text())
        self._transactions = json.loads((self.protected_dir / "transactions.json").read_text())
        # Party -> transactions touching it, built once. `transactions_for_party` and every
        # hop of `transaction_network` were full-ledger scans: O(hops x T) per expansion and
        # O(T) per lookup, called per task and per investigation. Fine at 6.8k rows, dominant
        # at 100x, and an index that mirrors how the data is actually queried is the honest
        # structure for it.
        self._by_party: dict[str, list[dict]] = {}
        for _txn in self._transactions:
            self._by_party.setdefault(_txn["origin_party_id"], []).append(_txn)
            self._by_party.setdefault(_txn["beneficiary_party_id"], []).append(_txn)
        detection_source = self.detection_dir or self.protected_dir
        self._detection_transactions = (
            self._transactions
            if detection_source == self.protected_dir
            else json.loads((detection_source / "transactions.json").read_text())
        )

    # -- retrieval helpers -------------------------------------------------------

    def transactions_for_party(self, party_id: str) -> list[dict]:
        """All transactions touching a party.

        Party *ids* are deliberately never protected, they are internal surrogate keys, not
        identifiers of a person, and keeping them queryable is what lets an investigator open
        a case at all. Semantic search cannot find a party by name once tokenized, so structured lookup by id is the intended entry point.
        """
        return list(self._by_party.get(party_id, ()))

    def transaction_network(
        self,
        party_id: str,
        hops: int = 2,
        limit: int = 120,
        source: list[dict] | None = None,
    ) -> list[dict]:
        """Transactions within `hops` of a party, following the money outward.

        A subject-only view cannot represent multi-hop typologies. A layering chain of three
        transfers typically touches the subject in only the first, leaving the remaining hops
        structurally invisible, the task becomes unanswerable regardless of protection, which
        would measure the retrieval design rather than tokenization.

        Expanding outward is also what an investigator actually does: follow the money. The
        cost is a wider prompt containing more unrelated activity, which is realistic and
        which the task instructions account for.
        """
        # Frontier expansion over the party adjacency index: work is proportional to the
        # transactions actually reached, not hops x whole-ledger. When an explicit `source`
        # ledger is supplied (the detection-confound path), a local index is built for it -
        # correctness first, the prebuilt index only for the common case.
        if source is None:
            by_party = self._by_party
        else:
            by_party = {}
            for txn in source:
                by_party.setdefault(txn["origin_party_id"], []).append(txn)
                by_party.setdefault(txn["beneficiary_party_id"], []).append(txn)

        frontier = {party_id}
        seen_parties = {party_id}
        collected: dict[str, dict] = {}

        for _ in range(max(1, hops)):
            next_frontier: set[str] = set()
            for party in frontier:
                for txn in by_party.get(party, ()):
                    collected[txn["transaction_id"]] = txn
                    for endpoint in (txn["origin_party_id"], txn["beneficiary_party_id"]):
                        if endpoint not in seen_parties:
                            seen_parties.add(endpoint)
                            next_frontier.add(endpoint)
            if not next_frontier:
                break
            frontier = next_frontier

        # Chronological order: temporal adjacency is itself evidence for layering and
        # structuring, and shuffling it would hide the pattern.
        ordered = sorted(collected.values(), key=lambda t: (t["value_date"], t["transaction_id"]))
        if len(ordered) <= limit:
            return ordered

        # Over the cap, filter by analytical relevance rather than truncating arbitrarily.
        #
        # Arbitrary truncation was measurably wrong: two-hop networks run to 127-286
        # transactions, so a flat cap discarded 50-80% of the ledger and the model was scored
        # against counts it could not see. Sending everything is equally bad, it buries the
        # pattern in unrelated activity and wastes context.
        #
        # Direct transactions are always kept: every checkpoint about the subject's own
        # activity depends on them. Indirect ones are kept when they could plausibly be a
        # chain hop, which is what multi-hop typologies are made of.
        direct = [
            t for t in ordered if party_id in (t["origin_party_id"], t["beneficiary_party_id"])
        ]
        direct_ids = {t["transaction_id"] for t in direct}
        counterparties = {
            t["beneficiary_party_id"] if t["origin_party_id"] == party_id else t["origin_party_id"]
            for t in direct
        }

        # A chain hop moves value onward *between* two of the subject's counterparties -
        # exactly the shape layering and round-tripping take.
        chain_candidates = [
            t
            for t in ordered
            if t["transaction_id"] not in direct_ids
            and t["origin_party_id"] in counterparties
            and t["beneficiary_party_id"] in counterparties
        ]

        remaining = max(0, limit - len(direct))
        kept = direct + chain_candidates[:remaining]

        # Backfill any leftover budget with the largest remaining transactions: value size is
        # the next most informative signal once structural relevance is exhausted.
        if len(kept) < limit:
            kept_ids = {t["transaction_id"] for t in kept}
            leftovers = sorted(
                (t for t in ordered if t["transaction_id"] not in kept_ids),
                key=lambda t: Decimal(str(t.get("amount", "0")))
                if str(t.get("amount", "0")).replace(".", "").isdigit()
                else Decimal("0"),
                reverse=True,
            )
            kept.extend(leftovers[: limit - len(kept)])

        return sorted(kept, key=lambda t: (t["value_date"], t["transaction_id"]))

    def candidate_patterns(self, party_id: str, hops: int = 2) -> list[dict]:
        """Deterministically detect candidate typologies and compute their exact figures.

        LLMs are unreliable at precise arithmetic over many rows, measured here, a 14B model
        summing a 14-row ledger returned $47,596 across 10 transactions where the answer was
        $66,733 across 7, because it swept in superficially similar benign rows.

        A production AML system would not ask a model to do that arithmetic. Detection rules
        and aggregation are deterministic code; the model's job is to *interpret* the
        candidates, decide which pattern fits, weigh the narrative evidence, and explain the
        conclusion. That division is both better engineering and a cleaner measurement: it
        isolates protection's effect on **reasoning** rather than confounding it with the
        model's arithmetic weakness.

        Crucially this runs entirely on **protected** data. Amounts and dates are read as they
        appear in the corpus, so at scopes where they are tokenized these computations degrade
        exactly as the model's would, which is itself part of the finding.
        """
        # Three hops, not two: a layering chain of N transfers spans N hops from the subject,
        # and a two-hop view provably cannot contain the tail of a three-transfer chain, the
        # final hop was measured absent from the network entirely, so the pattern was
        # undetectable rather than merely hard to spot.
        # Detection runs over the **whole ledger**, not a neighbourhood.
        #
        # This is stated plainly because the hop parameter is misleading at this corpus
        # density: measured, a five-hop expansion returns 2182 of 2182 transactions for every
        # subject, and even three hops reaches ~2150. Counterparty activity follows a Zipf
        # distribution, so a handful of high-degree parties connect everything within a few
        # steps and "neighbourhood" stops meaning anything past two hops.
        #
        # Accuracy justifies it rather than habit: 18/28 planted sets detected at two hops,
        # 25/28 at three, 26/28 at five. But the honest description of what the detector reads
        # is "everything", and the cost is negligible (~1.7ms per call against 25-30s of
        # inference), so the hop count is a bound that no longer binds.
        #
        # Chain depth is limited separately inside `follow`, which is what actually prevents
        # unbounded search.
        network = self.transaction_network(
            party_id, hops=max(hops, 5), limit=1200, source=self._detection_transactions
        )
        candidates: list[dict] = []

        def amount_of(txn: dict) -> Decimal | None:
            try:
                return Decimal(str(txn.get("amount", "")))
            except (ArithmeticError, ValueError, TypeError):
                # Tokenized at wider scopes, unparseable, and that is the point.
                return None

        # -- structuring: repeated deposits below the reporting threshold ---------------
        deposits = [
            t
            for t in network
            if t["origin_party_id"] == party_id
            and "deposit" in str(t.get("channel", "")).lower()
        ]
        under_threshold = [
            t for t in deposits if (a := amount_of(t)) is not None and a < Decimal("10000")
        ]
        if len(under_threshold) >= 3:
            total = sum(amount_of(t) or Decimal(0) for t in under_threshold)
            candidates.append(
                {
                    "pattern": "structuring",
                    "transaction_ids": [t["transaction_id"] for t in under_threshold],
                    "transaction_count": len(under_threshold),
                    "total_amount": str(total),
                    "counterparties": sorted({t["beneficiary_party_id"] for t in under_threshold}),
                    "evidence": (
                        f"{len(under_threshold)} deposits, each below the $10,000 reporting "
                        f"threshold, totalling {total}"
                    ),
                }
            )

        # -- funnel account: many depositors converge on one collector -------------------
        #
        # FinCEN FIN-2014-A005: multiple sub-threshold cash deposits into one account from
        # several depositors, withdrawn shortly afterwards. Distinct from structuring in shape
        #, structuring is one party splitting one sum, a funnel is many parties converging -
        # so a rule keyed on per-party deposit counts misses it entirely.
        inbound_deposits = [
            t
            for t in network
            if t["beneficiary_party_id"] == party_id
            and "deposit" in str(t.get("channel", "")).lower()
            and (a := amount_of(t)) is not None
            and a < Decimal("10000")
        ]
        depositors = {t["origin_party_id"] for t in inbound_deposits}
        if len(depositors) >= 3 and len(inbound_deposits) >= 4:
            total = sum(amount_of(t) or Decimal(0) for t in inbound_deposits)
            # The rapid onward withdrawal is the defining temporal signature.
            latest_deposit = max(t["value_date"] for t in inbound_deposits)
            # `>=` on the date alone missed withdrawals recorded on the same day as the final
            # deposit, and a same-day sweep is the strongest form of the signature rather than
            # a marginal case. Ordering by date keeps the earliest qualifying withdrawal.
            # The sweep is identified by size as well as timing: it moves most of what was
            # deposited. Selecting purely on date picked up whichever unrelated payment
            # happened to fall on the same day, so candidates are ranked by how closely they
            # match the deposited total and only a genuine sweep (>60%) qualifies.
            def sweep_score(txn: dict) -> Decimal:
                amount = amount_of(txn)
                return abs(amount - total) if amount is not None else Decimal("9" * 12)

            withdrawals = sorted(
                (
                    t
                    for t in network
                    if t["origin_party_id"] == party_id
                    and t["value_date"] >= latest_deposit
                    and "deposit" not in str(t.get("channel", "")).lower()
                    and (a := amount_of(t)) is not None
                    and a > total * Decimal("0.6")
                ),
                key=sweep_score,
            )
            candidates.append(
                {
                    "pattern": "funnel_account",
                    "transaction_ids": [t["transaction_id"] for t in inbound_deposits]
                    + [t["transaction_id"] for t in withdrawals[:1]],
                    "transaction_count": len(inbound_deposits) + min(1, len(withdrawals)),
                    "total_amount": str(total),
                    "counterparties": sorted(depositors),
                    "evidence": (
                        f"{len(inbound_deposits)} sub-threshold deposits from "
                        f"{len(depositors)} separate depositors totalling {total}"
                        + (", withdrawn within days" if withdrawals else "")
                    ),
                }
            )

        # -- trade-based: repeated payments against one invoice, to varied beneficiaries ---
        #
        # FATF/Egmont (2021): payments settled to third parties unrelated to the underlying
        # trade, with an invoice reference repeating across supposedly distinct shipments.
        by_reference: dict[str, list[dict]] = {}
        for txn in network:
            memo = str(txn.get("memo", ""))
            if txn["origin_party_id"] == party_id and "INV-" in memo:
                by_reference.setdefault(memo, []).append(txn)

        for reference, payments in by_reference.items():
            beneficiaries = {t["beneficiary_party_id"] for t in payments}
            # One invoice paid repeatedly, to more than one party, is the anomaly. A single
            # beneficiary is ordinary instalment payment.
            if len(payments) >= 3 and len(beneficiaries) >= 2:
                total = sum(amount_of(t) or Decimal(0) for t in payments)
                candidates.append(
                    {
                        "pattern": "trade_based",
                        "transaction_ids": [t["transaction_id"] for t in payments],
                        "transaction_count": len(payments),
                        "total_amount": str(total),
                        "counterparties": sorted(beneficiaries),
                        "evidence": (
                            f"{len(payments)} payments totalling {total} against a single "
                            f"reference ({reference}), settled to {len(beneficiaries)} "
                            f"different beneficiaries"
                        ),
                    }
                )

        # -- layering / round-tripping: chains where each hop feeds the next -------------
        by_origin: dict[str, list[dict]] = {}
        for txn in network:
            by_origin.setdefault(txn["origin_party_id"], []).append(txn)

        def follow(chain: list[dict], depth: int = 6) -> list[dict]:
            """Extend a chain forward, preferring the hop that best fits a laundering chain.

            An earlier version took the *first* qualifying hop and produced a two-hop chain of
            small benign payments while missing the real three-hop, $1.9M chain beside it.
            Laundering hops shed a modest commission and follow closely in time, so candidate
            hops are ranked by how well they fit that shape and the whole subtree is explored
            rather than the first match taken.
            """
            if depth == 0:
                return chain

            last = chain[-1]

            last_amount = amount_of(last)
            seen = {c["transaction_id"] for c in chain}

            viable: list[tuple[Decimal, dict]] = []
            for nxt in by_origin.get(last["beneficiary_party_id"], []):
                if nxt["transaction_id"] in seen or nxt["value_date"] < last["value_date"]:
                    continue
                # A hop that closes the circuit terminates the chain immediately. Deferring
                # this to the next recursion let a fifth transfer be appended to a
                # four-transfer round-trip, because the closure test ran before the extension
                # rather than at the moment of closing.
                if nxt["beneficiary_party_id"] == party_id:
                    return chain + [nxt]
                next_amount = amount_of(nxt)
                if last_amount is None or next_amount is None or last_amount == 0:
                    continue
                ratio = next_amount / last_amount
                # A commission, not a different transaction: 80-99% of the inbound amount.
                if not (Decimal("0.80") <= ratio <= Decimal("0.99")):
                    continue
                viable.append((ratio, nxt))

            if not viable:
                return chain

            # Highest retention first, the smallest commission is the most chain-like hop.
            best = chain
            for _, nxt in sorted(viable, key=lambda pair: pair[0], reverse=True)[:3]:
                extended = follow(chain + [nxt], depth - 1)
                if len(extended) > len(best):
                    best = extended
            return best

        for seed in by_origin.get(party_id, []):
            chain = follow([seed])
            if len(chain) < 2:
                continue
            parties = [c["origin_party_id"] for c in chain] + [chain[-1]["beneficiary_party_id"]]
            closes = chain[-1]["beneficiary_party_id"] == party_id
            origin_amount = amount_of(chain[0])
            candidates.append(
                {
                    "pattern": "round_tripping" if closes else "layering",
                    "transaction_ids": [c["transaction_id"] for c in chain],
                    "transaction_count": len(chain),
                    "total_amount": str(origin_amount) if origin_amount is not None else "unknown",
                    "counterparties": sorted(set(parties) - {party_id}),
                    "evidence": (
                        f"{len(chain)} sequential transfers through "
                        f"{len(set(parties)) - 1} intermediaries"
                        + (", returning to the subject" if closes else "")
                    ),
                }
            )

        # Rank by chain length first, then by value moved. Length alone let trivial
        # two-hop chains of small benign payments outrank the material one; value is what
        # makes a chain worth an investigator's attention.
        def rank(candidate: dict) -> tuple[int, Decimal]:
            try:
                value = Decimal(candidate["total_amount"])
            except (ArithmeticError, ValueError, TypeError):
                value = Decimal(0)
            return candidate["transaction_count"], value

        candidates.sort(key=rank, reverse=True)
        return candidates[:6]

    def counterparties_of(self, party_id: str) -> list[str]:
        """Distinct parties on the other side of this party's transactions."""
        others: list[str] = []
        for txn in self.transactions_for_party(party_id):
            other = (
                txn["beneficiary_party_id"]
                if txn["origin_party_id"] == party_id
                else txn["origin_party_id"]
            )
            if other not in others:
                others.append(other)
        return others

    def total_flow(self, party_id: str) -> Decimal:
        """Total value moved by a party.

        Returns zero when amounts are tokenized: at wider scopes `amount` is no longer a
        parseable number, and that failure is a *result*, it is precisely how protection
        scope destroys aggregation, so it is surfaced rather than worked around.
        """
        total = Decimal("0")
        for txn in self.transactions_for_party(party_id):
            try:
                total += Decimal(str(txn.get("amount", "0")))
            except (ArithmeticError, ValueError, TypeError):
                continue
        return total

    # -- the investigation --------------------------------------------------------

    def build_prompt(
        self, question: str, party_id: str | None = None, top_k: int = 5
    ) -> tuple[str, list[RetrievedChunk]]:
        """Assemble the payload. Everything in it is already protected."""
        chunks = self.index.search(question, top_k=top_k)

        sections = [f"QUESTION\n{question}"]
        if party_id:
            # Deterministic detection first. The model interprets candidates rather than
            # doing ledger arithmetic, which it does unreliably, and which is not what this
            # project measures.
            candidates = self.candidate_patterns(party_id)
            if candidates:
                sections.append(
                    "DETECTED CANDIDATE PATTERNS (computed by deterministic rules over the "
                    "protected ledger, figures here are exact)\n"
                    + json.dumps(candidates, indent=2)
                )

            # Two hops: enough to contain a layering chain or a round-trip circuit, which a
            # subject-only view cannot represent.
            transactions = self.transaction_network(party_id, hops=2, limit=120)
            sections.append(
                f"TRANSACTION NETWORK AROUND {party_id} "
                f"({len(transactions)} transactions, including counterparties' own activity)\n"
                + format_transactions(transactions, limit=120)
            )
        sections.append("RELEVANT CASE NOTES\n" + format_chunks(chunks))
        return "\n\n".join(sections), chunks

    def investigate(
        self,
        task_id: str,
        question: str,
        response_shape: str,
        party_id: str | None = None,
        top_k: int = 5,
        max_tokens: int = 900,
    ) -> InvestigationResult:
        """Run one investigation end to end, without unprotecting anything."""
        prompt, chunks = self.build_prompt(question, party_id, top_k)
        full_prompt = f"{prompt}\n\nRESPOND WITH JSON MATCHING THIS SHAPE\n{response_shape}"

        # Resolve the system prompt from the Langfuse registry (editable/versioned in the UI),
        # falling back to the code constant when the registry is unreachable or unseeded.
        from amlguard.domains import get_domain
        from amlguard.observability import managed_prompt

        prompt_name = (self.domain or get_domain()).investigation_prompt
        system = managed_prompt(prompt_name)
        try:
            completion = self.llm.complete(system, full_prompt, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001, a failed call is a scored outcome
            return InvestigationResult(
                task_id=task_id,
                question=question,
                prompt=full_prompt,
                raw_completion="",
                answer={},
                retrieved=chunks,
                error=f"llm: {exc}",
            )

        try:
            answer = extract_json(completion)
        except Exception as exc:  # noqa: BLE001
            return InvestigationResult(
                task_id=task_id,
                question=question,
                prompt=full_prompt,
                raw_completion=completion,
                answer={},
                retrieved=chunks,
                error=f"parse: {exc}",
            )

        return InvestigationResult(
            task_id=task_id,
            question=question,
            prompt=full_prompt,
            raw_completion=completion,
            answer=answer,
            retrieved=chunks,
        )

    def present(
        self,
        result: InvestigationResult,
        protector: Protector,
        role: Role = AUDITOR,
        guardrail: Any = None,
    ) -> InvestigationResult:
        """Re-identify the model's output for a viewer.

        Called **after** inference, never before. Tokens the role may not see stay protected.
        The default role is the least-privileged one: a call site that forgets to pass a
        role reveals nothing, instead of everything. Seeing plaintext requires naming the
        role that is entitled to it.

        Egress leak-check: `analyst_view` is a human-facing surface, so the same egress scan
        the review path runs belongs here too, or a model that wrote out an *untagged* clear
        identifier (one re-identification never touches, because it only reverses wrapped
        tokens) would reach a human unscanned. The scan runs on the raw completion **before**
        re-identification, so it flags what the model leaked, not the identifiers the role is
        entitled to see. Fails closed: a blocked completion is withheld, never shown. Opt-in
        by passing a guardrail (default None keeps the demo path dependency-free), matching
        review.py; when supplied it is load-bearing.
        """
        if guardrail is not None:
            verdict = guardrail.scan_response(result.raw_completion or "")
            if verdict.blocked:
                # The withheld notice must never contain what was withheld (same rule as
                # review.py): count and reason only, never echo the caught values.
                reason = (
                    f"{len(verdict.leaked_values)} forbidden value(s) detected"
                    if verdict.leaked_values else verdict.outcome
                )
                result.analyst_view = f"[withheld: response failed the egress check, {reason}]"
                result.reidentified = 0
                result.error = result.error or "egress-blocked"
                return result

        reidentified = reidentify(result.raw_completion, protector, role)
        result.analyst_view = reidentified.text
        result.reidentified = reidentified.revealed
        return result
