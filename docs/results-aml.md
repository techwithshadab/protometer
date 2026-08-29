# Results (aml): what data protection costs an AI pipeline

- **Domain:** aml
- **Use case:** batch measurement pipeline
- **Corpus fingerprint:** `36e2bee325e4`
- **Model(s) measured:** bedrock-sonnet-5
- **Generated:** 2026-08-28 (UTC) by `python scripts/generate_results.py --domain aml > docs/results-aml.md`

Every figure below is generated from the evaluation harness output; nothing here is
hand-entered. This document covers the **aml** domain only; other domains
have their own `docs/results-<domain>.md` and never overwrite this one.


## bedrock/us.anthropic.claude-sonnet-5

17 investigation tasks, 57 checkpoints per scope.

| Scope | Mean | Verifiable | Retained | Task completion | Aggregation | Identity | Typology | Narrative |
|---|---|---|---|---|---|---|---|---|
| `none` | 0.821 | 0.855 | 100% | 47% | 0.88 | 0.98 | 0.73 | 0.71 |
| `direct` | 0.821 | 0.832 | 100% | 53% | 0.88 | 0.98 | 0.67 | 0.79 |
| `direct-plus-context` | 0.821 | 0.832 | 100% | 53% | 0.88 | 0.98 | 0.67 | 0.79 |
| `direct-plus-temporal` | 0.803 | 0.808 | 98% | 47% | 0.88 | 0.90 | 0.67 | 0.79 |
| `direct-plus-monetary` | 0.803 | 0.808 | 98% | 47% | 0.88 | 0.90 | 0.67 | 0.79 |
| `quasi` | 0.785 | 0.785 | 96% | 47% | 0.88 | 0.90 | 0.60 | 0.79 |
| `all` | 0.750 | 0.785 | 91% | 41% | 0.88 | 0.90 | 0.60 | 0.64 |
| `direct-nondeterministic` | 0.838 | 0.855 | 102% | 53% | 0.88 | 0.98 | 0.73 | 0.79 |

Copy-the-detector baseline: 0.52-0.52 mean checkpoint score across scopes, against model means of 0.75-0.84: the model adds 23-32 points over transcribing detector output.

_Judged checkpoints (narrative stratum) default to the subject model as its
own judge; the Verifiable column excludes them entirely._


247 billed calls, $2.7133 total, per-scope median latency averaging 3s per call. (Billed figures are what the artifact set as-committed cost to produce, re-runs over a warm response cache bill only fresh calls; a from-scratch single-model measurement of all eight scopes bills every call.)

### Separation

- `none` and `direct` differ by 0 checkpoints and are **not distinguishable** at this task count.
- `direct` and `direct-plus-context` differ by 0 checkpoints and are **not distinguishable** at this task count.
- `direct-plus-context` and `direct-plus-temporal` differ by 1 checkpoints and are **not distinguishable** at this task count.
- `direct-plus-temporal` and `direct-plus-monetary` differ by 0 checkpoints and are **not distinguishable** at this task count.
- `direct-plus-monetary` and `quasi` differ by 1 checkpoints and are **not distinguishable** at this task count.
- `quasi` and `all` differ by 2 checkpoints and are **not distinguishable** at this task count.


## Classifier: what protection costs a trained model

Random forest over the protected ledger plus graph features, **temporally split** -
train on the earlier ledger, test on the later. Population aggregates and the graph
are fit on the training fold only.

| Scope | AP (±seed SD) | ROC-AUC | Retained | P@25 | P@50 | Lift@25 | ECE |
|---|---|---|---|---|---|---|---|
| `none` | 0.473 ±0.013 | 0.690 | 100% | 0.92 | 0.84 | 11.8x | 0.016 |
| `direct` | 0.473 ±0.013 | 0.690 | 100% | 0.92 | 0.84 | 11.8x | 0.016 |
| `direct-plus-context` | 0.473 ±0.013 | 0.690 | 100% | 0.92 | 0.84 | 11.8x | 0.016 |
| `direct-plus-temporal` | 0.498 ±0.010 | 0.802 | 105% | 0.92 | 0.84 | 11.8x | 0.027 |
| `direct-plus-monetary` | 0.451 ±0.012 | 0.675 | 95% | 0.84 | 0.84 | 10.8x | 0.029 |
| `quasi` | 0.470 ±0.011 | 0.796 | 99% | 0.84 | 0.84 | 10.8x | 0.033 |
| `all` | 0.470 ±0.011 | 0.796 | 99% | 0.84 | 0.84 | 10.8x | 0.033 |

The `±` is the SD of AP across RandomForest seeds (split, features and labels fixed; only `random_state` varies). The seed set **includes** the seed the reported AP is fit at, so the band is an interval around a member of its own population, not around an unsampled point. Read a scope delta smaller than this SD as seed noise, not protection cost.

Top features by SHAP on the clear ledger:

- `day_index` 0.096
- `o_gw_hit_rate` 0.089
- `o_gw_mean_length` 0.087
- `b_gw_mean_length` 0.067
- `b_gw_hit_rate` 0.065

Recall by typology at the operating threshold (clear ledger), with the test-fold
denominator. **Read these as anecdote, not rates:** trade_based is a handful of
transactions, so a single one moves its recall double digits. The operating
threshold is F1-selected on the **training** fold and applied unchanged to the
test fold, so these carry no threshold-overfitting bias (AP and ROC-AUC are
threshold-free regardless). Trade-based is nearly invisible because its defining
indicator, invoice value mismatch, cannot exist in a ledger-only corpus:

| Typology | Recall | n (caught / test txns) |
|---|---|---|
| round_tripping | 21% | 4/19 |
| funnel_account | 0% | 0/50 |
| layering | 0% | 0/16 |
| structuring | 0% | 0/73 |


## The protection-technique frontier

**The metadata tokenization deliberately leaves clear, scored by the vendor's own risk engine across all three attacker models:** k-anonymity 2, prosecutor risk 0.5 (worst-case), journalist 0.5, marketer 0.0067 (average-case bulk linkage), rated high across 4000 parties on jurisdiction, party_type, risk_rating, is_pep. The split matters: worst-case exposure is high but average-case bulk-linkage is low, an institution reads both. The open-metadata trade is now a number, not an assertion.

| AMOUNT treatment | Average precision | vs clear |
|---|---|---|
| clear | 0.473 | 100% |
| k-anonymity interval generalization (midpoints) | 0.485 | 103% |
| format-preserving tokenization | 0.451 | 95% |

Generalization keeps the magnitude signal tokenization destroys: the generalized AP is +0.012 vs clear (within the ~0.013 RF-seed SD), so read it as no measurable loss rather than 'better than clear', while tokenizing AMOUNT costs 0.023 AP on this corpus. On this corpus draw that AMOUNT cost is small and within seed noise; a different draw produced an amount-reliant model where it was ~10% (the single-seed sensitivity the utility curve is built to expose). The techniques answer different exposures: tokenization is reversible per row behind a role gate; generalization is an irreversible release format.

**Synthetic twin (vine copula):** 6827 rows, identity linkage none by construction: no real identifier is emitted. Fidelity: amount mean 29820.77 real vs 29939.6 synthetic, p95 144064.87 vs 144280.22, channel-mix L1 distance 0.0354. **Task utility (TSTR):** a classifier trained on the synthetic table and tested on real data retains **0.799** of the train-on-real score (predict channel from amount (macro-F1 on a held-out real test set)) (reference task near chance — TRTR 0.1055 vs chance 0.0909; read the retention ratio as DIRECTIONAL only, not precise utility), so the synthetic arm is scored by downstream utility like the other techniques, not fidelity moments alone.


## Adversarial evaluation

Adversary holds the protected corpus at `direct` plus an auxiliary graph, and not
the tokenization key.

| Attack | Success rate | Relabeled-graph control | Chance |
|---|---|---|---|
| format_leakage | 60.4% | n/a | 25.00% |
| neighbourhood_linkage | 52.0% | 0.04% | 0.04% |
| structural_linkage | 3.9% | 0.00% | 0.04% |
| frequency_analysis[PERSON] | 3.3% | n/a | 3.33% |
| frequency_analysis[ORGANIZATION] | 0.7% | n/a | 0.69% |

For the structural attacks the success rate is a **disclosure rate**: the share
of parties whose graph signature is unique against an exact auxiliary graph. The
honest null is the **relabeled-graph control** (identical structure, randomized
identities), not random guessing, the attack measures real linkage only insofar as
the control collapses toward chance (neighbourhood 52.0% -> 0.04%). Lift-over-chance is omitted here
because for a conclude-only-when-unique procedure it merely restates the raw count.


## Semantic Erasure

Both arms scored the same way: recall of a known-correct document in the top 10.

| Scope | Behavioural found | Identity found | Identity mean rank | Fisher p vs baseline |
|---|---|---|---|---|
| `none` | 4/5 | 26/40 | 3.92 | - |
| `direct` | 5/5 | 1/40 | 8.0 | 1.2e-09 |
| `direct-plus-context` | 4/5 | 2/40 | 8.5 | 1.3e-08 |
| `direct-plus-temporal` | 4/5 | 1/40 | 8.0 | 1.2e-09 |
| `direct-plus-monetary` | 4/5 | 2/40 | 7.0 | 1.3e-08 |
| `quasi` | 4/5 | 2/40 | 8.0 | 1.3e-08 |
| `all` | 4/5 | 2/40 | 9.0 | 1.3e-08 |


## Hybrid triage

Queue at **alert grain**, the unit an analyst dispositions. Ranked on a composite of
model evidence, days remaining on the 31 CFR 1020.320 filing clock, and repeat-alert
history.

| Scope | Queue | P@50 | Distinct subjects | Egress (blocked/discounted) | Ungrounded | Cost |
|---|---|---|---|---|---|---|
| `none` | 675 | 0.48 | 18/25 | 0/6 | 0/25 | $0.1707 |
| `quasi` | 675 | 0.48 | 19/25 | 0/7 | 0/25 | $0.1690 |

The queue is restricted to alerts raised in the scoring window (the temporal test
fold), and precision is scored against whether the *case* deserved escalation. The
structural ceiling is well below 1.0 because only the in-window alerts have evidence
the test-fold model can see, so a clear-scope P@50 of 0.48 is a strong multiple
over working the queue in random order at this base rate, not a weak absolute number.
'Ungrounded' counts rationales carrying at least one figure with no basis in their
evidence; each is marked on the decision.

**Filing-clock caveat:** the urgency term anchors `as_of` to the newest alert, but
the corpus's alert dates all precede the SAR deadline window, so every alert in the
head is past the 30-day clock and the urgency term saturates uniformly, on this
corpus the ordering is effectively score-driven. The clock becomes discriminative
only on a live feed where alert ages straddle the deadline; stated so the queue is
not read as demonstrating deadline triage.


## Queue composition (fairness check)

| Attribute | Population | Queued alerts | Reviewed head | Ranker lift (head vs queue) |
|---|---|---|---|---|
| jurisdiction US | 76.1% | 76.4% | 40.0% | 0.5x |
| jurisdiction LB | 4.0% | 3.6% | 16.0% | 4.5x |
| jurisdiction PA | 3.1% | 3.4% | 12.0% | 3.5x |
| jurisdiction VG | 3.8% | 3.3% | 12.0% | 3.7x |
| jurisdiction AE | 3.5% | 4.6% | 12.0% | 2.6x |
| jurisdiction CY | 2.9% | 2.8% | 8.0% | 2.8x |
| PEP | 3.3% | 1.8% | 0.0% | 0.0x |
| organization | 46.0% | 48.1% | 68.0% | 1.4x |

**Limits of this table, stated:** the head holds 25 subjects, so a
zero (e.g. PEP) is uninformative rather than reassuring, at a 3.3% base rate the
expectation in 25 draws is ~0.83, and zero observed cannot distinguish fairness
from chance. Offshore-jurisdiction lift over the population reflects that layering
chains genuinely route through such entities in the planted typologies; the
classifier takes no geographic, PEP or party-type input, and the ranker-lift
column is the one that would expose the ranker adding disparity. Reported so drift
is visible, not because current values alarm.


## Scope definitions

| Scope | Protects |
|---|---|
| `none` | Baseline. Clear text throughout. The reference every delta is measured against. |
| `direct` | Direct identifiers only, the realistic minimum a bank would deploy. |
| `direct-plus-context` | Direct plus locational and demographic quasi-identifiers. Amounts and dates stay clear. |
| `direct-plus-temporal` | Adds dates. Isolates whether temporal reasoning drives the utility cliff. |
| `direct-plus-monetary` | Adds amounts but not dates. Isolates whether arithmetic drives the cliff. |
| `quasi` | Direct plus quasi-identifiers. Measured location of the utility knee. |
| `all` | Everything the discovery service detects. Maximum protection. |
| `direct-nondeterministic` | Ablation: direct scope with external_iv, breaking cross-document token stability. |
