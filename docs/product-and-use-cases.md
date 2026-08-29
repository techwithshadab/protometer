# AMLGuard / Aegis — Product & Use Cases

*A non-technical overview: what this is, who it's for, and what it does in each domain. For how it's
built, see [architecture.md](architecture.md). To run it, see [SETUP.md](SETUP.md).*

---

## The problem

Enterprises want to point AI at their most valuable data — transactions, patient records, customer
cases — but that data is exactly what regulation says must be protected. The usual answer is to keep
the AI away from it, which blocks the project, or to expose it and accept the risk. Neither is a
real answer, and nobody quantifies the middle ground: *if we protect the data, does the AI still
work, and what can still leak?*

**AMLGuard answers both — with a working product and measurements, not assertions.** It runs a real
investigation copilot entirely over **protected (tokenized) data**, and it ships an *instrument* that
measures what the protection costs the AI at every stage, including the residual re-identification
risk that honest vendors leave out.

The product is presented in the UI as **Aegis** ("Protected-Pipeline Intelligence"); the codebase and
engineering log use the name **AMLGuard**. Same system, two names.

## Who it's for

- **Financial-crime / AML teams** who want an alert-triage copilot but cannot hand customer identities
  to a model.
- **Healthcare data owners** who need to release patient data for AI under HIPAA and must show, not
  assert, that de-identification holds.
- **Customer-support orgs** that need least-privilege access to customer data at the point of contact.
- **Risk, compliance, and security reviewers** who need the protection/utility trade-off stated as a
  number they can govern, including where it is *not* safe.

## What it does, in one line per role

- **The analyst** gets a copilot that ranks the alert queue and drafts case notes — over data where
  every identity is a token, re-identified only for the role entitled to see it.
- **The compliance officer** gets a measured residual-risk report (before/after de-identification),
  including the cases where the bar is not met.
- **The security reviewer** gets an end-to-end protected pipeline with three observability planes and
  a leak-check on every model response.

## The three use cases

Each is the measured answer to a customer's real question. The numbers below are regenerated from
committed evaluation artifacts (they never go stale); the AML curve was measured with a hosted
frontier model, and the whole demo also **runs on a free local open-source model** so anyone can try
it (see below).

### 1. Financial crime (AML) — *"can the AI still work on protected data, and what still leaks?"*

- **A triage copilot works on protected data.** Clear and protected alert queues rank
  near-identically (precision@50 0.48 vs 0.48), so tokenizing identities does not break the deployable
  product.
- **Retrieval survives or dies by what the query asks for.** Identity-document search collapses under
  protection (recall 26/40 → 1/40) while behavioural search holds (4/5 → 5/5). Practical guidance:
  protect identities freely; identity-lookup RAG will not work over protected text, but behavioural
  RAG will.
- **Residual risk is quantified, not hidden.** About half of parties are re-identifiable from
  transaction topology alone (52% vs a 0.04% control) — a limit an institution can plan around rather
  than discover later.
- **What protection costs a trained model is instrumented.** Identity-only protection retains ~100% of
  the model's accuracy; the only measurable cost is small and honestly reported.

### 2. Healthcare — *"can we release patient data for AI without violating HIPAA?"*

- **Two HIPAA de-identification standards, measured.** *Safe Harbor* removes/tokenizes the direct
  identifiers; *Expert Determination* quantifies the residual re-identification risk before and after
  k-anonymization (average-case risk drops 0.964 → 0.47 after k=5).
- **Reported honestly.** Worst-case risk stays high and k-anonymity was not reached at this
  suppression, so Expert Determination is **not certified** on this sample — exactly the finding a
  compliance team needs stated plainly, not glossed.

### 3. Customer support — *"the agent shouldn't see the full card; the supervisor might."*

- **Role dual-gate.** The same protected reply is re-identified two ways: a front-line agent sees the
  customer masked, a supervisor may reveal in full — one tokenized message, gated by role at the point
  of presentation.

## What makes it different

- **It measures, it doesn't assert.** Every claim is a number regenerated from a committed run.
- **It reports the residual risk.** Most demos stop at "we protected the data." This one quantifies
  what an attacker could still recover, and says where the bar is not met.
- **One protection boundary, three domains.** The same seam runs AML, healthcare, and support.
- **It runs anywhere.** The live assistant uses a hosted frontier model when cloud credentials are
  present, and otherwise falls back to a **free local open-source model** — so a reviewer with no
  cloud account can still run the complete protected pipeline end to end. Tokenization always stays on
  Protegrity; only the reasoning model changes.

## Scope, assumptions & caveats

We state these up front, because a measurement is only trustworthy if its boundaries are on the table.
Everything below is a deliberate scope choice or a measured limit, not a gap we found late.

**What this is.** A working, evaluated *reference implementation* and a measurement instrument. It
demonstrates the capability and quantifies the trade-off end to end. It is not a production
deployment, and the numbers should be read as **comparative between protection levels**, not as
absolute production performance.

**The data.** The corpus is **synthetic** (no real person is depicted) and, deliberately, easier than
real AML: the illicit population is denser than production, and the workflow is simplified. We
*hardened* it across review rounds — removing detection giveaways our own review found — and each
removal lowered our scores and raised the result's credibility. Read absolute scores as
scope-to-scope comparisons.

**The headline numbers are single-corpus and single-seed.** The committed results cover one corpus and
one hosted model. Where a result is seed-sensitive we say so explicitly — for example, the cost of
protecting the transaction *amount* is small on this corpus but single-seed, and a different draw
produced a larger cost. We report the sensitivity rather than the flattering number; a multi-seed
sweep is on the roadmap.

**Where the bar is not met, we say so.** In healthcare, k-anonymization lowers the average-case
re-identification risk but does **not** reach the threshold on this sample, so we mark Expert
Determination **"not certified"** rather than implying it passed. This is the finding a compliance team
needs, and we surface it plainly.

**What it defends, and what it does not.** Tokenize-then-reason protects identities in use: the model,
the vector store, the logs, and the traces never hold a real identifier. It does **not** claim
differential privacy, and it does **not** hide *structure* — roughly half of parties remain
re-identifiable from transaction topology alone. That residual is measured and reported, because an
institution can only plan around a risk it can see.

**Live vs. measured model.** The evaluation curve was produced by a hosted frontier model. The live
demo can run that same hosted model, or fall back to a free local open-source model so anyone can run
it — but the local model is for *running the demo*, not for reproducing the headline numbers.

**Roles, in this edition.** Re-identification is role-gated and enforced in the application; the same
gate can be centrally managed in Protegrity policy in a Team-Edition deployment. Live chat ships for
all three domains, each protected against its own party corpus; a domain whose corpus is not
loaded fails closed with a precise message rather than protecting against the wrong entity set.

**Not fine-tuning.** The training stage is a supervised classifier over the protected ledger, which is
what the protected-pipeline claim needs. It is not LLM fine-tuning, and we do not present it as such.

## From blocked project to governed deployment

The business value is the shape of the transition: a data-rich AI project that compliance would
otherwise block becomes a governed deployment, because the protection is provable and its cost is
measured. The analyst keeps the copilot; the institution keeps the audit trail and the risk numbers.

---

*Built on Protegrity Developer Edition for the Protegrity 2026 AI Pipeline Security Hackathon.*
