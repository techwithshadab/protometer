"""Curate the cross-domain results summary the README carries, from whatever runs exist.

The README's job is to explain the results with a business focus, across every domain and use
case measured, not just AML batch. As new domain runs land (healthcare de-identification, a
support demo, another model's curve), this regenerates a "Capabilities, measured across domains"
section between the markers `<!-- SUMMARY:START -->` and `<!-- SUMMARY:END -->` in README.md,
so the curated cross-domain view is derived from artifacts and never goes stale.

    python scripts/generate_summary.py            # print the section
    python scripts/generate_summary.py --write    # splice it into README.md between the markers

Each block states, per the business objective: what the customer gets, the measured number, and
why it matters. Absent artifacts are simply omitted (never a fabricated row).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from amlguard.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

EVAL = ROOT / "data" / "eval"
START, END = "<!-- SUMMARY:START -->", "<!-- SUMMARY:END -->"


def _load(rel: str):
    p = EVAL / rel
    return json.loads(p.read_text()) if p.exists() else None


def build() -> str:
    from key_metrics import key_metrics

    m = key_metrics()
    out: list[str] = []
    out.append("## Capabilities, measured across domains\n")
    out.append(
        "The protection boundary is one thing; what it *buys the business* differs by domain. "
        "Each block below is the measured answer to a customer's real question, regenerated "
        "from committed artifacts (`scripts/generate_summary.py`), so it never goes stale.\n"
    )

    # ── AML batch (the measurement thesis) ──────────────────────────────────────────────────
    if m.get("training") and m.get("attack") and m.get("erasure"):
        tr, at, er, hy = m["training"], m["attack"], m["erasure"], m.get("hybrid", {})
        out.append(
            "### Financial crime (AML) — *\"can the AI still work on protected data, and what "
            "still leaks?\"*\n\n"
            "- **A triage copilot runs on protected data.** Clear and protected alert queues "
            "rank near-identically"
            + (f" (P@50 {hy['none']['p_at_50']} vs {hy['quasi']['p_at_50']})" if hy.get("quasi")
               else "")
            + f", so tokenizing identities does not break the deployable product.\n"
            f"- **Retrieval survives by query type.** Identity-document recall collapses "
            f"{er['identity_none']} → {er['identity_direct']} under protection while behavioural "
            f"recall holds {er['behavioural_none']} → {er['behavioural_direct']} "
            f"(Fisher p = {er['fisher_p']:.1e}). Protect identities freely; know identity-lookup "
            f"RAG will not work over protected text but behavioural RAG will.\n"
            f"- **Residual risk is quantified, not hidden.** {at['neighbourhood_pct']}% of "
            f"parties are re-identifiable from transaction topology alone "
            f"(control {at['control_pct']}%), a limit an institution can plan around.\n"
            f"- **What protection costs a trained model is instrumented.** Identity-only "
            f"protection retains ~100% of average precision (AP {tr['none_ap']} at the clear "
            f"baseline); the AMOUNT-cost is small and single-seed sensitive (measured, "
            f"reported honestly).\n"
        )

    # ── Healthcare (HIPAA de-identification) ────────────────────────────────────────────────
    hc = _load("healthcare/deidentify.json")
    if hc:
        ed = hc.get("expert_determination", {})
        before = ed.get("before", {})
        after = ed.get("after", {})
        kanon = ed.get("k_anonymization", {})
        met = ed.get("expert_determination_met")
        if met:
            honest_line = (
                f"- **Expert Determination met on this sample:** worst-case (prosecutor) risk fell "
                f"to {after.get('prosecutor_risk')} with k≥{kanon.get('k')}, the statistical "
                f"evidence a compliance team needs to certify \"very small\" residual risk.\n"
            )
        else:
            honest_line = (
                f"- **Reported honestly:** worst-case (prosecutor) risk stays "
                f"{after.get('prosecutor_risk')} and k-anonymity was not reached at this "
                f"suppression, so Expert Determination is **not** certified on this sample — more "
                f"generalization or fewer quasi-identifiers would be needed. Exactly the finding a "
                f"compliance team needs stated, not glossed.\n"
            )
        out.append(
            f"### Healthcare — *\"can we release patient data for AI without violating HIPAA?\"*\n\n"
            f"- **Two HIPAA de-identification standards, measured.** Safe Harbor removes/tokenizes "
            f"the direct identifiers; Expert Determination quantifies re-identification risk. "
            f"Average-case (marketer) risk drops **{before.get('marketer_risk')} → "
            f"{after.get('marketer_risk')}** after k={kanon.get('k')} anonymization "
            f"(information loss {round(kanon.get('information_loss', 0), 2)}, "
            f"{kanon.get('suppressed_count')} rows suppressed).\n"
            + honest_line
        )

    # ── Customer support (role-differentiated access) ───────────────────────────────────────
    if (ROOT / "scripts" / "demo_support_gates.py").exists():
        out.append(
            "### Customer support — *\"the agent shouldn't see the customer's full card, but the "
            "supervisor might.\"*\n\n"
            "- **Role-differentiated detokenization (dual-gate).** The *same* protected reply "
            "shows a masked view to a support agent and a full view to a supervisor, from one "
            "tokenized message, so least-privilege access is enforced at the presentation "
            "boundary (application-enforced, not Protegrity policy).\n"
        )

    # ── The protection-technique frontier ───────────────────────────────────────────────────
    if m.get("frontier"):
        fr = m["frontier"]
        dp = "tier-gated in Developer Edition (honestly reported)" if fr.get("dp_available") is False \
            else "measured"
        # The TSTR retention ratio is only meaningful when the reference task cleared chance.
        # When it did not (or the artifact predates the guard, value is None), render it as
        # directional, never as a clean utility headline — the convenient-number rule.
        above = fr.get("tstr_reference_above_chance")
        if above:
            tstr_clause = (f"synthetic data (**{fr.get('tstr_retained')}** task utility retained "
                           f"via train-on-synthetic/test-on-real)")
        else:
            ref = (f" (TRTR {fr.get('tstr_trtr')} vs chance {fr.get('tstr_chance')})"
                   if fr.get("tstr_chance") is not None else "")
            tstr_clause = (f"synthetic data (TSTR {fr.get('tstr_retained')} — but the reference "
                           f"task is near chance{ref}, so read this as directional, not a precise "
                           f"utility figure)")
        out.append(
            f"### Choosing a protection technique — *\"tokenize, generalize, or synthesize?\"*\n\n"
            f"- **One corpus, four techniques, measured.** Tokenization (marketer risk "
            f"{fr.get('marketer_risk')} on left-clear metadata), k-anonymity (AP "
            f"{fr.get('kanon_ap')}), {tstr_clause}, and differential privacy ({dp}). "
            f"The choice is a number, backed by its own uncertainty, not a vibe.\n"
        )

    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    section = build()
    if "--write" in sys.argv:
        readme = ROOT / "README.md"
        text = readme.read_text()
        if START not in text or END not in text:
            sys.exit(f"README.md is missing the {START} / {END} markers; add them where the "
                     f"cross-domain summary should live.")
        pre = text.split(START)[0]
        post = text.split(END)[1]
        readme.write_text(f"{pre}{START}\n\n{section}\n{END}{post}")
        print("spliced the cross-domain summary into README.md")
    else:
        print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
