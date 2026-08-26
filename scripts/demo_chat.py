"""Multi-turn protected chatbot demo for ANY domain (LIVE, paid).

Runs several turns through ONE ConversationSession so the demo shows the per-turn protection
boundary (tokenize -> reason over tokens -> egress scan -> role-gated reply) AND the cross-turn
properties: conversation history and token stability (the same subject in turn 1 and turn 5 gets
the SAME token). One Protegrity login and one LLM client for the whole run (matches the UI serving
path), so N turns never open N logins.

    python scripts/demo_chat.py --domain healthcare        # 8 turns, ~$0.12
    python scripts/demo_chat.py --domain customer-support --turns 6
    python scripts/demo_chat.py --domain aml

Each turn is a real Claude Sonnet 5 call counted against the spend cap. The subjects named in the
prompts are corpus parties (the only populated roster) so tokenization actually fires.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from amlguard.domains import domain_names, get_domain  # noqa: E402
from amlguard.guardrail import Guardrail  # noqa: E402
from amlguard.llm import get_llm  # noqa: E402
from amlguard.protect import Protector  # noqa: E402
from amlguard.reidentify import INVESTIGATOR  # noqa: E402
from amlguard.roster import roster_from_parties  # noqa: E402
from amlguard.serving import ConversationSession  # noqa: E402

# PII-SHAPED patterns to redact from a stored reply. The reply is written over TOKENS (no real
# corpus value survives — the egress leak check guarantees that), but a MODEL can still emit a
# token that happens to LOOK like a raw SSN/card/phone/email, and a demo artifact for a
# data-protection product must not display those raw. Replace the SHAPE with a typed placeholder so
# the committed transcript is visibly clean while the protection story stays intact.
_PII_SHAPES = [
    (re.compile(r"\b\d{3}-\d{2,3}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[CARD]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE]"),
]


def _redact_pii_shapes(text: str | None) -> str:
    """Replace any raw SSN/card/email/phone-SHAPED substring with a typed placeholder. Applied to a
    reply BEFORE it is persisted, so a committed transcript never displays a raw PII shape even
    though every such string is a protection token (not a real value)."""
    out = text or ""
    for pat, repl in _PII_SHAPES:
        out = pat.sub(repl, out)
    return out

# Subjects are REAL corpus parties WITH narrative records, so (a) the roster tokenizes their names
# and (b) we can feed the model their protected (tokenized) case notes as context. Without that
# context the model has nothing to reason over and correctly answers "I have no records"; WITH it,
# the model produces substantive answers ENTIRELY over tokens — the actual thesis of the demo.
_SUBJECT_IDS = ["P02463", "P03247", "P00703"]  # Sana Choudhury, Bruno Reyes, Marcus Santoro

# Per-domain system prompt + turn set. Turns reference the subjects by name; the roster tokenizes
# the names on the way in, and the protected case notes (context) give the model real material.
DOMAIN_TURNS: dict[str, dict] = {
    "healthcare": {
        "system": "You are a clinical-record assistant. You are given case notes written entirely "
                  "over stable pseudonymous tokens. Reason over the provided notes only; cite what "
                  "they say; never invent record contents beyond them.",
        "turns": [
            "Summarize the case notes on file for Sana Choudhury.",
            "What concerns are documented for Bruno Reyes?",
            "Based on the notes, what is the risk profile for Sana Choudhury?",
            "Draft a one-line handoff note for the case involving Marcus Santoro.",
            "Do the notes show any shared context between Bruno Reyes and Sana Choudhury?",
            "List the open items that still need review for Marcus Santoro.",
            "Give a neutral status update on Sana Choudhury for the supervisor.",
            "What follow-up do the notes suggest for Bruno Reyes?",
        ],
    },
    "customer-support": {
        "system": "You are a customer-support assistant. You are given case notes written entirely "
                  "over stable pseudonymous tokens. Use the provided notes to help; never reveal "
                  "or invent account details beyond them.",
        "turns": [
            "Summarize the case history on file for Sana Choudhury.",
            "What is documented about Bruno Reyes's recent activity?",
            "Based on the notes, what should we follow up on for Sana Choudhury?",
            "Draft a polite reply to Marcus Santoro acknowledging their case.",
            "Do the notes mention Bruno Reyes contacting us before?",
            "List the open items for Marcus Santoro from the notes.",
            "Write a one-line escalation note for Sana Choudhury's case.",
            "What next step do the notes suggest for Bruno Reyes?",
        ],
    },
    "aml": {
        "system": "You are an AML investigation assistant. You are given case notes written "
                  "entirely over stable pseudonymous tokens. Reason over the provided notes only; "
                  "cite what they document; never invent evidence beyond them.",
        "turns": [
            "Summarize the alert notes on file for Sana Choudhury.",
            "What structuring or layering concerns are documented for Bruno Reyes?",
            "Based on the notes, what is the risk picture for Sana Choudhury?",
            "Draft a suspicious-activity-report note for the case involving Marcus Santoro.",
            "Do the notes show shared counterparties between Bruno Reyes and Sana Choudhury?",
            "List the open alerts documented for Marcus Santoro.",
            "Give a status update on Sana Choudhury for the supervisor.",
            "What investigative follow-up do the notes suggest for Bruno Reyes?",
        ],
    },
}


# Domain-appropriate CLEAR case-note templates per subject. Named fields ({name}, {ssn}, {mrn}…)
# are tokenized inline via the roster/protector before use, so the model only ever sees tokens.
# AML uses the real corpus narratives (below, via _aml_context); healthcare/support use these.
_HC_NOTES = {
    "P02463": "Clinical note — Patient {name}, MRN {mrn}: type-2 diabetes and hypertension; "
              "A1c trending down on metformin; last visit {date}; follow-up in 3 months.",
    "P03247": "Clinical note — Patient {name}, MRN {mrn}: post-op review after knee arthroplasty; "
              "wound healing well; physiotherapy ongoing; no signs of infection.",
    "P00703": "Clinical note — Patient {name}, MRN {mrn}: asthma review; inhaler technique "
              "reinforced; peak flow stable; annual review scheduled.",
}
_CS_NOTES = {
    "P02463": "Support ticket — Customer {name} ({email}): reported a login loop after a password "
              "reset; account {acct}; issue reproduced; escalated to platform team.",
    "P03247": "Support ticket — Customer {name} ({email}): billing question on order {acct}; "
              "duplicate charge suspected; refund under review.",
    "P00703": "Support ticket — Customer {name} ({email}): delivery delayed on order {acct}; "
              "carrier trace opened; customer offered a credit.",
}


def _tokenize(protector, value: str, element: str) -> str:
    """Protect one value and wrap it as a tagged span, so the context is written over tokens the way
    the ingest path emits them."""
    try:
        tok = protector.protect_value(value, element)
    except Exception:  # noqa: BLE001, if protection fails, fall back to a masked placeholder
        return "[REDACTED]"
    tag = {"string": "PERSON", "ccn": "CREDIT_CARD"}.get(element, "ACCOUNT_NUMBER")
    return f"[{tag}]{tok}[/{tag}]"


def _domain_context(domain: str, subject_ids, id_to_name: dict, protector=None) -> str:
    """A compact, tokenized dossier for the demo subjects, appropriate to the domain."""
    if domain == "aml":
        return _aml_context(subject_ids, id_to_name)
    templates = _HC_NOTES if domain == "healthcare" else _CS_NOTES
    from amlguard.protect import Protector
    p = protector or Protector()
    blocks = []
    for i, sid in enumerate(subject_ids):
        name = id_to_name.get(sid, sid)
        tmpl = templates.get(sid)
        if not tmpl:
            continue
        note = tmpl.format(
            name=_tokenize(p, name, "string"),
            mrn=f"[MED_REC]MRx{100 + i}[/MED_REC]",
            date="[DATE]2025[/DATE]",
            email=_tokenize(p, f"{name.split()[0].lower()}@example.com", "string"),
            acct=_tokenize(p, f"ORD-{90000 + i}", "number"),
        )
        blocks.append(f"Record for {name} (tokenized):\n- {note}")
    return "\n\n".join(blocks)


def _aml_context(subject_ids, id_to_name: dict) -> str:
    """AML context: the REAL protected corpus narratives for the subjects (structuring case notes)."""
    protected_narr = json.loads(
        (ROOT / "data" / "protected" / "direct" / "narratives.json").read_text())
    notes_by_subject: dict[str, list[str]] = {}
    for rec in protected_narr:
        sid = rec.get("subject_party_id")
        if sid in subject_ids:
            notes_by_subject.setdefault(sid, []).append(rec.get("text", ""))
    blocks = []
    for sid in subject_ids:
        name = id_to_name.get(sid, sid)
        notes = notes_by_subject.get(sid, [])[:3]
        if notes:
            joined = "\n".join(f"- {t}" for t in notes)
            blocks.append(f"Case notes for {name} (tokenized):\n{joined}")
    return "\n\n".join(blocks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="healthcare", choices=domain_names())
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--model", default="bedrock-sonnet-5")
    ap.add_argument("--json", action="store_true",
                    help="save the full transcript (per-turn pipeline internals) to "
                         "data/eval/<domain>/chat_replay.json for the UI's Replay mode")
    args = ap.parse_args()

    domain = get_domain(args.domain)
    spec = DOMAIN_TURNS.get(args.domain, DOMAIN_TURNS["aml"])
    turns = spec["turns"]
    n = max(1, min(args.turns, len(turns)))

    print(f"\n=== {domain.label} chatbot: {n} live turns (domain: {domain.name}) ===")
    print(f"    guardrail model: {domain.injection_processor} · LLM: {args.model}\n")

    # Singletons, built once (one Protegrity login for the whole run).
    protector = Protector()
    llm = get_llm(args.model, trace_component=f"serving-{domain.name}",
                  enable_cache=False, allow_fallback=False)
    parties_path = ROOT / "data" / "corpus" / "parties.json"
    try:
        guardrail = Guardrail.for_corpus(parties_path, probe=False, domain=domain)
    except Exception as exc:  # noqa: BLE001
        guardrail = None
        print(f"    (guardrail unavailable: {type(exc).__name__}; replies shown UNSCANNED)\n")
    parties = json.loads(parties_path.read_text())
    roster = roster_from_parties(parties)

    # Build the DOMAIN-APPROPRIATE tokenized context for the demo subjects, so the assistant reasons
    # over material that fits its domain (clinical notes for healthcare, support tickets for support,
    # AML case notes for financial crime) — never AML content leaking into a clinical chat. The
    # context is already tokenized (the model reasons over tokens, never plaintext), and the same
    # subject tokens recur across turns (the token-stability the demo highlights).
    id_to_name = {p["party_id"]: p["full_name"] for p in parties}
    tokenized_context = _domain_context(args.domain, _SUBJECT_IDS, id_to_name, protector=protector)

    session = ConversationSession(
        protector=protector, llm=llm, conversation_id=f"{domain.name}-demo",
        role=INVESTIGATOR, domain=domain, system_prompt=spec["system"],
        guardrail=guardrail, roster=roster, require_guardrail=bool(guardrail),
    )

    summary = {"ok": 0, "blocked": 0, "error": 0, "pii_protected": 0}
    transcript = []  # per-turn record for the UI Replay mode (same shape as /api/chat/turn)
    for i, msg in enumerate(turns[:n], 1):
        # Pass the tokenized dossier as context so the model has protected material to reason over.
        result = session.turn(msg, context=tokenized_context)
        summary["pii_protected"] += result.entities_protected
        if result.ok:
            summary["ok"] += 1
        elif result.egress_blocked:
            summary["blocked"] += 1
        else:
            summary["error"] += 1
        eg = result.egress_detail or {}
        discounted = not result.egress_blocked and eg.get("outcome") == "rejected"
        # IMPORTANT: the transcript is persisted to disk, so it must NEVER contain cleartext PII.
        # We store `raw_completion` (the model's reply written over TOKENS, before re-identification)
        # as the displayed reply, plus the `revealed` count — so the replay shows the protected
        # boundary honestly (tokens in → tokens reasoned over → N identifiers re-identifiable for the
        # entitled role) without ever writing a real SSN/email/name to a committed artifact. The
        # role-gated cleartext reply exists only live, at the presentation boundary.
        transcript.append({
            "turn": i,
            "user": msg,
            "reply_over_tokens": _redact_pii_shapes(result.raw_completion),  # tokenized + shape-redacted
            "would_reidentify": result.revealed,          # count only; no cleartext persisted
            "internals": {
                "protected_input": result.protected_input,
                "model_saw": result.protected_input,
                "entities_protected": result.entities_protected,
                "revealed": result.revealed,
                "egress_blocked": result.egress_blocked,
                "egress_detail": result.egress_detail,
                "guardrail_model": domain.injection_processor,
                "role": "investigator",
                "domain": domain.name,
            },
            "ok": result.ok,
            "error": result.error,
        })
        print(f"--- Turn {i}/{n} " + "-" * 50)
        print(f"  user      : {msg}")
        print(f"  tokenized : {result.protected_input}")
        print(f"  model saw : tokens only ({result.entities_protected} PII protected)")
        print(f"  egress    : {eg.get('outcome', 'n/a')} "
              f"(score {eg.get('score', 'n/a')}, discounted={discounted})")
        reply = (result.reply or "").replace("\n", " ")
        print(f"  reply     : {reply[:220]}{'...' if len(reply) > 220 else ''}  "
              f"(revealed={result.revealed}, ok={result.ok})")
        print()

    print("=" * 66)
    print(f"  {n} turns · ok={summary['ok']} · egress-blocked={summary['blocked']} · "
          f"error={summary['error']} · total PII protected={summary['pii_protected']}")
    print(f"  history: {len(session.history)} turns retained (tokenized)\n")

    if args.json:
        from amlguard.persist import atomic_write_json
        record = {
            "domain": domain.name,
            "domain_label": domain.label,
            "role": "investigator",
            "model": args.model,
            "guardrail_model": domain.injection_processor,
            "system_prompt": spec["system"],
            "n_turns": n,
            "summary": summary,
            "turns": transcript,
        }
        dest = ROOT / "data" / "eval" / domain.name / "chat_replay.json"
        atomic_write_json(dest, record)
        print(f"transcript written to {dest.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
