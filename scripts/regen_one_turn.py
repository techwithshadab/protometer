"""Regenerate ONE turn of a committed chat_replay transcript, in place.

Motivation: the AML turn "Draft a SAR-supporting note …" made the Data Discovery classifier
tag the bare token "SAR" as CURRENCY_CODE (ISO 4217 Saudi Riyal) and tokenize it, so the UI
showed a confusing `[CURRENCY_CODE]…[/CURRENCY_CODE]` span for an AML term of art. The turn's
prompt in demo_chat.py was reworded to "suspicious-activity-report note", which no longer trips
the false positive (verified with a free discover_entities call). This script re-runs JUST that
one turn against the reworded prompt and splices the fresh record into the committed transcript,
preserving the other turns' (paid) replies. It bills exactly one hosted LLM turn.

The record shape written here MUST match scripts/demo_chat.py's transcript entries exactly.

Usage:
  python scripts/regen_one_turn.py --domain aml --index 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from protometer.domains import get_domain  # noqa: E402
from protometer.guardrail import Guardrail  # noqa: E402
from protometer.llm import get_llm  # noqa: E402
from protometer.persist import atomic_write_json  # noqa: E402
from protometer.protect import Protector  # noqa: E402
from protometer.reidentify import INVESTIGATOR  # noqa: E402
from protometer.roster import roster_from_parties  # noqa: E402
from protometer.serving import ConversationSession  # noqa: E402

# Reuse the exact turn specs + context builder that produced the committed transcript, so the
# regenerated turn is generated the same way as its siblings.
from demo_chat import (  # noqa: E402
    _SUBJECT_IDS,
    DOMAIN_TURNS,
    _domain_context,
    _redact_pii_shapes,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", default="aml")
    ap.add_argument("--index", type=int, required=True, help="0-based turn index to regenerate")
    ap.add_argument("--model", default="bedrock-sonnet-5")
    args = ap.parse_args()

    domain = get_domain(args.domain)
    spec = DOMAIN_TURNS[args.domain]
    new_msg = spec["turns"][args.index]

    dest = ROOT / "data" / "eval" / args.domain / "chat_replay.json"
    transcript = json.loads(dest.read_text())
    old = transcript["turns"][args.index]
    print(f"Regenerating {args.domain} turn index {args.index}")
    print(f"  old user: {old['user']!r}")
    print(f"  new user: {new_msg!r}")
    if old["user"] == new_msg:
        print("  (prompt unchanged — nothing to do)")
        return 0

    # Same singletons and context as demo_chat.py. A fresh single-turn session is faithful here
    # because this turn's subject is not referenced by prior turns (checked before running).
    protector = Protector()
    llm = get_llm(args.model, trace_component=f"serving-{domain.name}",
                  enable_cache=False, allow_fallback=False)
    parties_path = ROOT / "data" / "corpus" / "parties.json"
    guardrail = Guardrail.for_corpus(parties_path, probe=False, domain=domain)
    parties = json.loads(parties_path.read_text())
    roster = roster_from_parties(parties)
    id_to_name = {p["party_id"]: p["full_name"] for p in parties}
    tokenized_context = _domain_context(args.domain, _SUBJECT_IDS, id_to_name, protector=protector)

    session = ConversationSession(
        protector=protector, llm=llm, conversation_id=f"{domain.name}-demo",
        role=INVESTIGATOR, domain=domain, system_prompt=spec["system"],
        guardrail=guardrail, roster=roster, require_guardrail=True,
    )
    result = session.turn(new_msg, context=tokenized_context)

    new_record = {
        "turn": old["turn"],
        "user": new_msg,
        "reply_over_tokens": _redact_pii_shapes(result.raw_completion),
        "would_reidentify": result.revealed,
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
    }

    print(f"  tokenized : {result.protected_input}")
    print(f"  egress    : blocked={result.egress_blocked} revealed={result.revealed} ok={result.ok}")

    # Guard: the whole point is to remove the CURRENCY_CODE false positive.
    if "CURRENCY_CODE" in result.protected_input:
        print("  WARNING: CURRENCY_CODE still present in the tokenized input; NOT writing.")
        return 1

    # Splice only this turn; every other turn's paid reply is preserved verbatim.
    transcript["turns"][args.index] = new_record
    atomic_write_json(dest, transcript)
    print(f"  spliced into {dest.relative_to(ROOT)} (other turns unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
