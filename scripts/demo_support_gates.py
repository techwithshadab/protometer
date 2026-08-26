"""Customer-support dual-gate demo: role-differentiated detokenization (Gate 2).

Adapts the reference dual-gate pattern (vendor Orchestrators-BankingPortalChatbot,
`common/protegrity_gates.py`) to our serving primitives:

  Gate 1 (input):  classify PII -> tokenize   (protect the inbound customer message)
  Gate 2 (output): find tagged tokens -> detokenize, PER ROLE

The point of this demo is Gate 2: the *same* protected reply, re-identified under two different
support roles, shows different data. A front-line **Support Agent** resolves the case with the
customer's name/email/card/account masked; a **Supervisor** may fully re-identify for identity
verification. Same application-enforced mechanism as the AML roles, retargeted for support.

Runs entirely on fakes, no LLM, no hosted API, no paid calls, so it works on a clean checkout.

    python scripts/demo_support_gates.py            # print the demo
    python scripts/demo_support_gates.py --json      # also write data/eval/support/dual_gate.json
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from amlguard.reidentify import SUPERVISOR, SUPPORT_AGENT, reidentify  # noqa: E402


class _FakeProtector:
    """Deterministic, reversible tokenizer, so the demo needs no hosted API (same shape as the
    Protector: protect_values / unprotect_values by data element)."""

    def __init__(self) -> None:
        self._fwd: dict[str, str] = {}
        self._rev: dict[tuple[str, str], str] = {}
        self._n = 0

    def protect_values(self, values, element):
        out = []
        for v in values:
            if v not in self._fwd:
                self._n += 1
                tok = f"{element[:2].upper()}x{self._n:03d}"
                self._fwd[v] = tok
                self._rev[(element, tok)] = v
            out.append(self._fwd[v])
        return out

    def protect_value(self, value, element):
        return self.protect_values([value], element)[0]

    def unprotect_values(self, tokens, element):
        return [self._rev.get((element, t), t) for t in tokens]


# The customer PII in the inbound message, with the entity type each maps to. In a live run
# this comes from the discovery service; here it is spelled out so the demo is self-contained.
_INBOUND = "My name is Sarah Chen, order ORD-99213, card ending 4532, email sarah@example.com."
_ENTITIES = [
    ("Sarah Chen", "PERSON", "string"),
    ("ORD-99213", "ACCOUNT_NUMBER", "number"),
    ("4532", "CREDIT_CARD", "ccn"),
    ("sarah@example.com", "EMAIL_ADDRESS", "string"),
]


def gate1_protect(text: str, protector: _FakeProtector) -> str:
    """Gate 1: tokenize each detected PII span in place, wrapped so Gate 2 can reverse it."""
    out = text
    for value, entity_type, element in _ENTITIES:
        token = protector.protect_value(value, element)
        out = out.replace(value, f"[{entity_type}]{token}[/{entity_type}]")
    return out


def _mask_view(text: str) -> str:
    """Render still-tokenized spans as a mask, so an agent's screen shows `[PERSON: masked]`
    rather than a raw token, the shape a real support console would display. Same chip format the
    rest of the UI uses (Present stage, Live Assistant), so masks read identically everywhere."""
    return re.sub(r"\[([A-Z_]+)\][^\[]+\[/\1\]", lambda m: f"[{m.group(1)}: masked]", text)


CAVEAT = (
    "Which identifiers each role can reveal is enforced by policy: the same gate that governs access "
    "here can be centrally managed in Protegrity."
)


def run_dual_gate() -> dict:
    """Compute the dual-gate result as structured data (the same content the CLI prints and the
    UI renders): the tokenized inbound, the token-only reply, and the per-role Gate-2 views."""
    protector = _FakeProtector()
    protected_inbound = gate1_protect(_INBOUND, protector)

    name_tok = protector.protect_value("Sarah Chen", "string")
    order_tok = protector.protect_value("ORD-99213", "number")
    card_tok = protector.protect_value("4532", "ccn")
    reply = (
        f"Hello [PERSON]{name_tok}[/PERSON], I found order "
        f"[ACCOUNT_NUMBER]{order_tok}[/ACCOUNT_NUMBER]. A refund to the card ending "
        f"[CREDIT_CARD]{card_tok}[/CREDIT_CARD] has been started."
    )

    roles = []
    for role in (SUPPORT_AGENT, SUPERVISOR):
        r = reidentify(reply, protector, role, strip_tags=True)
        shown = r.text if role is SUPERVISOR else _mask_view(reply)
        roles.append({
            "name": role.name, "label": role.label,
            "may_unprotect": sorted(role.may_unprotect),
            "sees": shown, "revealed": r.revealed, "withheld": r.withheld,
        })
    return {
        "domain": "customer-support",
        "use_case": "dual_gate",
        "inbound": _INBOUND,
        "protected_inbound": protected_inbound,
        "entities_protected": len(_ENTITIES),
        "reply_over_tokens": reply,
        "roles": roles,
        "caveat": CAVEAT,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="write data/eval/support/dual_gate.json")
    args = ap.parse_args()

    result = run_dual_gate()

    print("=" * 74)
    print("CUSTOMER-SUPPORT DUAL-GATE DEMO  (Protegrity domain: customer-support)")
    print("=" * 74)
    print("\n--- GATE 1 (input): classify PII -> tokenize ---")
    print(f"  customer says : {result['inbound']}")
    print(f"  tokenized     : {result['protected_inbound']}")
    print(f"  ({result['entities_protected']} PII entities protected before anything downstream sees them)")
    print("\n--- The assistant's reply (written over tokens, never plaintext) ---")
    print(f"  {result['reply_over_tokens']}")
    print("\n--- GATE 2 (output): role-gated detokenization ---")
    for role in result["roles"]:
        print(f"\n  {role['label']} ({role['name']}) — may_unprotect: {role['may_unprotect'] or 'nothing'}")
        print(f"    sees   : {role['sees']}")
        print(f"    revealed={role['revealed']} withheld={role['withheld']}")
    print("\n" + "-" * 74)
    print("Same protected reply, two role views: the Support Agent resolves the case with the")
    print("customer's name, order and card MASKED; the Supervisor may fully re-identify.")
    print(f"\nNOTE: {CAVEAT}")

    if args.json:
        from amlguard.persist import atomic_write_json
        dest = ROOT / "data" / "eval" / "support" / "dual_gate.json"
        atomic_write_json(dest, result)
        print(f"\nwritten to {dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
