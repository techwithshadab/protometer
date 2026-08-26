"""Customer-support role-gated detokenization (Gate 2), the "agent masked / supervisor all"
pattern. Pins that the two support roles reveal different data from the SAME reply
and that the support agent never sees customer identifiers.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.reidentify import ROLES, SUPERVISOR, SUPPORT_AGENT, reidentify


class _FakeProtector:
    def __init__(self):
        self._rev = {}

    def seed(self, element, token, value):
        self._rev[(element, token)] = value

    def unprotect_values(self, tokens, element):
        return [self._rev.get((element, t), t) for t in tokens]


def _reply(p):
    # a reply carrying wrapped tokens the fake can reverse
    p.seed("string", "STx1", "Sarah Chen")
    p.seed("number", "NUx1", "ORD-99213")
    p.seed("ccn", "CCx1", "4532")
    return ("Hello [PERSON]STx1[/PERSON], order [ACCOUNT_NUMBER]NUx1[/ACCOUNT_NUMBER], "
            "card [CREDIT_CARD]CCx1[/CREDIT_CARD].")


def test_support_agent_sees_no_customer_identifiers():
    p = _FakeProtector()
    r = reidentify(_reply(p), p, SUPPORT_AGENT, strip_tags=True)
    # agent may only see ORGANIZATION/LOCATION -> none of these three reveal
    assert r.revealed == 0 and r.withheld == 3
    for leaked in ("Sarah Chen", "ORD-99213", "4532"):
        assert leaked not in r.text


def test_supervisor_fully_reidentifies():
    p = _FakeProtector()
    r = reidentify(_reply(p), p, SUPERVISOR, strip_tags=True)
    assert r.revealed == 3 and r.withheld == 0
    assert "Sarah Chen" in r.text and "ORD-99213" in r.text and "4532" in r.text


def test_support_roles_registered_and_distinct():
    assert ROLES["support_agent"] is SUPPORT_AGENT
    assert ROLES["supervisor"] is SUPERVISOR
    # the agent's permission set is a strict subset of the supervisor's
    assert SUPPORT_AGENT.may_unprotect < SUPERVISOR.may_unprotect
