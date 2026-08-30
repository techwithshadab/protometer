"""The reveal ledger is tamper-evident and the canary tripwire fires; scope-bound reveal narrows
what an authorized role may detokenize. These pin the three added protection defenses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amlguard import reidentify as R  # noqa: E402
from amlguard.reveal_ledger import CanaryTripwire, RevealLedger  # noqa: E402


def test_ledger_chains_and_verifies(tmp_path):
    ledger = RevealLedger(path=tmp_path / "l.jsonl")
    h1 = ledger.append(actor="app", role="investigator", purpose="case",
                       entity_counts={"PERSON": 2})
    h2 = ledger.append(actor="app", role="investigator", purpose="case",
                       entity_counts={"ACCOUNT_NUMBER": 1})
    assert h1 != h2
    ok, broken = ledger.verify_chain()
    assert ok and broken == 0
    # the second record chains off the first
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    assert json.loads(lines[1])["prev"] == json.loads(lines[0])["hash"]


def test_ledger_detects_tampering(tmp_path):
    p = tmp_path / "l.jsonl"
    ledger = RevealLedger(path=p)
    ledger.append(actor="a", role="investigator", purpose="case", entity_counts={"PERSON": 1})
    ledger.append(actor="a", role="investigator", purpose="case", entity_counts={"PERSON": 1})
    ledger.append(actor="a", role="investigator", purpose="case", entity_counts={"PERSON": 1})
    # Tamper with the middle record's payload without recomputing hashes.
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1]); rec["entity_counts"] = {"PERSON": 999}
    lines[1] = json.dumps(rec, separators=(",", ":"))
    p.write_text("\n".join(lines) + "\n")
    ok, broken = ledger.verify_chain()
    assert not ok and broken == 2  # 1-indexed: the altered second line


class _FakeProtector:
    """Reverses tokens of the form TOK:<value> to <value>, per element (element ignored)."""

    def unprotect_values(self, tokens, element):
        return [t.split("TOK:", 1)[1] if "TOK:" in t else t for t in tokens]


def _doc(*pairs):
    # pairs of (entity_type, token) -> a tagged document
    return " ".join(f"[{e}]{t}[/{e}]" for e, t in pairs)


def test_scope_bound_reveal_narrows_an_authorized_role():
    prot = _FakeProtector()
    # investigator MAY see PERSON, but scope authorizes only Alice's token.
    doc = _doc(("PERSON", "TOK:Alice"), ("PERSON", "TOK:Bob"))
    res = R.reidentify(doc, prot, role=R.ROLES["investigator"],
                       scope_tokens=frozenset({"TOK:Alice"}))
    assert "Alice" in res.text and "TOK:Bob" in res.text  # Bob stays protected
    assert res.revealed == 1 and res.out_of_scope == 1


def test_canary_tripwire_fires_on_reveal():
    prot = _FakeProtector()
    trip = CanaryTripwire(values=frozenset({"Eve Canary"}))
    doc = _doc(("PERSON", "TOK:Eve Canary"), ("PERSON", "TOK:Alice"))
    res = R.reidentify(doc, prot, role=R.ROLES["investigator"], tripwire=trip)
    assert res.canary_hits == 1 and trip.tripped == 1


def test_ledger_records_a_real_reveal(tmp_path):
    prot = _FakeProtector()
    ledger = RevealLedger(path=tmp_path / "l.jsonl")
    doc = _doc(("PERSON", "TOK:Alice"))
    R.reidentify(doc, prot, role=R.ROLES["investigator"], ledger=ledger,
                 actor="tester", purpose="unit")
    ok, _ = ledger.verify_chain()
    assert ok
    rec = json.loads((tmp_path / "l.jsonl").read_text().splitlines()[0])
    assert rec["role"] == "investigator" and rec["entity_counts"] == {"PERSON": 1}
    assert "Alice" not in json.dumps(rec)  # ledger never stores plaintext
