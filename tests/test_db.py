"""The Postgres data layer: fail-soft when the DB is absent, faithful when it is up.

The load-bearing property is fail-soft: with no Postgres, every read returns None (so callers fall
back to the JSON corpus) and the loader no-ops, WITHOUT raising into a request or a run. The
DB-up tests are skipped automatically when Postgres is not reachable, so the suite is green with or
without the container.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard import db  # noqa: E402


def test_schema_mapping_is_per_domain():
    assert db._schema_for("aml") == "aml"
    assert db._schema_for("healthcare") == "healthcare"
    assert db._schema_for("customer-support") == "support"
    # an unknown domain still yields a safe identifier (no hyphens)
    assert db._schema_for("new-domain") == "new_domain"


def test_fail_soft_when_postgres_absent(monkeypatch):
    """With no engine, reads return None and the loader no-ops - never raise."""
    monkeypatch.setattr(db, "engine", lambda: None)
    assert db.read_parties("aml") is None
    assert db.read_table("aml", "transactions") is None
    assert db.corpus_fingerprint("aml") is None
    assert db.load_domain_corpus("aml", ROOT / "data" / "corpus", "deadbeef") == {}


def test_available_reflects_engine(monkeypatch):
    monkeypatch.setattr(db, "engine", lambda: None)
    assert db.available() is False
    monkeypatch.setattr(db, "engine", lambda: object())
    assert db.available() is True


def test_json_shape_restores_corpus_representation():
    """read_table's reverse coercion must turn DB-typed values back into the JSON shape the corpus
    used (numeric -> string, date -> ISO string), so a reconstructed record is json-serialisable.
    A Decimal or datetime.date leaking through would raise 'not JSON serializable' downstream."""
    import json
    from datetime import date
    from decimal import Decimal

    assert db._json_shape(Decimal("67865.05"), "numeric") == "67865.05"
    assert db._json_shape(Decimal("1000"), "numeric") == "1000"        # no trailing-zero blowup
    assert db._json_shape(date(2025, 1, 6), "date") == "2025-01-06"
    assert db._json_shape(5, "integer") == 5                            # already JSON-native
    assert db._json_shape(None, "numeric") is None
    # the whole point: the result serialises
    rec = {"amount": db._json_shape(Decimal("12.50"), "numeric"),
           "value_date": db._json_shape(date(2025, 1, 6), "date")}
    assert json.loads(json.dumps(rec)) == {"amount": "12.50", "value_date": "2025-01-06"}


# ---- DB-up tests: skipped automatically when Postgres is not reachable -------------------------

def _pg_up() -> bool:
    try:
        # Reset the memoized engine so a real probe runs in this test session.
        db._engine = None
        db._engine_tried = False
        return db.available()
    except Exception:  # noqa: BLE001
        return False


pg = pytest.mark.skipif(not _pg_up(), reason="Postgres not reachable")


@pg
def test_loader_is_idempotent_and_faithful(tmp_path):
    """Loading twice yields the same counts, and the rows match the source JSON."""
    corpus = ROOT / "data" / "corpus"
    if not (corpus / "parties.json").exists():
        pytest.skip("no corpus to load")
    # load into a throwaway schema so we never disturb the real aml schema. Drop it FIRST too, so a
    # previous crashed run that skipped teardown can't leave stale rows that break the idempotency
    # assertion (order-independence under the shared test Postgres).
    domain = "test_db_domain"
    from sqlalchemy import text
    with db.engine().begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{db._schema_for(domain)}" CASCADE'))
    c1 = db.load_domain_corpus(domain, corpus, "fp-test")
    c2 = db.load_domain_corpus(domain, corpus, "fp-test")
    assert c1 == c2 and c1  # idempotent, non-empty
    parties = db.read_table(domain, "parties")
    source = json.loads((corpus / "parties.json").read_text())
    assert parties is not None and len(parties) == len(source) == c1["parties"]
    assert db.corpus_fingerprint(domain) == "fp-test"
    # Every reconstructed table must be JSON-serialisable: numeric/date columns come back as the
    # corpus's string shape, never raw Decimal/date, so an endpoint serialising these rows can't 500.
    if (corpus / "transactions.json").exists():
        txns = db.read_table(domain, "transactions")
        assert txns and json.dumps(txns[0])  # no "Object of type Decimal/date is not serializable"
        src_txn = {t["transaction_id"]: t for t in json.loads((corpus / "transactions.json").read_text())}
        got = txns[0]
        exp = src_txn[got["transaction_id"]]
        # amount comes back a decimal STRING (not Decimal) and is numerically faithful. Trailing-zero
        # scale is not recoverable (the NUMERIC column has no fixed scale, so 2381.00 stores as 2381)
        # — that is a load-time property, not a serialisation bug; equality is numeric.
        assert isinstance(got["amount"], str) and float(got["amount"]) == float(exp["amount"])
        assert str(got["value_date"]) == str(exp["value_date"])  # date round-trips as ISO string
    # teardown the throwaway schema
    from sqlalchemy import text
    with db.engine().begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{db._schema_for(domain)}" CASCADE'))
