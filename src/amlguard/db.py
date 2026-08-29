"""The app's Postgres data layer: a queryable mirror of the corpus, per domain.

A deep module. Callers see a tiny interface - `engine()`, `load_domain_corpus()`, `read_table()`,
`read_parties()`, `available()` - and never touch SQL, connection strings, or psycopg. Everything
about how the data is stored (per-domain schemas, JSONB columns, idempotent reload) lives here.

Two invariants shape the design, and note that this LIBRARY layer and the ONLINE APP make
opposite choices on the same fact — deliberately:

  * **For the offline pipeline, the file corpus stays the source of truth.** `data/corpus/*.json`
    is what the pipeline fingerprints and reproduces from; this DB is a *mirror* the loader rebuilds
    from those files. We never let Postgres become a second place the corpus could drift - the
    loader always derives from the JSON, and stamps the corpus fingerprint so a mirror's provenance
    is explicit.
  * **The library layer is fail-soft; the online app is NOT.** These `read_*` functions return
    None/empty when Postgres is unavailable, so a batch script may still fall back to the JSON files.
    But the demo/serving app (ui/api/app.py) treats Postgres as its source of truth and raises 503
    with NO JSON fallback: a fallback there would silently serve data the operator never
    loaded and mask a misconfigured deployment. So "fail soft" describes THIS module's return
    contract, not the app's request behaviour.

The schema is RELATIONAL and follows the corpus: `load_domain_corpus` builds each table from
`CORPUS_SCHEMA` (typed columns + foreign keys to parties + indexes), with any non-schema fields
preserved in a per-row `extra` jsonb, so `read_table` reconstructs the original record shape.
Adding a modelled field means extending CORPUS_SCHEMA; unmodelled fields ride along in `extra`
with no migration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from amlguard import settings as _settings
from amlguard.log import get_logger

_log = get_logger("db")

# ---------------------------------------------------------------------------------------------
# Relational corpus schema. Each table is typed columns + a JSONB `extra` for anything
# not modelled, so the relational shape follows the corpus without losing fields. Foreign keys
# encode the real graph (transaction/alert -> party), and indexes cover the join + filter keys the
# app actually queries. `amount` values arrive as strings in the JSON and are CAST to numeric.
#
# Each column spec is (json_key, sql_type). `pk` is the primary key column. `fks` is a list of
# (column, referenced_table, referenced_column). `indexes` lists columns to index (join/filter
# keys). Anything in the JSON but not in `columns` is preserved in the `extra jsonb` column.
#
# NOTE: `parties` and `narratives` hold CLEARTEXT PII (names, SSN, accounts) - the same clear data
# as data/corpus/none. This mirror is a local, gitignored convenience layer, not a place protected
# data lives; the protected corpus stays on the filesystem per scope.
CORPUS_SCHEMA: dict[str, dict] = {
    "parties": {
        "pk": "party_id",
        "columns": [
            ("party_id", "text"), ("party_type", "text"), ("full_name", "text"),
            ("account_number", "text"), ("bank_account", "text"), ("jurisdiction", "text"),
            ("address", "text"), ("city", "text"), ("email", "text"), ("phone", "text"),
            ("ssn", "text"), ("date_of_birth", "text"), ("tax_id", "text"),
            ("credit_card", "text"), ("risk_rating", "text"), ("is_pep", "boolean"),
        ],
        "fks": [],
        "indexes": ["party_type", "is_pep", "jurisdiction", "risk_rating"],
    },
    "transactions": {
        "pk": "transaction_id",
        "columns": [
            ("transaction_id", "text"), ("origin_party_id", "text"),
            ("beneficiary_party_id", "text"), ("amount", "numeric"), ("currency", "text"),
            ("value_date", "date"), ("channel", "text"), ("memo", "text"),
        ],
        # FKs to parties encode the transaction graph. Deferred/not-validated is unnecessary here:
        # parties load first, so referents exist.
        "fks": [("origin_party_id", "parties", "party_id"),
                ("beneficiary_party_id", "parties", "party_id")],
        "indexes": ["origin_party_id", "beneficiary_party_id", "channel", "value_date"],
    },
    "alerts": {
        "pk": "alert_id",
        "columns": [
            ("alert_id", "text"), ("subject_party_id", "text"), ("typology_id", "text"),
            ("scenario_id", "text"), ("raised_on", "date"), ("reason", "text"),
            ("score", "integer"), ("status", "text"), ("disposition", "text"),
            ("held_out_disposition", "text"), ("held_out_rationale", "text"),
            ("triggering_transaction_id", "text"),
            ("prior_match_count_same_scenario", "integer"),
            ("prior_match_count_all", "integer"), ("escalated", "boolean"),
        ],
        # subject_party_id -> parties. triggering_transaction_id is nullable and points at a txn;
        # we index it but do NOT FK it (some alerts have no triggering txn, and enforcing it adds
        # no query value while risking a load failure on an edge case).
        "fks": [("subject_party_id", "parties", "party_id")],
        "indexes": ["subject_party_id", "scenario_id", "status", "typology_id",
                    "triggering_transaction_id"],
    },
    "ground_truth": {
        # typology_id repeats across rows, so a surrogate row id is the PK; typology_id is indexed.
        "pk": "id",
        "columns": [
            ("id", "text"), ("typology_id", "text"), ("typology", "text"),
            ("subject_party_id", "text"), ("total_amount", "numeric"),
        ],
        "fks": [("subject_party_id", "parties", "party_id")],
        "indexes": ["typology_id", "subject_party_id"],
    },
    "narratives": {
        "pk": "document_id",
        "columns": [
            ("document_id", "text"), ("document_type", "text"),
            ("subject_party_id", "text"), ("typology_id", "text"), ("text", "text"),
        ],
        "fks": [("subject_party_id", "parties", "party_id")],
        "indexes": ["subject_party_id", "typology_id", "document_type"],
    },
}

# Load order matters: parties first (they are the FK target of every other table).
LOAD_ORDER = ["parties", "transactions", "alerts", "ground_truth", "narratives"]

# Domain name -> Postgres schema. The domain seam, mirrored in the database namespace.
DOMAIN_SCHEMA: dict[str, str] = {
    "aml": "aml",
    "healthcare": "healthcare",
    "customer-support": "support",
}

_engine = None
_engine_tried = False


def _schema_for(domain: str) -> str:
    """The Postgres schema for a KNOWN domain. Strict allowlist: schema names are interpolated
    into SQL identifiers, so an unknown domain must never produce one. Raises KeyError."""
    return DOMAIN_SCHEMA[domain]


def engine():
    """The process-wide SQLAlchemy engine, or None when Postgres is unavailable.

    Built once and memoized. A failed build (driver missing, DB down) is cached as None so callers
    take the file-fallback path without re-probing a dead server on every call.
    """
    global _engine, _engine_tried
    if _engine_tried:
        return _engine
    _engine_tried = True
    try:
        from sqlalchemy import create_engine

        # Force the psycopg v3 driver: SQLAlchemy maps a bare `postgresql://` to psycopg2, which we
        # do not install. `postgresql+psycopg://` selects psycopg 3 explicitly.
        url = _settings.postgres_url().replace("postgresql://", "postgresql+psycopg://", 1)
        eng = create_engine(url, pool_pre_ping=True, future=True)
        # One cheap probe so a down server costs one failure, not a hang per query.
        with eng.connect() as conn:
            from sqlalchemy import text

            conn.execute(text("SELECT 1"))
        _engine = eng
    except Exception as exc:  # noqa: BLE001, any failure means the same thing: use the file path
        _log.info("Postgres unavailable (%s); callers fall back to JSON files", type(exc).__name__)
        _engine = None
    return _engine


def available() -> bool:
    """True when Postgres is reachable. Callers use this to choose DB vs. file reads."""
    return engine() is not None


def _coerce(value: Any, sql_type: str) -> Any:
    """Coerce a JSON value to what the typed column expects.

    Numeric/integer/date empties become NULL (an empty string is not a valid number/date). TEXT
    columns PRESERVE the empty string, so `ssn: ""` round-trips as `""` (not None) and the mirror
    matches the JSON record shape faithfully for the fields the app reads.
    """
    if value is None:
        return None
    if sql_type == "text":
        return str(value)
    if value == "":
        return None
    if sql_type in ("numeric", "integer"):
        try:
            return float(value) if sql_type == "numeric" else int(value)
        except (TypeError, ValueError):
            return None
    if sql_type == "boolean":
        return bool(value)
    if sql_type == "date":
        return str(value)  # ISO string; Postgres casts text->date on insert
    return str(value)


def _json_shape(value: Any, sql_type: str | None) -> Any:
    """Inverse of `_coerce`: turn a DB-typed value back into the JSON representation the corpus
    file used, so a reconstructed record is json-serialisable and byte-shape-compatible.

    numeric -> str (the corpus stored amounts as strings like "67865.05"); date -> ISO str;
    integer/boolean/text pass through (already JSON-native). None stays None. `extra` (jsonb) has
    no schema type and is handled by the caller.
    """
    if value is None:
        return None
    if sql_type == "numeric":
        # Decimal -> plain decimal string (never scientific notation). We do NOT .normalize(): that
        # would strip trailing zeros ("12.50" -> "12.5") and drift from the corpus's stored scale.
        # `format(d, "f")` renders the Decimal's own scale, so a value stored as 12.50 comes back
        # "12.50". Integers-as-numeric render without a spurious ".0".
        from decimal import Decimal
        if isinstance(value, Decimal):
            return format(value, "f")
        return str(value)
    if sql_type == "date":
        # datetime.date -> ISO string; already-str values (unlikely) pass through.
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    return value


def _create_table_sql(schema: str, table: str, spec: dict) -> list[str]:
    """DDL for one typed table: columns, PK, FKs, and its indexes."""
    cols = [f'"{c}" {t}' for c, t in spec["columns"]]
    cols.append("extra jsonb")  # un-modelled fields preserved here
    ddl = [
        f'DROP TABLE IF EXISTS "{schema}"."{table}" CASCADE',
        f'CREATE TABLE "{schema}"."{table}" ('
        + ", ".join(cols)
        + f', PRIMARY KEY ("{spec["pk"]}")'
        + "".join(
            f', FOREIGN KEY ("{col}") REFERENCES "{schema}"."{ref_t}" ("{ref_c}")'
            for col, ref_t, ref_c in spec.get("fks", [])
        )
        + ")",
    ]
    for col in spec.get("indexes", []):
        ddl.append(f'CREATE INDEX ON "{schema}"."{table}" ("{col}")')
    return ddl


def load_domain_corpus(domain: str, corpus_dir: Path, fingerprint: str = "") -> dict[str, int]:
    """Load a domain's corpus JSON into a typed relational schema, idempotently. {table: count}.

    Builds each table from CORPUS_SCHEMA (typed columns + FKs + indexes + a jsonb `extra`), in
    dependency order (parties first, they are the FK target), and bulk-inserts coerced values. A
    `_corpus_meta` row records the fingerprint the mirror was loaded from. No-op when Postgres is
    down. The file corpus stays the source of truth; this rebuilds the mirror from it.
    """
    eng = engine()
    if eng is None:
        return {}
    from sqlalchemy import text

    schema = _schema_for(domain)
    counts: dict[str, int] = {}
    with eng.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        conn.execute(text(
            f'CREATE TABLE IF NOT EXISTS "{schema}"._corpus_meta '
            f'(key text primary key, value text)'))
        # Drop child tables first so parties can be recreated without FK-dependency errors.
        for table in reversed(LOAD_ORDER):
            conn.execute(text(f'DROP TABLE IF EXISTS "{schema}"."{table}" CASCADE'))
        for table in LOAD_ORDER:
            spec = CORPUS_SCHEMA[table]
            path = corpus_dir / f"{table}.json"
            if not path.exists():
                continue
            records = json.loads(path.read_text())
            for stmt in _create_table_sql(schema, table, spec):
                conn.execute(text(stmt))
            col_names = [c for c, _ in spec["columns"]]
            col_types = dict(spec["columns"])
            rows = []
            for i, rec in enumerate(records):
                row: dict[str, Any] = {}
                for c in col_names:
                    if c == "id" and table == "ground_truth":
                        row["id"] = f"{rec.get('typology_id', '')}:{i}"  # surrogate PK
                    else:
                        row[c] = _coerce(rec.get(c), col_types[c])
                # everything not modelled goes to extra
                extra = {k: v for k, v in rec.items() if k in _extra_keys(rec, col_names, table)}
                row["extra"] = json.dumps(extra) if extra else None
                rows.append(row)
            if rows:
                placeholders = ", ".join(f":{c}" for c in col_names) + ", CAST(:extra AS jsonb)"
                collist = ", ".join(f'"{c}"' for c in col_names) + ", extra"
                conn.execute(
                    text(f'INSERT INTO "{schema}"."{table}" ({collist}) VALUES ({placeholders})'),
                    rows,
                )
            counts[table] = len(rows)
        if fingerprint:
            conn.execute(
                text(f'INSERT INTO "{schema}"._corpus_meta (key, value) VALUES (:k, :v) '
                     f'ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value'),
                {"k": "corpus_fingerprint", "v": fingerprint},
            )
    _log.info("loaded %s corpus into schema %s (relational): %s", domain, schema, counts)
    return counts


def _extra_keys(rec: dict, col_names: list[str], table: str) -> set[str]:
    """JSON keys not mapped to a typed column (kept in `extra`). `id` is synthetic, not a json key."""
    modelled = {c for c in col_names if not (c == "id" and table == "ground_truth")}
    return {k for k in rec if k not in modelled}


def read_table(domain: str, table: str) -> list[dict[str, Any]] | None:
    """All records of a domain table as dicts, or None when Postgres is unavailable/absent.

    Reconstructs each record from its typed columns plus the `extra` jsonb, so callers get the same
    record shape they would from the JSON file (minus the synthetic ground_truth `id`). None (not
    []) signals "use the file fallback"; an empty list means the table exists but is empty.
    """
    eng = engine()
    if eng is None:
        return None
    from sqlalchemy import text

    # Identifiers are interpolated into the query, so both come from constant allowlists:
    # an unknown domain or table is refused here, never quoted.
    schema = DOMAIN_SCHEMA.get(domain)
    if schema is None:
        return None
    spec = CORPUS_SCHEMA.get(table)
    if spec is None:
        return None
    table = {name: name for name in CORPUS_SCHEMA}[table]
    # Reverse coercion map: the DB returns numeric columns as Decimal and date columns as
    # datetime.date, but the JSON corpus stored them as STRINGS (amount "67865.05", value_date
    # "2025-01-06"). Reconstruct that shape so a reconstructed record is json.dumps-able —
    # otherwise any caller that serialises these rows (an endpoint, a report) would raise "Object
    # of type Decimal/date is not JSON serializable". Values are numerically faithful; trailing-zero
    # scale on amounts is not preserved (the NUMERIC column carries no fixed scale), which is a
    # load-time property, not a serialisation concern.
    col_types = dict(spec["columns"])
    try:
        with eng.connect() as conn:
            result = conn.execute(text(f'SELECT * FROM "{schema}"."{table}"'))
            cols = list(result.keys())
            out: list[dict[str, Any]] = []
            for row in result.fetchall():
                rec = {k: _json_shape(v, col_types.get(k)) for k, v in zip(cols, row)}
                extra = rec.pop("extra", None) or {}
                # drop the synthetic surrogate PK for ground_truth (not a real corpus field)
                if spec and spec["pk"] == "id" and table == "ground_truth":
                    rec.pop("id", None)
                rec.update(extra)
                out.append(rec)
        return out
    except Exception as exc:  # noqa: BLE001, missing table / transient error -> file fallback
        _log.info("read_table(%s.%s) fell back to files (%s)", schema, table, type(exc).__name__)
        return None


def read_parties(domain: str = "aml") -> list[dict[str, Any]] | None:
    """A domain's parties from Postgres, or None to signal the caller should read parties.json."""
    return read_table(domain, "parties")


def corpus_fingerprint(domain: str = "aml") -> str | None:
    """The fingerprint the domain's mirror was loaded from, or None if unavailable/unstamped."""
    eng = engine()
    if eng is None:
        return None
    from sqlalchemy import text

    schema = DOMAIN_SCHEMA.get(domain)
    if schema is None:
        return None
    try:
        with eng.connect() as conn:
            row = conn.execute(text(
                f'SELECT value FROM "{schema}"._corpus_meta WHERE key = :k'),
                {"k": "corpus_fingerprint"}).fetchone()
        return row[0] if row else None
    except Exception:  # noqa: BLE001
        return None
