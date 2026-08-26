"""Verify every protected corpus derives from the same corpus generation.

The curve attributes score differences to protection scope. That inference holds only if the
scopes differ in **nothing else**, same transactions, same ids, same values apart from the
fields the scope protects.

This check exists because the invariant broke silently and nearly produced a false finding:
after a generator fix, one scope was re-ingested and another was not, and comparing them showed
"detector accuracy degrades from 26/28 to 22/28 under protection". Protection of *names* cannot
affect pattern detection, and the real cause was five transaction dates differing between corpus
generations. A stale scope is invisible in the output and looks exactly like a result.

    python scripts/verify_corpus_parity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.ingest import TRANSACTION_FIELDS  # noqa: E402
from amlguard.scopes import CURVE_ORDER, get_scope  # noqa: E402

PROTECTED = ROOT / "data" / "protected"


def check() -> int:
    scopes = [s for s in CURVE_ORDER if (PROTECTED / get_scope(s).slug).exists()]
    if len(scopes) < 2:
        print("Fewer than two scopes ingested; nothing to compare.")
        return 0

    baseline_name = scopes[0]
    baseline = {
        x["transaction_id"]: x
        for x in json.loads(
            (PROTECTED / get_scope(baseline_name).slug / "transactions.json").read_text()
        )
    }

    failed = False
    for name in scopes[1:]:
        scope = get_scope(name)
        other = {
            x["transaction_id"]: x
            for x in json.loads(
                (PROTECTED / scope.slug / "transactions.json").read_text()
            )
        }

        if set(baseline) != set(other):
            print(
                f"  {name:<28} FAIL  different transaction ids "
                f"({len(set(baseline) ^ set(other))} not shared), regenerate all scopes"
            )
            failed = True
            continue

        # Fields this scope legitimately alters. Everything else must match exactly.
        protected_fields = {
            field
            for field, entity in TRANSACTION_FIELDS.items()
            if scope.protects(entity)
        }

        unexpected: dict[str, int] = {}
        for txn_id, row in baseline.items():
            for field, value in row.items():
                if field in protected_fields:
                    continue
                if other[txn_id].get(field) != value:
                    unexpected[field] = unexpected.get(field, 0) + 1

        if unexpected:
            failed = True
            print(f"  {name:<28} FAIL  unprotected fields differ: {unexpected}")
            print(f"  {'':<28}       scope protects only {sorted(protected_fields) or 'nothing'}")
        else:
            print(
                f"  {name:<28} PASS  matches {baseline_name} except "
                f"{sorted(protected_fields) or '(nothing protected)'}"
            )

    if failed:
        print(
            "\nScopes derive from different corpus generations. Any curve computed across them "
            "\nconfounds protection with corpus variation. Re-ingest every scope."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(check())
