#!/usr/bin/env python3
"""Build `data/protected/token-manifest.json`: the protection-token set the SERVING guardrail
needs for its surrogate-discount, shipped in the image.

Why this exists
---------------
The egress guardrail discounts a flagged span when it is one of the pipeline's own protection
tokens (a `[PERSON]` surrogate the model echoed, which the vendor's data-discovery mislabels as
PASSWORD). It learns the token set from the protected corpus on disk. But the serving container
`.dockerignore`s the large per-scope `parties.json`/`transactions.json` (they are big and some
scopes hold partially-clear values), so in the container the token set is EMPTY and the discount
silently dies — roughly half of live chat turns get a safe reply withheld.

This script distills those artifacts into ONE compact, token-ONLY manifest (~1.9MB, already
normalized) that IS shipped (whitelisted in `.dockerignore`). It contains no clear values: only the
genuinely-tokenizing scopes are read (the cleartext baseline and its copies are excluded), exactly
as the runtime loader did. The runtime STILL applies Rail 2 (subtract every forbidden clear value)
on top, so even a mistake here cannot discount a real leak.

Run it after (re)protecting the corpus, e.g. as the last step of ingest:
    python scripts/build_token_manifest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.guardrail import _normalize_for_match  # noqa: E402

# Scopes whose structured artifacts hold CLEAR or partially-clear values, never a token source.
# NOTE the disk scope is `quasi` (keeps some quasi-identifiers clear); include it here so its clear
# fields are not harvested. (The runtime loader's list predates this scope name; the field-level
# allow-list below is the real safety, so a scope-name drift cannot leak a clear value.)
_NON_TOKEN_SCOPES = frozenset({"none", "_anon-monetary", "quasi-yearclear", "quasi"})

# FREE-TEXT / CATEGORICAL fields that are CLEAR by construction (business descriptions, enums,
# currency codes) — never tokenized identifiers. Harvesting them puts generic English words
# ("consulting", "management", "services" from a `memo`, "medium" from `risk_rating`) into the token
# set, which is noise at best and, worse, a word shared with a real org name. Skip them: the manifest
# should hold ONLY protection tokens. Everything NOT in this set is treated as a potential
# identifier field. (Runtime Rail 2 still subtracts forbidden clear values as a second guard.)
_CLEAR_TEXT_FIELDS = frozenset({
    "memo", "party_type", "jurisdiction", "channel", "currency", "risk_rating", "is_pep",
    "city", "amount", "value_date",
})


def build() -> frozenset[str]:
    base = ROOT / "data" / "protected"
    toks: set[str] = set()
    if not base.exists():
        return frozenset()
    for scope_dir in sorted(base.glob("*")):
        if not scope_dir.is_dir() or scope_dir.name in _NON_TOKEN_SCOPES:
            continue
        for fname in ("parties.json", "transactions.json"):
            fp = scope_dir / fname
            if not fp.exists():
                continue
            for rec in json.loads(fp.read_text()):
                for k, v in rec.items():
                    if k in _CLEAR_TEXT_FIELDS:
                        continue  # free-text / categorical: clear by construction, not a token
                    if isinstance(v, str) and len(v) >= 5:
                        toks.add(_normalize_for_match(v))
                        # Per-word spans: the service splits a multi-word token into per-word
                        # findings, so store each word (len >= 3) too. Same rule as the loader.
                        for word in v.split():
                            if len(word) >= 3:
                                toks.add(_normalize_for_match(word))
    return frozenset(t for t in toks if t)


def main() -> int:
    tokens = build()
    out = ROOT / "data" / "protected" / "token-manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(tokens)))
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out.relative_to(ROOT)}: {len(tokens)} tokens, {size_kb:.1f} KB")
    if not tokens:
        print("WARNING: 0 tokens — is data/protected populated? (run ingest/protect first)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
