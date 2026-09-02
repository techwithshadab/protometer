"""Leak check, verify no in-scope identifier survives in a protected corpus.

This is the invariant the whole submission rests on: the payload reaching the LLM must
contain no real identifiers. A successful ingestion run proves only that calls succeeded,
not that they covered everything, an entity the detectors miss passes through silently and
ingestion still reports success.

That silent-passthrough failure is exactly what this catches, and it caught a real one:
ORGANIZATION was initially classified as a quasi-identifier, so at `direct` scope the roster
found counterparty organizations and then skipped protecting them, leaving 116 of 300 party
names in the clear while ingestion reported success.

    python scripts/verify_protection.py            # every ingested scope
    python scripts/verify_protection.py direct     # one scope
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# README step 2 is `cp .env.example .env`; make that instruction true.
from protometer.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from protometer.ingest import PARTY_FIELDS  # noqa: E402
from protometer.scopes import CURVE_ORDER, get_scope  # noqa: E402

CORPUS_DIR = ROOT / "data" / "corpus"
PROTECTED_DIR = ROOT / "data" / "protected"

# Values too short or too common to search for literally, a false positive here would be
# noise, not a leak.
MIN_VALUE_LENGTH = 5


def check_scope(scope_name: str) -> tuple[int, list[str]]:
    """Return (leak_count, sample_leaks) for one protected corpus."""
    scope = get_scope(scope_name)
    protected_dir = PROTECTED_DIR / scope.slug
    if not (protected_dir / "narratives.json").exists():
        return -1, [f"{scope_name}: not ingested"]

    clear_parties = json.loads((CORPUS_DIR / "parties.json").read_text())
    protected_narratives = json.loads((protected_dir / "narratives.json").read_text())
    protected_parties = json.loads((protected_dir / "parties.json").read_text())

    # Every clear value whose entity type this scope claims to protect.
    should_be_gone: list[tuple[str, str]] = []
    for party in clear_parties:
        for field_name, entity_type in PARTY_FIELDS.items():
            if not scope.protects(entity_type):
                continue
            value = str(party.get(field_name) or "")
            if len(value) >= MIN_VALUE_LENGTH:
                should_be_gone.append((field_name, value))
        # Organization names reach narratives via the roster rather than PARTY_FIELDS.
        if party.get("party_type") == "organization" and scope.protects("ORGANIZATION"):
            name = str(party.get("full_name") or "")
            if len(name) >= MIN_VALUE_LENGTH:
                should_be_gone.append(("full_name", name))

    # Values the corpus emitted into narrative prose, which the party roster does not cover.
    #
    # Without this the check passes while prose dates and written-out amounts sit in the clear:
    # the detector cannot see those formats, and `parties.json` never contained them,
    # so nothing was ever verifying them.
    clear_narratives = json.loads((CORPUS_DIR / "narratives.json").read_text())
    for narrative in clear_narratives:
        for entity_type, values in (narrative.get("narrative_values") or {}).items():
            if not scope.protects(entity_type):
                continue
            for value in values:
                if len(str(value)) >= MIN_VALUE_LENGTH:
                    should_be_gone.append((f"narrative:{entity_type}", str(value)))

    # Scan the **whole serialized file**, not one field.
    #
    # Scanning `text` alone missed a leak living in a sibling key of the same records: the
    # protected corpus carried a `plaintext_entities` sidecar containing every name the text
    # had just been stripped of, and this check reported PASS on it. A verifier that searches
    # only where the leak was expected cannot find the leak that was not expected.
    # ensure_ascii=False everywhere a blob is scanned: default escaping renders
    # non-ASCII values as \uXXXX and the word-bounded search can never match them -
    # a silent false-PASS class. Latent on this ASCII corpus; structural otherwise.
    narrative_blob = json.dumps(protected_narratives, ensure_ascii=False)
    party_blob = json.dumps(protected_parties, ensure_ascii=False)

    # Transactions were never opened at all, so a total regression in amount or date
    # protection would have been invisible.
    transaction_blob = json.dumps(
        json.loads((protected_dir / "transactions.json").read_text()),
        ensure_ascii=False,
    )


    # One pass per blob rather than one regex scan per value.
    #
    # The naive form is O(values x blob), which on the scaled corpus (600 parties, 122
    # typologies, ~6800 transactions) takes minutes and made the check too slow to run
    # routinely, a safety check nobody runs is not a safety check. Candidate values are
    # compiled into a single alternation and the blobs scanned once each.
    leaks: list[str] = []
    by_value = {value: field for field, value in should_be_gone}
    if by_value:
        # Longest first so a shorter value nested in a longer one does not shadow it.
        alternation = "|".join(
            re.escape(v) for v in sorted(by_value, key=len, reverse=True)
        )
        pattern = re.compile(rf"(?<!\w)({alternation})(?!\w)")
        for label, blob in (
            ("narratives", narrative_blob),
            ("parties.json", party_blob),
            ("transactions.json", transaction_blob),
        ):
            # Occurrences *inside* a protection tag are not leaks. A tokenized value can
            # legitimately contain a clear value as a substring, the protected text held
            # `[AMOUNT]25 point 7 thousand dollars[/AMOUNT]`, correctly tokenized, and the
            # clear value `7 thousand dollars` sits inside it at a word boundary. Only
            # occurrences outside every tag are genuine exposure.
            tagged = [
                (m.start(), m.end())
                for m in re.finditer(r"\[([A-Z_]+)\].*?\[/\1\]", blob, re.DOTALL)
            ]
            # Binary search over the interval starts instead of a linear scan per match.
            # `tagged` is already sorted by construction (finditer yields in order, and
            # protection tags never nest), so the containing candidate is uniquely the
            # interval with the greatest start <= position. The linear form was O(matches x
            # tags), quadratic precisely when leaks are widespread, i.e. when this check
            # matters most.
            import bisect

            tag_starts = [lo for lo, _ in tagged]

            def inside_tag(start: int, tag_starts=tag_starts, tagged=tagged) -> bool:
                index = bisect.bisect_right(tag_starts, start) - 1
                return index >= 0 and start < tagged[index][1]

            for match in pattern.finditer(blob):
                if inside_tag(match.start()):
                    continue
                value = match.group(1)
                entry = f"{by_value[value]}={value!r} in {label}"
                if entry not in leaks:
                    leaks.append(entry)

    leaks = [
        entry
        for entry in leaks
        if not _is_token_collision(entry, clear_parties, protected_parties)
    ]
    leaks.extend(
        _derived_value_leaks(scope, clear_parties, protected_parties, narrative_blob)
    )
    return len(leaks), leaks[:8]


def _is_token_collision(
    entry: str, clear_parties: list[dict], protected_parties: list[dict]
) -> bool:
    """True when a flagged value is one party's *token* coinciding with another's clear value.

    Format-preserving tokenization maps a date to a date and a number to a number, so in a
    small domain two parties will occasionally collide: party A's tokenized
    `date_of_birth` comes out equal to party B's real one. The whole-value scan cannot tell
    that from a genuine leak, it sees a clear value present in the protected output, and
    reported FAIL on three such coincidences across 2,160 dates.

    A value is only exposed if it survives on **the record it belongs to**. Comparing per
    party id distinguishes the two: a real leak means party B's own field still holds party
    B's own value.

    This is deliberately narrow. It only excuses a match when the owning party's field was
    genuinely changed, so a scope that failed to protect anything still fails.
    """
    match = re.match(r"^(\w+)='(.*)' in ", entry)
    if not match:
        return False
    field_name, value = match.group(1), match.group(2)

    protected_by_id = {p.get("party_id"): p for p in protected_parties}
    for party in clear_parties:
        if str(party.get(field_name) or "") != value:
            continue
        protected = protected_by_id.get(party.get("party_id"))
        # The owner still carries its own clear value, a genuine leak.
        if protected is not None and str(protected.get(field_name) or "") == value:
            return False
    return True


def _derived_value_leaks(
    scope, clear_parties: list[dict], protected_parties: list[dict], narrative_blob: str
) -> list[str]:
    """Leaks that survive inside a token rather than beside it.

    The whole-value scan above asks "does this clear string appear in the output". That is a
    much weaker property than "no identifier survives", and the two were being conflated: the
    checker reported PASS at `direct` on a corpus where all 257 organizations kept their real
    name in the domain of their supposedly protected email, because the *whole* value
    `accounts@sablefield-group.com` was genuinely absent while `sablefield-group.com` was not.

    A partial value is still an identifier. These checks look for the fragments:

      * **Email domain**, format-preserving tokenization returns it verbatim.
      * **Address components**, a street name or postcode surviving inside a tokenized
        address is as identifying as the whole line.

    Each is compared against the *clear* value for the same party, so a coincidental match
    between unrelated parties is not reported.
    """
    findings: list[str] = []
    clear_by_id = {p["party_id"]: p for p in clear_parties}

    for protected in protected_parties:
        clear = clear_by_id.get(protected.get("party_id"))
        if not clear:
            continue

        if scope.protects("EMAIL_ADDRESS"):
            clear_email = str(clear.get("email") or "")
            token_email = str(protected.get("email") or "")
            if "@" in clear_email and "@" in token_email:
                clear_domain = clear_email.rsplit("@", 1)[1]
                if clear_domain and clear_domain == token_email.rsplit("@", 1)[1]:
                    findings.append(
                        f"email domain {clear_domain!r} preserved in tokenized address "
                        f"{token_email!r} (party {protected['party_id']})"
                    )

        if scope.protects("ADDRESS"):
            clear_address = str(clear.get("address") or "")
            token_address = str(protected.get("address") or "")
            # Street name and postcode are the identifying components; house numbers alone
            # are not, and would produce noise.
            for component in re.findall(r"\b[A-Z][a-z]{4,}\b|\b\d{5}\b", clear_address):
                if component and component in token_address:
                    findings.append(
                        f"address component {component!r} preserved in tokenized address "
                        f"{token_address!r} (party {protected['party_id']})"
                    )
                    break

    # One line per class rather than 257 identical ones, the point is that the class leaks.
    summarised: list[str] = []
    for prefix in ("email domain", "address component"):
        matching = [f for f in findings if f.startswith(prefix)]
        if matching:
            summarised.append(f"{len(matching)} x {matching[0]}")
    return summarised


def main(argv: list[str]) -> int:
    # `--help` is the first thing anyone types. Treating it as a scope name produced
    # `KeyError: "Unknown scope '--help'"`, which reads as a broken script.
    if any(a in ("-h", "--help") for a in argv[1:]):
        print(__doc__)
        return 0

    scope_names = argv[1:] or [
        s for s in CURVE_ORDER if (PROTECTED_DIR / get_scope(s).slug).exists()
    ]
    if not scope_names:
        sys.exit("No ingested scopes found. Run: python scripts/ingest_all.py")

    failed = False
    for name in scope_names:
        scope = get_scope(name)
        count, samples = check_scope(name)

        if count < 0:
            print(f"  {name:<26} SKIP  {samples[0]}")
            continue
        # The baseline protects nothing, so every value is expected to be present.
        if not scope.entities:
            print(f"  {name:<26} N/A   baseline, nothing is protected by design")
            continue
        if count == 0:
            print(f"  {name:<26} PASS  no in-scope identifier found in output")
        else:
            failed = True
            print(f"  {name:<26} FAIL  {count} leaked values")
            for sample in samples:
                print(f"      {sample}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
