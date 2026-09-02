"""Measure which value formats Protegrity's discovery service actually detects.

No Protegrity artifact documents this, and the answer determines whether a protection scope
means what it claims: a scope declaring it protects AMOUNT and DATETIME protects **nothing**
in a sentence whose date and amount are written the way people write them.

Requires the local discovery stack (docker compose in vendor-de/data-discovery).

    python scripts/format_coverage.py
    python scripts/format_coverage.py --format markdown > docs/format-coverage.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.ingest import discover_entities  # noqa: E402


@dataclass(frozen=True)
class FormatCase:
    entity_type: str
    label: str
    value: str
    # Embedded in a sentence: detectors are context-sensitive, and a bare value is not a fair
    # test of how the field appears in a real record.
    sentence: str


CASES: tuple[FormatCase, ...] = (
    # -- dates ---------------------------------------------------------------------
    FormatCase("DATETIME", "ISO 8601", "2025-04-15", "Wire settled on 2025-04-15."),
    FormatCase("DATETIME", "US slash", "04/15/2025", "Wire settled on 04/15/2025."),
    FormatCase("DATETIME", "EU slash", "15/04/2025", "Wire settled on 15/04/2025."),
    FormatCase("DATETIME", "dotted", "15.04.2025", "Wire settled on 15.04.2025."),
    FormatCase("DATETIME", "prose (D M Y)", "15 April 2025", "Wire settled on 15 April 2025."),
    FormatCase("DATETIME", "prose (M D, Y)", "April 15, 2025", "Wire settled on April 15, 2025."),
    FormatCase("DATETIME", "abbreviated", "15-Apr-25", "Wire settled on 15-Apr-25."),
    # -- amounts -------------------------------------------------------------------
    FormatCase("AMOUNT", "bare decimal", "712500.00", "Transfer of 712500.00 was made."),
    FormatCase("AMOUNT", "US currency", "$712,500.00", "Transfer of $712,500.00 was made."),
    FormatCase("AMOUNT", "code + suffix", "USD 712.5k", "Transfer of USD 712.5k was made."),
    FormatCase("AMOUNT", "EU decimal", "712.500,00 EUR", "Transfer of 712.500,00 EUR was made."),
    FormatCase("AMOUNT", "written out", "seven hundred twelve thousand dollars",
               "Transfer of seven hundred twelve thousand dollars was made."),
    FormatCase("AMOUNT", "millions shorthand", "$1.2M", "Transfer of $1.2M was made."),
    # -- identifiers, for contrast --------------------------------------------------
    FormatCase("SOCIAL_SECURITY_ID", "hyphenated", "900-12-3456", "SSN 900-12-3456 on file."),
    FormatCase("SOCIAL_SECURITY_ID", "unseparated", "900123456", "SSN 900123456 on file."),
    FormatCase("PHONE_NUMBER", "hyphenated", "415-555-0142", "Call 415-555-0142 for details."),
    FormatCase("PHONE_NUMBER", "parenthesised", "(415) 555-0142", "Call (415) 555-0142 for details."),
    FormatCase("PHONE_NUMBER", "international", "+1 415 555 0142", "Call +1 415 555 0142 for details."),
    FormatCase("CREDIT_CARD", "spaced", "4111 1111 1111 1111", "Card 4111 1111 1111 1111 was used."),
    FormatCase("CREDIT_CARD", "unseparated", "4111111111111111", "Card 4111111111111111 was used."),
    FormatCase("CREDIT_CARD", "hyphenated", "4111-1111-1111-1111", "Card 4111-1111-1111-1111 was used."),
    # -- other elements, for completeness of the published table ---------------------
    FormatCase("EMAIL_ADDRESS", "plain", "a.b@example.com", "Contact a.b@example.com for details."),
    FormatCase("EMAIL_ADDRESS", "subaddressed", "first.last+tag@sub.domain.co.uk",
               "Contact first.last+tag@sub.domain.co.uk for details."),
    FormatCase("IP_ADDRESS", "IPv4", "192.168.1.1", "Access logged from 192.168.1.1 overnight."),
    FormatCase("MAC_ADDRESS", "colon-separated", "00:1A:2B:3C:4D:5E",
               "Device 00:1A:2B:3C:4D:5E connected."),
    FormatCase("URL", "https", "https://example.com/a?b=1",
               "Referral came from https://example.com/a?b=1 last week."),
    FormatCase("CRYPTO_ADDRESS", "bech32", "bc1qxy2kgdygjrsqtzq2n0yrf249",
               "Funds moved to bc1qxy2kgdygjrsqtzq2n0yrf249 on chain."),
    FormatCase("ADDRESS", "US street", "1369 Riverbend Ln, Reno, NV 89501",
               "Registered at 1369 Riverbend Ln, Reno, NV 89501 since 2019."),
)


def evaluate(case: FormatCase) -> tuple[str, str]:
    """Return (verdict, detail) for one format case."""
    try:
        entities = discover_entities(case.sentence)
    except Exception as exc:  # noqa: BLE001, a service failure is a reportable outcome
        return "ERROR", str(exc)[:60]

    # Detected as the expected type, covering the whole value?
    for entity in entities:
        if entity["entity_type"] != case.entity_type:
            continue
        if entity["text"].strip() == case.value.strip():
            return "FULL", entity["entity_type"]
        if entity["text"].strip() and entity["text"].strip() in case.value:
            return "PARTIAL", f"matched {entity['text']!r} of {case.value!r}"

    # Detected, but as something else, still protected, though under the wrong element.
    others = {e["entity_type"] for e in entities}
    if others:
        return "MISCLASSIFIED", ", ".join(sorted(others))
    return "MISSED", "no entity detected"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("text", "markdown"), default="text")
    args = parser.parse_args()
    markdown = args.format == "markdown"

    if markdown:
        print("# Discovery format coverage\n")
        print("Measured against Protegrity Developer Edition's Data Discovery service.\n")
        print("| Entity | Format | Example | Result | Detail |")
        print("|---|---|---|---|---|")
    else:
        print(f"{'entity':<20}{'format':<20}{'result':<16}detail")
        print("-" * 92)

    tally: dict[str, int] = {}
    # Verdicts are computed once and reused. The summary below previously called `evaluate`
    # a second time for every case, doubling the HTTP traffic and, worse, allowing the table
    # and the warning to be derived from two different measurements of a flaky service.
    verdicts: list[tuple[str, str]] = []
    for case in CASES:
        verdict, detail = evaluate(case)
        verdicts.append((verdict, detail))
        tally[verdict] = tally.get(verdict, 0) + 1
        if markdown:
            print(
                f"| {case.entity_type} | {case.label} | `{case.value}` | "
                f"{'**' + verdict + '**' if verdict == 'MISSED' else verdict} | {detail} |"
            )
        else:
            print(f"{case.entity_type:<20}{case.label:<20}{verdict:<16}{detail}")

    summary = "  ".join(f"{k}={v}" for k, v in sorted(tally.items()))
    print(f"\n{summary}")

    # A service that is down produces ERROR for every case, which left `missed` empty, the
    # warning suppressed, and the exit code 0, "no formats missed" and "nothing was measured"
    # were the same result. For a coverage report that is the one confusion that matters.
    errors = tally.get("ERROR", 0)
    if errors == len(CASES):
        print(
            "\nEvery probe errored, the discovery service did not answer, so this table "
            "measures nothing.\nStart it with: cd vendor-de/data-discovery && docker compose "
            "up -d"
        )
        return 2
    if errors:
        print(f"\nWARNING: {errors} of {len(CASES)} probes errored; those rows are not evidence.")

    if any(v == "MISSED" for v, _ in verdicts):
        note = (
            "\nFormats returning no entity pass through **unprotected** even when the active "
            "scope claims to protect that type."
        )
        print(note if markdown else note.replace("**", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
