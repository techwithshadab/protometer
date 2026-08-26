"""Determinism probe, run this the hour the API key arrives, before anything else.

Determinism is the gate on the whole architecture. Cross-document entity resolution
requires that protecting the same value twice yields the same token. Nothing in Protegrity's
documentation guarantees that; the evidence is circumstantial. This settles it in a minute.

Three outcomes:

  DETERMINISTIC, expected. Entity resolution is viable; the `external_iv` ablation becomes a real curve point.
  NON-DETERMINISTIC, cross-document entity resolution never worked. Stop and revisit the
                    architecture NOW, on day 1, not on day 9.
  INCONSISTENT, determinism is unreliable. That is itself a finding worth reporting.

Usage:
    export DEV_EDITION_EMAIL=... DEV_EDITION_PASSWORD=... DEV_EDITION_API_KEY=...
    python scripts/check_determinism.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# README step 2 is `cp .env.example .env`; make that instruction true. Nothing read the file,
# so a judge who followed the instructions was told their credentials were missing, which
# looks like their mistake and was ours.
from amlguard.env import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1])

# Values chosen to span data elements the AML corpus actually uses. If tokenization is
# deterministic for some elements but not others, that matters and we want it visible.
PROBES: list[tuple[str, str]] = [
    ("name", "Allison Hill"),
    ("ssn", "755-14-8936"),
    ("ccn", "4111 1111 1111 1111"),
    ("email", "allison.hill@example.com"),
    ("phone", "415-555-0142"),
]

REPEATS = 3
POLICY_USER = os.getenv("DEV_EDITION_POLICY_USER", "superuser")


def require_credentials() -> None:
    missing = [
        var
        for var in ("DEV_EDITION_EMAIL", "DEV_EDITION_PASSWORD", "DEV_EDITION_API_KEY")
        if not os.getenv(var)
    ]
    if missing:
        sys.exit(
            "Missing credentials: "
            + ", ".join(missing)
            + "\nRegister free at https://www.protegrity.com/developers/dev-edition-api"
        )


def main() -> int:
    require_credentials()

    try:
        from appython import Protector
    except ImportError:
        sys.exit("appython not installed. Run: pip install protegrity-ai-developer-python")

    session = Protector().create_session(POLICY_USER)

    print(f"Probing determinism ({REPEATS} repeats per value, policy_user={POLICY_USER!r})\n")

    verdicts: dict[str, str] = {}

    for element, plaintext in PROBES:
        try:
            tokens = [session.protect(plaintext, element) for _ in range(REPEATS)]
        except Exception as exc:  # noqa: BLE001, probe reports failures, never raises
            verdicts[element] = "ERROR"
            print(f"  {element:<8} ERROR  {type(exc).__name__}: {exc}")
            continue

        stable = len(set(tokens)) == 1
        verdicts[element] = "DETERMINISTIC" if stable else "NON-DETERMINISTIC"

        print(f"  {element:<8} {verdicts[element]}")
        print(f"    plaintext: {plaintext}")
        for i, token in enumerate(tokens, 1):
            print(f"    token {i}:   {token}")

        # Round-trip separately: a stable token that will not unprotect is still broken.
        try:
            recovered = session.unprotect(tokens[0], element)
            status = "ok" if recovered == plaintext else f"MISMATCH -> {recovered!r}"
        except Exception as exc:  # noqa: BLE001
            status = f"ERROR {type(exc).__name__}: {exc}"
        print(f"    round-trip: {status}\n")

    # external_iv is the documented mechanism for *breaking* determinism (protector.py:215).
    # If it does not change the token, the determinism ablation has nothing to vary.
    print("external_iv ablation viability:")
    try:
        element, plaintext = PROBES[0]
        plain_token = session.protect(plaintext, element)
        iv_token = session.protect(plaintext, element, external_iv=b"1234")
        if plain_token != iv_token:
            print("  VIABLE, external_iv changes the token; ablation is a real curve point.\n")
        else:
            print("  NOT VIABLE, external_iv did not change the token. Drop the ablation.\n")
    except Exception as exc:  # noqa: BLE001
        print(f"  UNKNOWN, {type(exc).__name__}: {exc}\n")

    distinct = set(verdicts.values()) - {"ERROR"}
    print("=" * 68)
    if not distinct:
        print("VERDICT: ERROR, no probe succeeded. Fix credentials before proceeding.")
        return 2
    if distinct == {"DETERMINISTIC"}:
        print("VERDICT: DETERMINISTIC, entity resolution is viable. Proceed as planned.")
        return 0
    if distinct == {"NON-DETERMINISTIC"}:
        print("VERDICT: NON-DETERMINISTIC, cross-document entity resolution does NOT work.")
        print("         STOP. Revisit the architecture before building further.")
        return 1
    print("VERDICT: INCONSISTENT, determinism varies by data element:")
    for element, verdict in verdicts.items():
        print(f"           {element:<8} {verdict}")
    print("         Treat determinism as unreliable; this is itself a reportable finding.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
