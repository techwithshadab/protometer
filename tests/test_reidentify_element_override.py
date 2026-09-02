"""Re-identification must reverse a token under the SAME element it was protected with.

A scope with element_overrides (e.g. quasi-yearclear protects dates with `datetime_yc`, not the
default `datetime`) emits a `[TYPE|element]token[/TYPE]` tag. If re-identification ignored that
element and reversed via the global ENTITY_TO_ELEMENT map, it would unprotect a datetime_yc token
under `datetime` — which does not round-trip, silently showing a wrong/lost date at the exact
boundary whose job is faithful re-identification. These tests pin that the tag's element wins.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.reidentify import INVESTIGATOR, find_tokens, reidentify  # noqa: E402

# Opaque tokens that do NOT embed their plaintext (a real token doesn't), with a reverse map
# keyed by (element, token): a token reverses only under the element it was protected with.
_VAULT = {
    ("datetime_yc", "TOKA"): "2020-08-06",
    ("string", "TOKB"): "Leila Rahman",
}


class ElementAwareProtector:
    """Reverses a token ONLY under the element it was 'protected' with, like the real API.

    unprotect under the wrong element returns the token unchanged (no round-trip), exactly the
    failure mode of unprotecting a datetime_yc token under the datetime element."""

    def unprotect_values(self, tokens, element):
        return [_VAULT.get((element, t), t) for t in tokens]


def _token(element: str, value: str) -> str:
    return {"datetime_yc": "TOKA", "string": "TOKB"}[element]


def test_tag_element_is_parsed():
    tagged = find_tokens("on [DATETIME|datetime_yc]TOKA[/DATETIME] a wire moved")
    assert tagged == [("DATETIME", "datetime_yc", "TOKA")]


def test_reidentify_reverses_under_override_element():
    # A date protected under datetime_yc, tagged with its element.
    token = _token("datetime_yc", "2020-08-06")
    text = f"The account opened on [DATETIME|datetime_yc]{token}[/DATETIME]."
    res = reidentify(text, ElementAwareProtector(), INVESTIGATOR)
    assert res.revealed == 1
    assert "2020-08-06" in res.text          # reversed correctly under datetime_yc
    assert token not in res.text


def test_wrong_element_without_the_tag_yields_garbage_not_the_date():
    # Same token but the tag omits the element -> falls back to the DEFAULT `datetime` element,
    # which does NOT match datetime_yc, so unprotect does not round-trip. This documents the BUG
    # the tag-element fixes: the true date "2020-08-06" is NOT recovered (the boundary shows a
    # wrong/garbage value), whereas the tag-carrying case above recovers it correctly.
    token = _token("datetime_yc", "2020-08-06")
    text = f"The account opened on [DATETIME]{token}[/DATETIME]."
    res = reidentify(text, ElementAwareProtector(), INVESTIGATOR)
    assert "2020-08-06" not in res.text      # the real date is NOT shown without the element tag


def test_default_element_tag_still_round_trips():
    # A PERSON protected under the default `string` element, no override, legacy tag form.
    token = _token("string", "Leila Rahman")
    text = f"Look into [PERSON]{token}[/PERSON]."
    res = reidentify(text, ElementAwareProtector(), INVESTIGATOR)
    assert res.revealed == 1
    assert "Leila Rahman" in res.text
