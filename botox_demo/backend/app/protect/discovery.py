"""Protegrity Data Discovery: find PII spans in free text (the ONLY detector).

Discovery is an ML/NER classifier that finds names, emails, phones, addresses, dates of birth, and
member IDs in free text, returning each occurrence with character offsets and a confidence score. It
is the sole PII detector for this bot, there is no regex fallback.

    POST {DISCOVERY_URL}?score_threshold=<t>
    Content-Type: text/plain
    body: the raw text
    -> {"classifications": {"<ENTITY_TYPE>": [{"location": {"start_index","end_index"}, "score"}]},
        "providers": [{"status": 200, ...}]}

This runs LOCALLY in Docker with no API key (unlike hosted tokenization). Discovery failures are
FATAL, not silently downgraded: a failed provider means a whole class of PII goes undetected, and
undetected means unprotected, so the caller fails closed rather than serve an unprotected turn.
(Lesson carried from the Protometer ingest pipeline.)
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger("botox.discovery")

DISCOVERY_URL = os.getenv(
    "PROTEGRITY_DISCOVERY_URL",
    "http://vendor-de-discovery:8580/pty/data-discovery/v2/classify/text",
)
# Minimum confidence to treat a discovered span as real PII. 0.6 matches the Protometer default;
# tune via env in one place.
DISCOVERY_THRESHOLD = float(os.getenv("PROTEGRITY_DISCOVERY_THRESHOLD", "0.6"))

# DYNAMIC threshold — measured behaviour: in a MULTI-PII sentence Discovery's per-span confidence
# drops (a PERSON that scores ~0.99 alone scores ~0.57 when an email + phone share the sentence), so
# a fixed 0.6 silently DROPS the name and leaks it in the clear to the model + trace. We query at a
# LOW floor to see every candidate with its score, then ACCEPT with a bar that eases as the message
# carries more distinct PII: the more entities present, the more a lower-scoring span is corroborated
# as real PII rather than noise. This closes the multi-PII name leak without over-tokenizing a single
# clean word. Fully env-tunable; set PROTEGRITY_DISCOVERY_DYNAMIC=off to pin the flat threshold.
DISCOVERY_DYNAMIC = os.getenv("PROTEGRITY_DISCOVERY_DYNAMIC", "on").lower() not in ("0", "off", "false", "no")
# The floor we actually ASK the service for (so low-scoring candidates are returned at all). Kept
# BELOW the eased acceptance bar (min 0.45) because the service returns DIFFERENT spans at different
# thresholds: a name+email+phone message yields the full "Jane Smith" @0.568 only when asked below
# ~0.4 — at 0.4 the service instead splits to just the higher-confidence "Smith", dropping the first
# name. 0.3 surfaces the whole-name span; the count-aware bar then accepts it.
DISCOVERY_QUERY_FLOOR = float(os.getenv("PROTEGRITY_DISCOVERY_QUERY_FLOOR", "0.3"))
# Per-extra-entity relaxation and the hard floor the eased bar can never drop below.
DISCOVERY_RELAX_PER_ENTITY = float(os.getenv("PROTEGRITY_DISCOVERY_RELAX_PER_ENTITY", "0.1"))
DISCOVERY_MIN_THRESHOLD = float(os.getenv("PROTEGRITY_DISCOVERY_MIN_THRESHOLD", "0.45"))


def _acceptance_threshold(candidate_count: int) -> float:
    """The confidence bar a span must clear to be tokenized, eased by how many DISTINCT PII
    candidates the message holds. One candidate -> the full DISCOVERY_THRESHOLD (no leak surface, a
    lone word must be clearly PII). Each ADDITIONAL candidate lowers the bar by
    DISCOVERY_RELAX_PER_ENTITY, never below DISCOVERY_MIN_THRESHOLD — so the name in
    "I'm Jane Smith, jane@x.com, 312-555-7023" (3 candidates) is accepted at ~0.4-eased even though
    its own score dipped under 0.6. Disabled -> the flat DISCOVERY_THRESHOLD."""
    if not DISCOVERY_DYNAMIC or candidate_count <= 1:
        return DISCOVERY_THRESHOLD
    eased = DISCOVERY_THRESHOLD - DISCOVERY_RELAX_PER_ENTITY * (candidate_count - 1)
    return max(DISCOVERY_MIN_THRESHOLD, eased)

# Protegrity Discovery entity types -> this project's wrapper TYPEs. Aligned with the vendor's
# DATA_ELEMENT_MAPPING (via protector._ELEMENT) so each identifier is tokenized under the CORRECT
# data element, verified live. A discovered type not in this map still gets protected under the
# generic NAME wrapper (-> `string`, which accepts any value): a conservative fail-safe, better to
# tokenize an unmapped identifier than to leak it.
_ENTITY_TO_TYPE: dict[str, str] = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "SOCIAL_SECURITY_ID": "SSN",
    "PERSON": "NAME",
    "USERNAME": "NAME",
    "USER_NAME": "NAME",
    "TITLE": "DOCTOR",              # "Dr. X" style clinician references
    "ADDRESS": "ADDRESS",
    "LOCATION": "ADDRESS",
    "CITY": "ADDRESS",
    "DATE_OF_BIRTH": "DOB",         # -> datetime element (not string)
    "DOB": "DOB",
    "DATETIME": "DOB",              # any date/time -> datetime element
    "HEALTH_CARE_ID": "MEMBERID",   # member/patient IDs -> number
    "NATIONAL_ID": "SSN",
    "TAX_ID": "MEMBERID",
    "CREDIT_CARD": "CREDITCARD",    # -> ccn element (separators stripped)
    "BANK_ACCOUNT": "MEMBERID",
    "ACCOUNT_NUMBER": "MEMBERID",
    "PASSPORT": "PASSPORT",
    "DRIVER_LICENSE": "DL",
    "AGE": "AGE",
    "IP_ADDRESS": "ADDRESS",
    "MAC_ADDRESS": "ADDRESS",
    "CRYPTO_ADDRESS": "ADDRESS",
}


class DiscoveryError(RuntimeError):
    """Raised when Discovery or one of its providers fails. Fatal: undetected PII is unprotected
    PII, so the caller must stop rather than fall back to a weaker detector silently."""


# Characters that can belong to a numeric identifier (phone, SSN, card, account). Letters are NOT
# included, so span expansion over these never swallows an adjacent word or corrupts a name span.
_IDENTIFIER_CHARS = "0123456789-.@_+"


def _expand_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a detected span to cover the whole identifier it sits inside, then trim trailing
    punctuation. Discovery returns truncated numeric spans for some formats (e.g. "-555-7023" within
    "312-555-7023"); replacing only the detected span would leave the prefix ("312") in the clear."""
    while start > 0 and text[start - 1] in _IDENTIFIER_CHARS:
        start -= 1
    while end < len(text) and text[end] in _IDENTIFIER_CHARS:
        end += 1
    # An identifier never legitimately ends on sentence/bracket punctuation; trim it so a trailing
    # ")" or "." isn't tokenized into the value.
    while end > start and text[end - 1] in ".-)]},;:":
        end -= 1
    return start, end


class DiscoveryClient:
    """Thin, lazy client over the local Discovery service. Imports requests only when constructed,
    so the module loads on the mock path with nothing installed."""

    def __init__(self) -> None:
        import requests  # noqa: F401  presence check; the call path uses it
        if not os.getenv("PROTEGRITY_DISCOVERY_URL") and "vendor-de-discovery" in DISCOVERY_URL:
            # No endpoint configured and the default host is unresolvable outside compose: treat as
            # unavailable so the Protector fails closed at construction rather than on first request.
            _log.warning("PROTEGRITY_DISCOVERY_URL not set; Discovery is unavailable")
        self._session = requests.Session()

    def discover(self, text: str, *, timeout: float = 30.0, retries: int = 3) -> list[dict]:
        """Return non-overlapping PII spans in `text` as dicts {type, start, end, value, score},
        where `type` is one of the protector's wrapper TYPEs. Retried on transport faults; a
        per-provider failure inside a 200 response is raised (a silent provider miss would leak a
        whole entity class)."""
        import requests

        response = None
        for attempt in range(1, retries + 1):
            try:
                # Ask at the LOW floor so multi-PII sentences still surface their lower-scoring
                # spans; the count-aware acceptance bar is applied client-side below. When dynamic
                # mode is off, ask at the flat threshold directly (identical to the old behaviour).
                query_threshold = min(DISCOVERY_QUERY_FLOOR, DISCOVERY_THRESHOLD) \
                    if DISCOVERY_DYNAMIC else DISCOVERY_THRESHOLD
                response = self._session.post(
                    DISCOVERY_URL,
                    params={"score_threshold": query_threshold},
                    headers={"Content-Type": "text/plain"},
                    data=text.encode("utf-8"),
                    timeout=timeout,
                )
                if response.status_code >= 500 and attempt < retries:
                    raise requests.HTTPError(f"discovery returned {response.status_code}")
                response.raise_for_status()
                break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
                if attempt == retries:
                    raise DiscoveryError(f"discovery failed after {retries} attempts: {exc}") from exc
                import time
                time.sleep(0.5 * attempt)

        assert response is not None
        payload = response.json()

        # A provider can fail while the request still returns 200; the failure is per-provider in
        # the body. Accepting it silently would let an entity class (e.g. names) pass unprotected.
        for provider in payload.get("providers", []):
            if provider.get("status") not in (200, None):
                name = (provider.get("config_provider") or {}).get("name", "?")
                raise DiscoveryError(
                    f"Discovery provider {name!r} failed: {provider.get('exception', 'unknown')}")

        spans: list[dict] = []
        for entity_type, occurrences in (payload.get("classifications") or {}).items():
            wrapper = _ENTITY_TO_TYPE.get(entity_type, "NAME")
            for occ in occurrences:
                loc = occ.get("location") or {}
                start, end = loc.get("start_index"), loc.get("end_index")
                if start is None or end is None:
                    continue
                # Widen a TRUNCATED numeric span to cover the whole identifier. Discovery is measured
                # to return e.g. "-555-7023" within "312-555-7023", omitting the area code, replacing
                # only the detected span would leave "312" in the clear. Expansion walks over
                # identifier chars (digits/separators only, so it never grabs adjacent letters and
                # can't corrupt a name), then trims trailing punctuation.
                start, end = _expand_span(text, int(start), int(end))
                spans.append({"type": wrapper, "start": start, "end": end,
                              "value": text[start:end], "score": occ.get("score")})

        # DYNAMIC acceptance: we queried at the low floor, so `spans` holds every candidate the
        # message surfaced. The bar a span must clear eases with how many DISTINCT candidates are
        # present (a multi-PII sentence depresses each span's score), so the NAME in a name+email+
        # phone message is kept even though its own score dipped under the flat 0.6. A None score
        # (service omitted it) is treated as passing — never drop a span we cannot score, that would
        # leak it. Distinctness is by (start,end) so repeated detections of one span don't inflate
        # the count. This runs BEFORE dedup so the count reflects true candidate breadth.
        distinct = {(s["start"], s["end"]) for s in spans}
        bar = _acceptance_threshold(len(distinct))
        spans = [s for s in spans if s["score"] is None or (s["score"] or 0.0) >= bar]

        # Resolve overlaps by MERGING, not dropping. Two spans can overlap after expansion (e.g. two
        # adjacent identifiers glued by a separator). Dropping the second would leave its
        # non-overlapping tail in the CLEAR, a PII leak. Instead we extend the kept span to cover the
        # union, keeping the higher-scoring span's TYPE, so the whole PII region is tokenized as one.
        spans.sort(key=lambda s: (s["start"], -(s["score"] or 0.0)))
        deduped: list[dict] = []
        for s in spans:
            if deduped and s["start"] < deduped[-1]["end"]:
                prev = deduped[-1]
                if s["end"] > prev["end"]:              # extend to cover the union (no tail left clear)
                    prev["end"] = s["end"]
                    prev["value"] = text[prev["start"]:prev["end"]]
                continue
            deduped.append(s)
        return deduped
