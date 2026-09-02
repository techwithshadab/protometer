"""Ingestion, apply a protection scope to the corpus and cache the result.

One run produces one protected corpus for one [[Protection Scope]]. The evaluation needs
five such corpora, and every one is cached to disk so the demo and the eval never
re-enter the protect path. Ingestion is the only stage that calls the hosted API
in bulk.

Two protection paths, for two shapes of data:

  **Structured** (parties, transactions), fields are already identified, so discovery is
  unnecessary. Values go straight to the batched protect path, grouped by data element.
  This is cheap: one round-trip per batch, deduplicated.

  **Unstructured** (narratives), entities must be *found* before they can be protected.
  This is where Protegrity's local discovery service earns its place, and it is expensive:
  the SDK's `find_and_protect` costs one round-trip per detected entity per document. We
  therefore run discovery ourselves, collect every entity across the whole corpus, protect
  them in batches, and splice tokens back in, turning thousands of calls into dozens.

That splice is what makes narrative protection affordable, and it is only sound because
tokenization is deterministic: the same name protected in a batch yields the
same token it would have had protected individually.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# Local discovery service (docker compose, no API key required). Overridable so the stack can
# run on another host or port without editing source, the hardcoded default was the only
# endpoint in the project with no environment escape hatch, unlike `OLLAMA_URL`.
# Resolved from the central settings surface (protometer.settings), so an endpoint or the
# discovery score threshold can be tuned via env in one documented place. Read at import for
# the module-level constants the codebase already references; the values still honour env.
from protometer import settings as _settings  # noqa: E402
from protometer.protect import ProtectionError, Protector
from protometer.roster import Roster, roster_from_narratives, roster_from_parties
from protometer.scopes import ProtectionScope

DISCOVERY_URL = _settings.discovery_url()
DISCOVERY_THRESHOLD = _settings.discovery_threshold()

# Substituted when a value cannot be tokenized, either the data element rejects its format,
# or the API returns it unchanged. Redaction loses utility, but emitting plaintext under a
# protection claim is a correctness failure, not a trade-off.
REDACTED = "[REDACTED]"

# Elements that reject values carrying their conventional separators, with the characters to
# strip before protecting. Measured: `ccn` accepts `4111111111111111` but rejects
# both `4111 1111 1111 1111` and `4111-1111-1111-1111` with error 44, and card numbers are
# written with separators far more often than without.
#
# Normalising is preferable to redacting: the value is genuinely protectable, just not in the
# shape it arrived in. Separators are restored when splicing so the surrounding text keeps its
# original shape.
SEPARATOR_SENSITIVE_ELEMENTS: dict[str, str] = {"ccn": " -"}


def normalise_for_element(value: str, element: str) -> str:
    """Strip separators an element rejects, leaving other elements untouched."""
    separators = SEPARATOR_SENSITIVE_ELEMENTS.get(element)
    if not separators:
        return value
    return "".join(ch for ch in value if ch not in separators)


class DiscoveryError(RuntimeError):
    """Raised when the discovery service or one of its providers fails.

    Treated as fatal rather than degraded: a failed provider means an entire class of
    entities goes undetected, and undetected means unprotected. Ingestion must stop rather
    than quietly produce a corpus that only looks protected.
    """

# Entity type -> Protegrity data element. Mirrors the SDK's DATA_ELEMENT_MAPPING; kept
# explicit here because a silent miss means an entity passes through *unprotected*, which
# is a correctness failure the pipeline must never make quietly.
ENTITY_TO_ELEMENT: dict[str, str] = {
    "PERSON": "string",
    "ORGANIZATION": "string",
    "ACCOUNT_NAME": "string",
    "ACCOUNT_NUMBER": "number",
    "BANK_ACCOUNT": "number",
    "SOCIAL_SECURITY_ID": "ssn",
    "NATIONAL_ID": "nin",
    "TAX_ID": "number",
    "PASSPORT": "passport",
    "DRIVER_LICENSE": "number",
    # `string`, not `email`. The `email` element is format-preserving in a way that defeats
    # the protection: it tokenizes the local part and returns the domain verbatim, so
    # `accounts@sablefield-group.com` became `VUOQhl23@sablefield-group.com`, all 257
    # organizations in the corpus kept their real name stem in the domain of their supposedly
    # protected address, and the domain alone narrows an org to a handful of candidates.
    #
    # Worse, determinism made the local part a public flag: every organization shares the
    # plaintext local part `accounts`, so every one tokenized to the identical `VUOQhl23`.
    #
    # `string` tokenizes the whole value and still returns an email-shaped result
    # (`68WpOPbA@bZwi4u5ZUX-vOfSO.ggo`), so downstream format expectations hold while the
    # organization is no longer recoverable. Verified against the live API, not assumed.
    "EMAIL_ADDRESS": "string",
    "PHONE_NUMBER": "phone",
    "CREDIT_CARD": "ccn",
    "CRYPTO_ADDRESS": "address",
    "USERNAME": "string",
    "USER_NAME": "string",
    "ADDRESS": "address",
    "LOCATION": "address",
    "CITY": "city",
    "DATETIME": "datetime",
    "DOB": "datetime",
    "DATE_OF_BIRTH": "datetime",
    "AMOUNT": "number",
    "AGE": "number",
    "GENDER": "string",
    "TITLE": "string",
    "IP_ADDRESS": "address",
    "MAC_ADDRESS": "address",
    "URL": "address",
    "HEALTH_CARE_ID": "number",
    "CURRENCY": "string",
    "CURRENCY_CODE": "string",
    "CURRENCY_NAME": "string",
    "CURRENCY_SYMBOL": "string",
    "PASSWORD": "string",
    "NRP": "number",
    "ORGANIZATION_NAME": "string",
    # Country-specific national identifiers. Absent from this US-domiciled corpus, but
    # detectable by the discovery service, so leaving them unmapped would mean a real
    # deployment's Indian, Korean or Thai identifiers were *detected and then skipped*,
    # passing through in the clear. Mapped to `string`, which accepts any shape, rather than
    # to a typed element that would reject their formats.
    "IN_GSTIN": "string",
    # Aligned to the SDK's own `DATA_ELEMENT_MAPPING`
    # (`protegrity_developer_python.utils.constants`). These four were mapped to `string`,
    # which protects the value but discards the format the vendor's mapping preserves, a
    # national ID tokenized as an opaque string no longer round-trips through systems that
    # validate its shape. Diffed against the official constant rather than assumed.
    "IN_VEHICLE_REGISTRATION": "number",
    "IN_VOTER": "number",
    "KR_RRN": "number",
    "TH_TNIN": "nin",
}


@dataclass
class IngestionReport:
    scope_name: str
    parties: int = 0
    transactions: int = 0
    narratives: int = 0
    entities_found: int = 0
    entities_protected: int = 0
    entities_skipped: dict[str, int] = field(default_factory=dict)
    # Attribution between the discovery model and the roster fallback. Reported because the
    # split is itself a measurement of the discovery service's coverage.
    entities_by_source: dict[str, int] = field(default_factory=dict)
    # Values the API refused outright, by data element.
    protection_failures: dict[str, int] = field(default_factory=dict)
    # Values the API returned unchanged while reporting success, silent non-protection.
    protection_noops: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    # Where the wall-clock went: hosted protect calls vs local discovery calls vs the rest
    # (roster matching, splicing, serialisation). Only the first was previously attributed.
    seconds_in_discovery: float = 0.0
    protection_stats: dict | None = None  # structured; see ProtectionStats.to_dict
    # Fingerprint of the CLEAR corpus this protected ledger was derived from. The eval read
    # path checks it against the current corpus so a regenerated corpus + stale data/protected
    # is caught, a failure class documented as having happened once.
    source_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "scope": self.scope_name,
            "parties": self.parties,
            "transactions": self.transactions,
            "narratives": self.narratives,
            "entities_found": self.entities_found,
            "entities_protected": self.entities_protected,
            "entities_skipped": self.entities_skipped,
            "entities_by_source": self.entities_by_source,
            "protection_failures": self.protection_failures,
            "protection_noops": self.protection_noops,
            "seconds": round(self.seconds, 1),
            "seconds_in_discovery": round(self.seconds_in_discovery, 2),
            "source_fingerprint": self.source_fingerprint,
            "protection_stats": self.protection_stats,
        }


# Wall-clock spent inside the discovery service this run. Module-level because
# `discover_entities` is called deep in the detection path with no report handle; ingestion
# resets it at the start and reads it at the end. Justified by measurement: `direct` ingestion
# took 394s of which `seconds_in_api` (Protegrity) explained 13.8s, the other 96% was 752
# sequential discovery round-trips that no counter attributed. Lock-guarded: the discovery
# loop may run under a thread pool.
_DISCOVERY_SECONDS = [0.0]
_DISCOVERY_LOCK = threading.Lock()
# One session for connection reuse: 752 bare posts per scope paid a TCP+TLS setup each.
_DISCOVERY_SESSION = requests.Session()


def discover_entities(text: str, timeout: float = 30.0, retries: int = 3) -> list[dict]:
    """Classify one document via the local discovery service.

    Returns entities with character offsets. Offsets are what make batched splicing
    possible, we replace by position rather than by string matching, so a name that also
    appears inside another word is not corrupted.

    Retried on transport errors: discovery is 97% of ingest wall-clock across ~750
    sequential calls, and a single sidecar hiccup at call 700 used to abort the scope and
    re-pay everything. The hosted-protect path retries six times; the path that costs the
    most retried zero.
    """
    for attempt in range(1, retries + 1):
        started = time.monotonic()
        try:
            response = _DISCOVERY_SESSION.post(
                DISCOVERY_URL,
                params={"score_threshold": DISCOVERY_THRESHOLD},
                headers={"Content-Type": "text/plain"},
                data=text.encode("utf-8"),
                timeout=timeout,
            )
            with _DISCOVERY_LOCK:
                _DISCOVERY_SECONDS[0] += time.monotonic() - started
            if response.status_code >= 500 and attempt < retries:
                raise requests.HTTPError(f"discovery returned {response.status_code}")
            response.raise_for_status()
            break
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            with _DISCOVERY_LOCK:
                _DISCOVERY_SECONDS[0] += time.monotonic() - started
            if attempt == retries:
                raise DiscoveryError(
                    f"discovery failed after {retries} attempts: {exc}"
                ) from exc
            time.sleep(0.5 * attempt)
    payload = response.json()

    # A provider can fail while the request still returns 200, the failure is reported
    # per-provider inside the body. Silently accepting that would mean whole entity classes
    # (names, in the case of the Context provider) passing through unprotected while
    # ingestion reports success, so provider errors are raised rather than ignored.
    for provider in payload.get("providers", []):
        if provider.get("status") != 200:
            raise DiscoveryError(
                f"Discovery provider {provider.get('config_provider', {}).get('name', '?')!r} "
                f"failed: {provider.get('exception', 'unknown error')}"
            )

    # `classifications` maps entity type -> list of occurrences, each with its own offsets.
    entities: list[dict] = []
    for entity_type, occurrences in (payload.get("classifications") or {}).items():
        for occurrence in occurrences:
            location = occurrence.get("location") or {}
            start, end = location.get("start_index"), location.get("end_index")
            if start is None or end is None:
                continue
            start, end = _expand_span(text, int(start), int(end))
            entities.append(
                {
                    "entity_type": entity_type,
                    "start": start,
                    "end": end,
                    "score": occurrence.get("score"),
                    "text": text[start:end],
                }
            )

    # Overlapping spans would corrupt the splice. Keep the highest-scoring span per region.
    entities.sort(key=lambda e: (e["start"], -(e["score"] or 0)))
    deduped: list[dict] = []
    for entity in entities:
        if deduped and entity["start"] < deduped[-1]["end"]:
            continue
        deduped.append(entity)
    return deduped


def _splice(text: str, replacements: list[tuple[int, int, str]]) -> str:
    """Apply (start, end, replacement) edits right-to-left so earlier offsets stay valid.

    Overlapping edits are dropped rather than applied: the roster may match a span that
    partially overlaps an unprotected discovery span, and applying both would interleave a
    token into the middle of another replacement and corrupt the text. The widest edit wins,
    since it covers the most sensitive material.
    """
    ordered = sorted(replacements, key=lambda r: (r[0], -(r[1] - r[0])))
    kept: list[tuple[int, int, str]] = []
    for start, end, replacement in ordered:
        if kept and start < kept[-1][1]:
            continue
        kept.append((start, end, replacement))

    out = text
    for start, end, replacement in sorted(kept, key=lambda r: r[0], reverse=True):
        out = out[:start] + replacement + out[end:]
    return out


def protect_batch_audited(
    protector: Protector,
    values: list[str],
    element: str,
    report: "IngestionReport | None" = None,
) -> list[str]:
    """Protect one element's values with the full leak-prevention audit, in one place.

    This is the single seam both the structured and the narrative path go through, because
    they must not differ. The audit is two guards, each pinned to a measured failure:

    1. **Batch-failure fallback.** `protect_values` raises on the first value an element
       rejects, failing the whole column; one malformed `datetime` would discard a batch of
       protectable ISO dates. On `ProtectionError` we fall back to per-value protection and
       redact only the values the element genuinely rejects.
    2. **No-op redaction.** `session.protect` can return the input *unchanged* with
       a success code, the most dangerous failure shape here: the plaintext survives under a
       protection claim. Every returned token is checked against its input and redacted if
       unchanged.

    A structured path that skipped these (it did, ingest.py before this seam existed) wrote
    unprotected amounts, account numbers, tax ids and DOBs verbatim into the canonical corpus.
    Counters are recorded on `report` when supplied, so the manifest reflects the real state.
    """
    try:
        tokens = protector.protect_values(values, element)
    except ProtectionError:
        tokens = []
        for value in values:
            try:
                tokens.append(protector.protect_value(value, element))
            except ProtectionError:
                tokens.append(REDACTED)
                if report is not None:
                    report.protection_failures[element] = (
                        report.protection_failures.get(element, 0) + 1
                    )
    audited: list[str] = []
    for value, token in zip(values, tokens):
        if Protector.is_noop(value, token):
            token = REDACTED
            if report is not None:
                report.protection_noops[element] = (
                    report.protection_noops.get(element, 0) + 1
                )
        audited.append(token)
    return audited


def protect_structured(
    records: list[dict],
    field_entity_map: dict[str, str],
    scope: ProtectionScope,
    protector: Protector,
    report: "IngestionReport | None" = None,
) -> list[dict]:
    """Protect known fields across many records, batching per data element.

    Records are protected column-wise rather than row-wise: every value of one field across
    all records goes in one batch. That is what keeps the call count proportional to the
    number of *fields* rather than the number of records.

    Every batch goes through `protect_batch_audited`, so a value the API returns unchanged
    (the no-op leak) or a column with one malformed value cannot write plaintext into the
    protected corpus, the same guarantee the narrative path has always had.
    """
    protected = [dict(record) for record in records]

    by_element: dict[str, list[str]] = {}
    positions: dict[str, list[tuple[int, str]]] = {}

    for field_name, entity_type in field_entity_map.items():
        if not scope.protects(entity_type):
            continue
        element = scope.element_for(entity_type, ENTITY_TO_ELEMENT.get(entity_type))
        if element is None:
            continue
        for i, record in enumerate(records):
            value = record.get(field_name)
            if not value:
                continue
            by_element.setdefault(element, []).append(
                normalise_for_element(str(value), element)
            )
            positions.setdefault(element, []).append((i, field_name))

    for element, values in by_element.items():
        tokens = protect_batch_audited(protector, values, element, report)
        for (index, field_name), token in zip(positions[element], tokens):
            protected[index][field_name] = token

    return protected


# Characters that may sit inside a single identifier (phone, SSN, account, email, card).
# Used to expand detector spans outward to the full token.
_IDENTIFIER_CHARS = "0123456789-.@_+"


def _expand_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a detected span to cover the whole identifier it sits inside.

    The discovery service returns truncated spans for some entity types, it detects
    `-555-7023` within `312-555-7023`, omitting the area code. Replacing only the detected
    span leaves the prefix in the clear, producing output like `312[PHONE]-684-6128[/PHONE]`
    where part of the real number survives.

    Expansion walks outward while the neighbouring character could belong to the same
    identifier, stopping at whitespace, letters or punctuation that ends it.
    """
    while start > 0 and text[start - 1] in _IDENTIFIER_CHARS:
        start -= 1
    while end < len(text) and text[end] in _IDENTIFIER_CHARS:
        end += 1
    # Never let the span end on punctuation that is really sentence/bracket structure, not
    # part of the identifier. The discovery service sometimes returns an end index that
    # already includes a trailing `)` (e.g. "account 7626876226)"), which then tokenized the
    # paren into the value and left `...4135027914)` inside the tag on the demo's own artifact.
    # Trim any trailing separator or closing bracket; identifiers never legitimately end on one.
    while end > start and text[end - 1] in ".-)]},;:":
        end -= 1
    return start, end


def detect_entities(
    text: str, roster: Roster | None = None, scope: ProtectionScope | None = None
) -> list[dict]:
    """Full detection: the discovery service, then the roster over what it left behind.

    Discovery runs first and its spans take precedence, it is the statistical detector and
    handles PERSON, SSN, EMAIL and PHONE well. The roster then fills the gap discovery
    cannot cover (ORGANIZATION, measured at zero across all thresholds).

    Only spans that will *actually be protected* block the roster. Without that condition a
    discovery hit the current scope ignores still claims its span: `Sablefield` detected as
    an out-of-scope LOCATION would block the roster from matching
    `Sablefield Advisory Services` as an in-scope ORGANIZATION, leaving the organization
    name in the clear.
    """
    entities = discover_entities(text)
    if roster is None:
        return entities

    blocking = [
        (e["start"], e["end"])
        for e in entities
        if scope is None or scope.protects(e["entity_type"])
    ]
    entities.extend(match.as_entity() for match in roster.find(text, blocking))
    entities.sort(key=lambda e: e["start"])
    return entities


def protect_narratives(
    narratives: list[dict],
    scope: ProtectionScope,
    protector: Protector,
    report: IngestionReport,
    roster: Roster | None = None,
) -> list[dict]:
    """Discover entities across all narratives, protect them in bulk, splice tokens back.

    The SDK's `find_and_protect` would issue one round-trip per entity per document. Doing
    discovery first and batching the protect step collapses that to a handful of calls,
    which is the difference between an overnight run and a few minutes.
    """
    # Discovery is embarrassingly parallel (a stateless local sidecar) and dominates the
    # wall-clock. `executor.map` preserves input order, so per_document is byte-identical
    # to the sequential result; only elapsed time changes. Workers=1 restores the
    # sequential path exactly.
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, int(os.getenv("PROTOMETER_DISCOVERY_WORKERS", "4")))
    if workers == 1:
        per_document = [
            detect_entities(n["text"], roster, scope) for n in narratives
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            per_document = list(
                pool.map(lambda n: detect_entities(n["text"], roster, scope), narratives)
            )
    for entities in per_document:
        report.entities_found += len(entities)
        for entity in entities:
            source = entity.get("source", "discovery")
            report.entities_by_source[source] = report.entities_by_source.get(source, 0) + 1

    # Group every entity occurrence in the corpus by the data element that protects it.
    #
    # Occurrences are tracked **by position**, not by value. Keying a lookup table on the
    # plaintext would silently restore determinism under the ablation: the same name
    # occurring five times yields five distinct tokens, but each write to a value-keyed table
    # overwrites the last, and every occurrence would then splice in the one surviving token.
    # Entity resolution would appear intact while the tokens were in fact unstable.
    to_protect: dict[str, list[str]] = {}
    occurrence_slots: dict[str, list[tuple[int, int]]] = {}

    for doc_index, entities in enumerate(per_document):
        for entity_index, entity in enumerate(entities):
            entity_type = entity["entity_type"]
            if not scope.protects(entity_type):
                continue
            element = scope.element_for(entity_type, ENTITY_TO_ELEMENT.get(entity_type))
            if element is None:
                report.entities_skipped[entity_type] = (
                    report.entities_skipped.get(entity_type, 0) + 1
                )
                continue
            to_protect.setdefault(element, []).append(
                normalise_for_element(entity["text"], element)
            )
            occurrence_slots.setdefault(element, []).append((doc_index, entity_index))

    # One batched call set per data element; results map back to the exact occurrence that
    # produced them.
    token_for: dict[tuple[int, int], str] = {}
    for element, values in to_protect.items():
        # Same audited seam as the structured path: batch-failure fallback + no-op redaction,
        # with failure/no-op counters recorded on the report.
        tokens = protect_batch_audited(protector, values, element, report)
        for slot, token in zip(occurrence_slots[element], tokens):
            token_for[slot] = token
            report.entities_protected += 1

    protected: list[dict] = []
    for doc_index, (narrative, entities) in enumerate(zip(narratives, per_document)):
        edits: list[tuple[int, int, str]] = []
        for entity_index, entity in enumerate(entities):
            entity_type = entity["entity_type"]
            if not scope.protects(entity_type):
                continue
            if ENTITY_TO_ELEMENT.get(entity_type) is None:
                continue
            token = token_for.get((doc_index, entity_index))
            if token is None:
                continue
            # Tag carries the RESOLVED element (respecting the scope's element_overrides) so
            # re-identification reverses the token under the SAME element it was protected with.
            # Omit the `|element` suffix when it equals the default, to keep default-scope corpora
            # byte-identical to the historical `[TYPE]token[/TYPE]` form.
            element = scope.element_for(entity_type, ENTITY_TO_ELEMENT.get(entity_type))
            default_element = ENTITY_TO_ELEMENT.get(entity_type)
            tag_open = (f"[{entity_type}|{element}]" if element and element != default_element
                        else f"[{entity_type}]")
            edits.append((entity["start"], entity["end"],
                          f"{tag_open}{token}[/{entity_type}]"))

        # Copy only the fields a protected corpus may carry.
        #
        # `dict(narrative)` copied *everything*, including the clear-corpus keys
        # `plaintext_entities` and `narrative_values`, so every name the splice had just
        # removed from `text` survived one key over, in the same file, beside its own token.
        # A ready-made token-to-plaintext mapping table shipped inside the artifact whose
        # entire claim is that it contains no real identifiers.
        #
        # Ground truth for the adversarial evaluation lives in the clear corpus, which those
        # scripts already read; it has no business here.
        record = {
            key: value
            for key, value in narrative.items()
            if key not in ("plaintext_entities", "narrative_values")
        }
        record["text"] = _splice(narrative["text"], edits)
        record["entity_count"] = len(edits)
        protected.append(record)

    return protected


# Field -> entity type for the structured records, matching Party.sensitive_fields.
PARTY_FIELDS: dict[str, str] = {
    "full_name": "PERSON",
    "account_number": "ACCOUNT_NUMBER",
    "bank_account": "BANK_ACCOUNT",
    "address": "ADDRESS",
    "city": "LOCATION",
    "email": "EMAIL_ADDRESS",
    "phone": "PHONE_NUMBER",
    "ssn": "SOCIAL_SECURITY_ID",
    "date_of_birth": "DATE_OF_BIRTH",
    "tax_id": "TAX_ID",
    "credit_card": "CREDIT_CARD",
}

# Transactions carry no names, only amounts and dates, which are quasi-identifiers. This
# is deliberate: it is what makes the quasi scope visibly break aggregation checkpoints.
TRANSACTION_FIELDS: dict[str, str] = {
    "amount": "AMOUNT",
    "value_date": "DATETIME",
}


def corpus_source_fingerprint(corpus_dir: Path) -> str:
    """Hash of the clear corpus files a protected ledger derives from.

    Deliberately the same five files and truncation as the eval runner's corpus fingerprint,
    so a protected ledger and an eval run agree on what "this corpus" means.
    """
    import hashlib

    digest = hashlib.sha256()
    for name in ("transactions.json", "narratives.json", "alerts.json",
                 "ground_truth.json", "parties.json"):
        try:
            digest.update((corpus_dir / name).read_bytes())
        except OSError:
            return "unknown"
    return digest.hexdigest()[:12]


def ingest(
    corpus_dir: Path,
    out_dir: Path,
    scope: ProtectionScope,
    protector: Protector | None = None,
    domain: "Any | None" = None,
) -> IngestionReport:
    """Protect the whole corpus under one scope and write it to `out_dir`.

    `domain` (a `domains.Domain`) supplies the field->entity map for structured protection;
    omitted, the AML `PARTY_FIELDS`/`TRANSACTION_FIELDS` are used. A non-AML domain's
    `record_fields` is applied to both structured record types (its schema is one map), so its
    patient/customer identifiers are actually tokenized rather than passed through under AML
    field names. Without threading this, `Domain.record_fields` was inert on the batch path.
    """
    started = time.monotonic()
    _DISCOVERY_SECONDS[0] = 0.0  # per-run attribution; see the accumulator's comment
    report = IngestionReport(scope_name=scope.name)
    report.source_fingerprint = corpus_source_fingerprint(corpus_dir)

    parties = json.loads((corpus_dir / "parties.json").read_text())
    transactions = json.loads((corpus_dir / "transactions.json").read_text())
    narratives = json.loads((corpus_dir / "narratives.json").read_text())

    # The baseline scope protects nothing, so it needs no client and no network at all.
    if not scope.entities:
        protected_parties, protected_transactions, protected_narratives = (
            parties,
            transactions,
            narratives,
        )
    else:
        if protector is None:
            protector = Protector(rotate_iv=scope.break_determinism)
        # Roster is built from the *clear* party list, before protection is applied, so it
        # matches the plaintext names still present in the narratives. Narrative-declared
        # values (prose dates, written amounts) are folded in for the same reason
        # organizations were: the detector is blind to them.
        roster = roster_from_parties(parties, roster_from_narratives(narratives))
        # A domain supplies its own field->entity map; AML's split defaults preserve behaviour.
        party_fields = dict(domain.record_fields) if domain is not None else PARTY_FIELDS
        txn_fields = dict(domain.record_fields) if domain is not None else TRANSACTION_FIELDS
        protected_parties = protect_structured(
            parties, party_fields, scope, protector, report
        )
        protected_transactions = protect_structured(
            transactions, txn_fields, scope, protector, report
        )
        protected_narratives = protect_narratives(narratives, scope, protector, report, roster)
        report.protection_stats = protector.stats.to_dict()
    report.seconds_in_discovery = _DISCOVERY_SECONDS[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    from protometer.persist import atomic_write_json

    atomic_write_json(out_dir / "parties.json", protected_parties)
    atomic_write_json(out_dir / "transactions.json", protected_transactions)
    atomic_write_json(out_dir / "narratives.json", protected_narratives)

    report.parties = len(protected_parties)
    report.transactions = len(protected_transactions)
    report.narratives = len(protected_narratives)
    report.seconds = time.monotonic() - started

    atomic_write_json(out_dir / "ingestion_report.json", report.to_dict())
    return report
