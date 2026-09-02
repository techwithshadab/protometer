"""Semantic Guardrail, the egress check the pipeline was missing.

Everything else in this project protects data on the way **in**: entities are discovered and
tokenized before embedding, and the model reasons over tokens. That leaves the return path
unguarded, names it as the gap, a model can be induced to emit something it
should not, and nothing was inspecting what came back.

Protegrity ships the check. Semantic Guardrail runs locally in Docker with no API key and
scans a conversation in both directions:

  * **user → ai** with a semantic processor, scoring the prompt for manipulation;
  * **ai → user** with the `pii` processor, scoring the response for identifiers.

    POST http://localhost:6001/pty/semantic-guardrail/v1.1/conversations/messages/scan
    (shared protegrity-shared tier; in-container: pty-guardrail:8001)

Scores run 0..1 where higher is riskier, and the service returns per-message outcomes plus a
conversation-level verdict.

## What the service does, and what this module adds

Measured against the running service, the two halves behave differently and the difference
matters:

  * **PII in responses**, a leaked name and email score **0.9909** with character offsets,
    `['EMAIL_ADDRESS : [9, 25]', 'PERSON : [0, 8]']`. This works, and is used as the vendor
    intends.
  * **Surrogate keys are false positives.** A correct, fully-protected rationale -
    *"Party P02386 shows 179 cycles and 9 outbound counterparties, consistent with layering"* -
    is **rejected at 0.7202**, because `P02386` is classified `USER_NAME`.
  * **No available processor separates AML queries from attacks.** The service exposes three
    domain models (`GET /domain-models/`: `customer-support`, `financial`, `healthcare`) plus
    the `pii` entity scanner. Measured over five benign analyst questions and five injection
    attempts, `customer-support` scored benign **0.620-0.782** and malicious **0.615-0.759**,
    fully overlapping, rejecting all ten. The obviously-named `financial` model is no better:
    it rejects a benign layering-analysis question at **0.80** and an injection attempt at
    **0.76** (still overlapping, still inverted). There is no AML-analyst-tuned model, and the
    general-purpose ones read "value moved through a circuit of intermediaries" as adversarial.

Those last two are the gaps this module complements, and they are complemented differently
because they are different kinds of gap.

**Surrogate keys** are a knowledge gap: the service cannot know that `P02386` identifies a row
rather than a person, because that is a property of this schema. The service classifies and
this module adjudicates, findings resolving entirely to known surrogate-key shapes are
recorded and discounted rather than raised. A guardrail that rejects every correct answer is
one an operator switches off, which is worse than not deploying it.

**Injection scoring** is a capability gap, and the honest response is not to invent a
threshold that the measurement says does not exist. The prompt path therefore defaults to
**off** (`scan_prompts=False`) and reports the measurement rather than pretending to a control
this project cannot demonstrate. The docs already list prompt injection as unmitigated; this
narrows *why*, with numbers, instead of quietly claiming coverage.

The counterpart to the surrogate-key discount is the leak check: a finding that names a *real*
corpus value is escalated to a hard failure regardless of score, because a plaintext
identifier in a model response is the one outcome this architecture exists to prevent. That
check is this module's own, not the service's, and it is what makes the egress guard
load-bearing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# Local service, no API key. Overridable like every other endpoint in this project.
from protometer import settings as _settings

GUARDRAIL_URL = _settings.guardrail_url()

# One pooled HTTP session for the guardrail, so a serving path that scans every reply reuses
# the TCP/TLS connection instead of paying a fresh handshake per turn. `requests.Session` is
# thread-safe for `post`.
_GUARDRAIL_SESSION = requests.Session()

# The processor that scores a user message for manipulation. The vendor's samples use
# "customer-support"; it is a semantic risk model, not a domain-specific one.
INJECTION_PROCESSOR = "customer-support"
PII_PROCESSOR = "pii"

# Surrogate-key shapes this corpus emits and the guardrail misreads as identifiers. Party ids
# (`P02386`), transaction ids (`TXN001710`), alert ids (`ALERT0042`) and document ids
# (`DOC0007`) are opaque keys: they identify a row, not a person, and resolving one to a human
# requires the clear corpus, which is the boundary this architecture already controls.
SURROGATE_KEY_PATTERN = re.compile(r"\b(?:P\d{4,6}|TXN\d{4,8}|ALERT\d{3,6}|DOC\d{3,6})\b")

# Entity types worth failing on if they appear in a model response. Deliberately narrow:
# these are direct identifiers, not the quasi-identifiers whose disclosure this project
# measures rather than forbids.
LEAKABLE_ENTITIES = frozenset(
    {
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "SOCIAL_SECURITY_ID",
        "CREDIT_CARD",
        "ADDRESS",
        "BANK_ACCOUNT",
        "ACCOUNT_NUMBER",
        "TAX_ID",
        "ORGANIZATION",
    }
)


def _strip_ignorable(text: str) -> str:
    """Drop format (Cf) and non-spacing-mark (Mn) characters an adversary threads through a
    value to defeat matching while leaving it visually identical: zero-widths, joiners,
    soft hyphen, RTL overrides, variation selectors, combining glyphs. Category-based rather
    than a fixed five-codepoint table, which left most of the class open."""
    import unicodedata

    return "".join(
        ch for ch in text if unicodedata.category(ch) not in ("Cf", "Mn")
    )


def _normalize_for_match(text: str) -> str:
    """One canonical form for leak matching: ignorable-stripped, NFKC, casefolded.

    Order matters: strip ignorables BEFORE NFKC so a combining/format char cannot survive
    into the normalized form, and NFKC folds fullwidth/circled/mathematical digits to ASCII
    so a value rewritten in them is still caught."""
    import unicodedata

    return unicodedata.normalize("NFKC", _strip_ignorable(text)).casefold()


def forbidden_values_from_parties(
    parties: "list[dict]", fields: "tuple[str, ...] | None" = None
) -> frozenset[str]:
    """The clear-corpus values a model response must never contain.

    One construction, because it existed as two copy-pasted comprehensions (`run_hybrid.py`
    and `eval/runner.py`), the same duplicated-seam pattern that let the hybrid classifier
    drift from the training pipeline. The field set defaults to `ingest.PARTY_FIELDS` rather
    than a re-listed copy (which had already drifted, omitting `city`), so a field added to
    the AML protection map is covered here automatically.

    `fields` overrides that default so a **non-AML domain** seeds the egress check from ITS
    own schema, healthcare patient names/MRNs, support customer emails. Without this the
    backstop silently forbids only AML party fields and lets a healthcare/support corpus's
    identifiers through (a cross-domain leak). The length floor drops values too short to
    search for without false positives.
    """
    if fields is None:
        # Lazy import keeps guardrail free of ingest's heavy protect dependency at module load.
        from protometer.ingest import PARTY_FIELDS

        fields = tuple(PARTY_FIELDS)
    return frozenset(
        value
        for party in parties
        for field_name in fields
        if len(value := str(party.get(field_name) or "")) >= 5
    )


# The fields whose values are digit-only identifiers with no legitimate reason to appear as a
# substring of unrelated digits: an SSN, account, card or tax id embedded in a longer digit run
# is far more likely an evasion than a coincidence, so these are checked on ANY occurrence, not
# only on a digit boundary. Amounts/phones are deliberately excluded, they DO collide with
# unrelated decimals (a bank-account digit-run inside a SHAP contribution), the false-positive
# class the boundary rule exists to prevent.
def _high_sensitivity_fields() -> tuple[str, ...]:
    """The AML high-sensitivity field set, sourced from the DOMAIN so there is ONE list.

    This used to be a private constant duplicating `domains._AML.high_sensitivity_fields`; the two
    were hand-maintained in separate files and could silently drift (the `domain is None` path
    here seeds the eval/hybrid guard, the `domain=` path seeds the live UI guard — a drift would
    make measurement and deployment disagree on what a hard leak is). Deriving from the domain
    makes `get_domain('aml').high_sensitivity_fields` the single source of truth. Lazy import
    keeps guardrail importable without domains at module load."""
    from protometer.domains import get_domain

    return get_domain("aml").high_sensitivity_fields


def high_sensitivity_values_from_parties(parties: "list[dict]") -> frozenset[str]:
    """Digit-only high-value identifiers checked on any occurrence (fields from the AML domain)."""
    fields = _high_sensitivity_fields()
    return frozenset(
        value
        for party in parties
        for field_name in fields
        if len(value := str(party.get(field_name) or "")) >= 5
    )


class GuardrailUnavailable(RuntimeError):
    """Raised when the guardrail service cannot be reached.

    Deliberately fatal rather than degraded. A security control that silently disables itself
    when its backend is down provides the appearance of protection and none of the substance -
    the same failure mode as the empty index scoring 0.0, and as `is_noop` returning plaintext
    with a success code. Callers that genuinely want to run without it pass `enabled=False`,
    which is a visible decision.
    """


@dataclass
class Finding:
    """One processor's verdict on one message."""

    processor: str
    score: float
    explanation: str
    entities: tuple[str, ...] = ()
    # True when every entity found resolves to a surrogate key rather than a person.
    surrogate_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "processor": self.processor,
            "score": round(self.score, 4),
            "explanation": self.explanation,
            "entities": list(self.entities),
            "surrogate_only": self.surrogate_only,
        }


@dataclass
class ScanResult:
    """A scanned message: what the service said, and what we do about it."""

    outcome: str
    score: float
    findings: list[Finding] = field(default_factory=list)
    # Real corpus values found verbatim in the text, a hard failure, not a score.
    leaked_values: tuple[str, ...] = ()
    # Conversation-level verdict the service returns alongside the per-message one: the
    # `batch` object's outcome/score. We surfaced only the per-message score before and
    # discarded this; a serving UI wants the conversation-level risk as its own signal.
    batch_outcome: str = ""
    batch_score: float = 0.0

    @property
    def blocked(self) -> bool:
        """Whether this message should be withheld.

        A leak is always blocking. Otherwise the service's own verdict stands, *unless* every
        finding behind it is a surrogate key, the false-positive class documented above.
        """
        if self.leaked_values:
            return True
        if self.outcome != "rejected":
            return False
        return not (self.findings and all(f.surrogate_only for f in self.findings))

    @property
    def discounted(self) -> bool:
        """True when the service rejected but this module overrode it."""
        return self.outcome == "rejected" and not self.blocked

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "score": round(self.score, 4),
            "blocked": self.blocked,
            "discounted": self.discounted,
            "leaked_values": list(self.leaked_values),
            "findings": [f.to_dict() for f in self.findings],
        }


def _parse_entities(explanation: str) -> tuple[str, ...]:
    """Pull entity types out of the service's explanation string.

    The service returns them as a stringified list, `"['EMAIL_ADDRESS : [9, 25]',
    'PERSON : [0, 8]']"`, rather than structured data, so this parses defensively and
    returns nothing rather than raising on an unfamiliar shape.
    """
    return tuple(sorted(set(re.findall(r"\b([A-Z][A-Z_]{2,})\b", explanation or ""))))


@dataclass
class Guardrail:
    """Client for the local Semantic Guardrail service."""

    url: str = GUARDRAIL_URL
    timeout: float = 30.0
    enabled: bool = True
    # Prompt scanning is off by default. Measured, `customer-support` scores benign AML
    # queries 0.620-0.782 against injection attempts 0.615-0.759, overlapping ranges, all
    # rejected. Enabling it would block every analyst question while claiming a control the
    # measurement does not support. Set True to reproduce the measurement.
    scan_prompts: bool = False
    # Corpus values that must never appear in a model response. Supplied by the caller from
    # the clear corpus; empty means the leak check is skipped.
    forbidden_values: frozenset[str] = frozenset()
    # High-sensitivity digit-only identifiers (SSN/account/card/tax id) checked on ANY
    # occurrence, not only a digit boundary: for these shapes an embedded match is an evasion,
    # not a coincidence, and the false-positive cost is worth paying. A subset of the values in
    # `forbidden_values`.
    high_sensitivity_values: frozenset[str] = frozenset()
    # The Semantic-Guardrail domain models this instance scores with. Instance fields (not the
    # bare module constants) so a domain can select its own, e.g. the `healthcare` prompt model,
    # while the defaults reproduce the AML behaviour for every existing call site.
    injection_processor: str = INJECTION_PROCESSOR
    pii_processor: str = PII_PROCESSOR

    @classmethod
    def for_corpus(
        cls, parties_path: "str | Path", probe: bool = True,
        domain: "Any | None" = None, **kwargs: Any,
    ) -> "Guardrail":
        """The one construction path for a corpus-seeded egress guard.

        Loads the clear party records, seeds the forbidden-value check from them, and
        (by default) probes the sidecar so an unreachable service fails at construction
        rather than mid-queue. This existed as two copy-pasted bring-up blocks
        (`eval/runner.py` and `run_hybrid.py`), the same duplicated-seam pattern that let
        the hybrid classifier drift. Failure semantics stay with the caller: the analyst
        path lets `GuardrailUnavailable` propagate, the measurement harness catches it.

        `domain` (a `domains.Domain`) selects the guardrail's prompt-scoring model and the
        high-sensitivity field set; omitted, the AML defaults apply, so every existing caller
        is unchanged.
        """
        try:
            parties = json.loads(Path(parties_path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            # One exception type for "the guard cannot be constructed", whatever the cause.
            # A missing parties file used to escape as FileNotFoundError, bypassing every
            # caller's curated fail-closed message.
            raise GuardrailUnavailable(
                f"cannot seed forbidden values from {parties_path}: {exc}"
            ) from exc
        # Seed the forbidden-value backstop from the DOMAIN's schema, not always AML's, or a
        # healthcare/support corpus's identifiers are never forbidden (cross-domain leak).
        if domain is not None:
            forbidden = forbidden_values_from_parties(parties, tuple(domain.record_fields))
            kwargs.setdefault("injection_processor", domain.injection_processor)
            kwargs.setdefault("pii_processor", domain.pii_processor)
            high_sensitivity = domain.high_sensitivity_values(parties)
        else:
            forbidden = forbidden_values_from_parties(parties)
            high_sensitivity = high_sensitivity_values_from_parties(parties)
        guard = cls(
            forbidden_values=forbidden,
            high_sensitivity_values=high_sensitivity,
            **kwargs,
        )
        # Install the same clear values as trace-redaction: a run that builds a guardrail is a
        # run that emits generations, and at scope none/partial those bodies carry these exact
        # identifiers. Scrubbing them before export keeps clear PII out of Langfuse's at-rest
        # store. Best-effort, observability is optional and must never break guardrail setup.
        try:
            from protometer.observability import set_trace_redaction

            set_trace_redaction(forbidden)
        except Exception:  # noqa: BLE001
            pass
        if probe:
            guard.scan_response("readiness probe")
        return guard

    def _scan(self, content: str, processor: str, direction: tuple[str, str],
              extra_tokens: "frozenset[str] | None" = None) -> ScanResult:
        sender, recipient = direction
        payload = {
            "messages": [
                {
                    "from": sender,
                    "to": recipient,
                    "content": content,
                    "processors": [processor],
                }
            ]
        }
        try:
            response = _GUARDRAIL_SESSION.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001, transport and decode both mean "no verdict"
            raise GuardrailUnavailable(
                f"Semantic Guardrail unreachable at {self.url}: {exc}. "
                f"Start it with: cd vendor-de/semantic-guardrail && docker compose up -d"
            ) from exc

        messages = body.get("messages") or [{}]
        message = messages[0]
        findings: list[Finding] = []
        for entry in message.get("processors") or []:
            explanation = str(entry.get("explanation", ""))
            entities = _parse_entities(explanation)
            findings.append(
                Finding(
                    processor=str(entry.get("name", processor)),
                    score=float(entry.get("score", 0.0)),
                    explanation=explanation,
                    entities=entities,
                    surrogate_only=self._is_surrogate_only(
                        content, entities, explanation, extra_tokens=extra_tokens),
                )
            )

        batch = body.get("batch") or {}
        return ScanResult(
            outcome=str(message.get("outcome", "unknown")),
            score=float(message.get("score", 0.0)),
            findings=findings,
            leaked_values=self._leaked(content),
            batch_outcome=str(batch.get("outcome", "")),
            batch_score=float(batch.get("score", 0.0)),
        )

    def _is_surrogate_only(
        self, content: str, entities: tuple[str, ...], explanation: str = "",
        extra_tokens: "frozenset[str] | None" = None,
    ) -> bool:
        """True when every flagged **span** is itself a surrogate key.

        The earlier form checked whole-message containment: any `P#####` anywhere in the text
        plus a finding typed outside LEAKABLE_ENTITIES discounted the whole verdict, so a
        real name the service misclassified as `USER_NAME` (its habitual label for names)
        passed whenever a party id appeared in the same rationale. The service reports
        character offsets in its explanation; checking the flagged substrings themselves
        closes that hole. If the offsets cannot be parsed, the answer is "not surrogate-only":
        an unparseable justification must fail closed, not open.

        The verdict is decided by the SPANS, not the entity TYPES. An earlier gate here refused to
        discount whenever any flagged type was in LEAKABLE_ENTITIES (PERSON/SOCIAL_SECURITY_ID/…).
        But the service labels a *protection token* by the type of the value it replaced — a
        tokenized SSN is still reported `SOCIAL_SECURITY_ID` — so that gate blocked every reply
        written over tokens, precisely the safe case the discount exists for. The per-span loop
        below is authoritative (every span must be a surrogate key or a verified protection token),
        and the hard `leaked_values` block (consulted in `blocked` BEFORE this) still catches any
        real clear corpus value, so judging by spans cannot let a genuine leak through.
        """
        if not entities:
            return False

        spans = re.findall(r"\[(\d+),\s*(\d+)\]", explanation or "")
        if not spans:
            return False  # cannot verify what was flagged -> do not discount
        for start_s, end_s in spans:
            start, end = int(start_s), int(end_s)
            flagged = content[start:end].strip()
            # Discount a flagged span that is EITHER a surrogate-key shape (P####/TXN####) OR a
            # protection token the pipeline emitted. The serving path surfaced the second case:
            # the service mislabels a Protegrity token like `4oB93 T7MdI3` as PASSWORD and
            # rejects a safe reply. A protection token is safe by construction, it is not a
            # real value. This does NOT weaken the leak check: `leaked_values` (checked before
            # `blocked` even consults this) is the authoritative hard block for any real clear
            # corpus value, so broadening the discount to protection tokens cannot pass a leak.
            if SURROGATE_KEY_PATTERN.fullmatch(flagged):
                continue
            if self._is_protection_token(flagged, extra_tokens=extra_tokens):
                continue
            return False
        return True

    def _is_protection_token(self, span: str,
                             extra_tokens: "frozenset[str] | None" = None) -> bool:
        """True when `span` is one of the pipeline's own protection tokens (not a real value).

        Authoritative, not heuristic: a token is safe iff it appears in the protected corpus's
        token set, which is loaded once from the protected artifacts. Falls back to a
        conservative NO (do not discount) if that set is unavailable, so an unverifiable span
        still fails closed. Never matches a real clear value, those live in `forbidden_values`
        and drive the hard `leaked_values` block regardless of this method.

        `extra_tokens` are caller-supplied protection tokens for THIS turn (live-chat surrogates
        the caller just minted, absent from the on-disk set the container ships without). They go
        through the SAME Rail 2 as the on-disk set — every forbidden clear value (and each of its
        words) is subtracted — so a live token can never be a real clear value, and per-word spans
        of a multi-word surrogate still match.
        """
        needle = _normalize_for_match(span)
        tokens = self._protection_tokens()
        if tokens and needle in tokens:
            return True
        if extra_tokens:
            safe_extra = self._sanitize_tokens(extra_tokens)
            if needle in safe_extra:
                return True
        return False

    def _sanitize_tokens(self, raw: "frozenset[str]") -> frozenset[str]:
        """Apply Rail 2 to a caller-supplied token set: normalize, expand to words (len>=3), and
        SUBTRACT every forbidden clear value (and its words). Identical treatment to the on-disk
        set in `_protection_tokens`, so a live token can never be a real clear value. Cached per
        input identity to avoid rebuilding the forbidden index on every span."""
        cache = getattr(self, "_extra_tok_cache", None)
        if cache is not None and cache[0] is raw:
            return cache[1]
        toks: set[str] = set()
        for v in raw:
            if not isinstance(v, str) or len(v) < 5:
                continue
            toks.add(_normalize_for_match(v))
            for word in v.split():
                if len(word) >= 3:
                    toks.add(_normalize_for_match(word))
        forbidden_norm = {_normalize_for_match(v) for v in self.forbidden_values}
        for fv in self.forbidden_values:
            for word in str(fv).split():
                if len(word) >= 3:
                    forbidden_norm.add(_normalize_for_match(word))
        result = frozenset(toks - forbidden_norm)
        object.__setattr__(self, "_extra_tok_cache", (raw, result))
        return result

    # Scopes whose structured artifacts hold CLEAR or partially-clear values, never a source of
    # protection tokens: `none` is the cleartext baseline, `_anon-monetary` is a verbatim copy of
    # `none`, and `quasi-yearclear` / `quasi` keep some quasi-identifiers in the clear. Reading
    # these was a real defect: it put real clear names/addresses into the "token" set and could
    # discount a genuine leak. `quasi` is the actual on-disk scope name (the older `quasi-yearclear`
    # is kept for back-compat); the field allow-list below is the real safety regardless.
    _NON_TOKEN_SCOPES = frozenset({"none", "_anon-monetary", "quasi-yearclear", "quasi"})

    # Free-text / categorical fields that are CLEAR by construction (business memos, enums, currency
    # codes) — never tokenized identifiers. Harvesting them would put generic words ("consulting",
    # "management" from a `memo`) into the token set. Skip them so the token set holds only
    # protection tokens. (Rail 2 below still subtracts forbidden clear values as a second guard.)
    _CLEAR_TEXT_FIELDS = frozenset({
        "memo", "party_type", "jurisdiction", "channel", "currency", "risk_rating", "is_pep",
        "city", "amount", "value_date",
    })

    def _protection_tokens(self) -> frozenset[str]:
        """Normalized protection tokens of this corpus, lazily loaded and cached.

        Two safety rails make this incapable of containing a real clear value:
        1. Only genuinely-tokenizing scopes are read (`_NON_TOKEN_SCOPES` excluded), so the
           cleartext baseline and its copies never contribute.
        2. Every value in `forbidden_values` (the clear corpus values) is SUBTRACTED, so even if
           a clear value slipped through rail 1 it cannot be treated as a token.
        Empty (a no-op) when the protected corpus is absent. This is the authoritative fix for
        the discount that must never override a real leak.

        Scope/domain-blindness is intentional and SAFE by direction: the set is the union of
        tokens across every scope on disk, so it may occasionally treat another scope's token as
        a token here. That only ever *under*-flags a value that is itself a token (never clear
        PII), while the dangerous direction, discounting a real clear value, is closed by rail 2
        and by `blocked` checking `leaked_values` first. A per-domain token partition would need
        the protected corpus split by domain on disk, which it is not; adding one would trade a
        benign, provably-bounded imprecision for real complexity.
        """
        cached = getattr(self, "_token_cache", None)
        if cached is not None:
            return cached
        toks: set[str] = set()
        try:
            base = Path(__file__).resolve().parents[2] / "data" / "protected"
            # PREFERRED: a compact, token-only manifest that the serving IMAGE ships. The full
            # per-scope parties.json/transactions.json are `.dockerignore`d (large, and they hold
            # partially-clear scopes), so in the container the glob below finds nothing and the
            # surrogate-discount silently dies — a live `[PERSON]` surrogate the model echoes from
            # retrieved context gets mislabelled PASSWORD and a safe reply is rejected. The manifest
            # (built by scripts/build_token_manifest.py from the tokenizing scopes only, ~1.9MB) is
            # already normalized token strings, no clear values. Rail 2 below still subtracts the
            # forbidden clear values, so this can never discount a real leak.
            manifest = base / "token-manifest.json"
            if manifest.exists():
                for t in json.loads(manifest.read_text()):
                    if isinstance(t, str) and t:
                        toks.add(t)
            else:
                for scope_dir in base.glob("*"):
                    if scope_dir.name in self._NON_TOKEN_SCOPES:
                        continue  # cleartext / year-clear scopes are not token sources
                    for fname in ("parties.json", "transactions.json"):
                        fp = scope_dir / fname
                        if not fp.exists():
                            continue
                        for rec in json.loads(fp.read_text()):
                            for k, v in rec.items():
                                if k in self._CLEAR_TEXT_FIELDS:
                                    continue  # free-text/categorical: clear, not a token
                                if isinstance(v, str) and len(v) >= 5:
                                    toks.add(_normalize_for_match(v))
                                    # The guardrail service flags a MULTI-WORD token (e.g. a
                                    # tokenized PERSON `2S3y A47Vmilfi`) as separate per-word spans,
                                    # neither of which equals the whole token. Add each word
                                    # (len >= 3) so those per-word spans also match. Rail 2 below
                                    # still subtracts every real clear value, so a word that is a
                                    # real name-part can never enter.
                                    for word in v.split():
                                        if len(word) >= 3:
                                            toks.add(_normalize_for_match(word))
        except Exception:  # noqa: BLE001, unavailable -> conservative empty set (no discount)
            toks = set()
        # Rail 2: a real clear value can never be a "protection token". Subtract every forbidden
        # (clear) value, normalized the same way, so the two sets are provably disjoint. This is
        # applied to the word-level tokens too, so a tokenized name whose word happens to collide
        # with a real clear name-part is excluded.
        forbidden_norm = {_normalize_for_match(v) for v in self.forbidden_values}
        # Also subtract the individual WORDS of every forbidden clear value, so a per-word span of a
        # real cleartext name (first or last name alone) can never be discounted as a "token".
        for fv in self.forbidden_values:
            for word in str(fv).split():
                if len(word) >= 3:
                    forbidden_norm.add(_normalize_for_match(word))
        result = frozenset(toks - forbidden_norm)
        object.__setattr__(self, "_token_cache", result)
        return result

    def _needle_index(self) -> "dict":
        """Normalized needles and one compiled digit-boundary alternation, built once.

        The naive per-call form re-normalized all ~24k needles and recompiled ~8,700 digit
        patterns on every scan (measured ~314 ms), thrashing the regex cache and taxing the
        analyst-facing hot path. This precomputes: a map from normalized text-needle ->
        original value, and a single alternation regex for all digit-only needles anchored
        on digit boundaries. Cached against the identity of the forbidden set, so a rebuilt
        Guardrail with the same values reuses it.
        """
        cache_key = (self.forbidden_values, self.high_sensitivity_values)
        cache = getattr(self, "_needle_cache", None)
        if cache is not None and cache[0] == cache_key:
            return cache[1]
        text_needles: dict[str, str] = {}
        digit_needles: list[str] = []
        for value in self.forbidden_values:
            if not value:
                continue
            needle = _normalize_for_match(value)
            if not needle:
                continue
            if needle.isdigit():
                digit_needles.append((needle, value))
            else:
                text_needles.setdefault(needle, value)
        digit_re = None
        if digit_needles:
            # Longest-first so the alternation prefers the longest matching value.
            alt = "|".join(
                re.escape(n) for n, _ in sorted(digit_needles, key=lambda p: -len(p[0]))
            )
            digit_re = re.compile(rf"(?<!\d)(?:{alt})(?!\d)")

        # High-sensitivity digit identifiers get a SECOND, boundary-free alternation so a value
        # embedded in a longer digit run (the accepted residual for ordinary digit needles) is
        # still caught for these narrow high-value shapes.
        hs_re = None
        hs_map: dict[str, str] = {}
        hs_digit = [
            (needle, value)
            for value in self.high_sensitivity_values
            if value and (needle := _normalize_for_match(value)) and needle.isdigit()
        ]
        if hs_digit:
            alt = "|".join(
                re.escape(n) for n, _ in sorted(hs_digit, key=lambda p: -len(p[0]))
            )
            hs_re = re.compile(f"(?:{alt})")
            hs_map = {n: v for n, v in hs_digit}

        index = {
            "text": text_needles,
            "digit_re": digit_re,
            "digit_map": {n: v for n, v in digit_needles},
            "hs_re": hs_re,
            "hs_map": hs_map,
        }
        object.__setattr__(self, "_needle_cache", (cache_key, index))
        return index

    def _leaked(self, content: str) -> tuple[str, ...]:
        """Real corpus values appearing in the content, the unambiguous failure.

        Matching is normalized, not naive substring: adversarial review showed the naive
        form was bypassable by case variants, NFKC-foldable homoglyphs, and format/combining
        characters threaded through a value, while pure-numeric values false-positived inside
        unrelated decimals (a bank account matched inside a SHAP contribution's digits, the
        false-positive class that gets guards switched off). Both sides are ignorable-stripped,
        NFKC-normalized and casefolded; digit-only values must match on a digit boundary.

        Residual, and how it is bounded: a digit-only value embedded in a longer digit run is
        a miss for *ordinary* digit needles (`978024684` hides `78024684`), because anchoring
        on a digit boundary is what prevents decimal-substring false positives on amounts and
        the like. But **high-sensitivity identifiers** (SSN, account, bank account, card, tax
        id) are additionally matched on ANY occurrence via a second alternation, so exactly the
        values a prompt-injection would try to smuggle inside a longer digit run are caught even
        there, at a deliberate false-positive cost for those narrow shapes. A leading
        NFKC-foldable digit folds to the longer-run case; fullwidth/circled/mathematical
        renderings of the value itself are caught. This check is a backstop for the load-bearing
        controls upstream, hardened here because on an unguarded path it can be the last line.
        """
        if not self.forbidden_values:
            return ()
        normalized = _normalize_for_match(content)
        index = self._needle_index()
        hits = [v for needle, v in index["text"].items() if needle in normalized]
        if index["digit_re"] is not None:
            for match in index["digit_re"].findall(normalized):
                hits.append(index["digit_map"][match])
        # High-sensitivity identifiers: any occurrence, even inside a longer digit run.
        if index.get("hs_re") is not None:
            for match in index["hs_re"].findall(normalized):
                hits.append(index["hs_map"][match])
        return tuple(sorted(set(hits)))

    def scan_prompt(self, content: str) -> ScanResult:
        """Score a prompt bound for the model, for manipulation.

        Off unless `scan_prompts=True`, because the only available processor does not
        discriminate on this domain, see the module docstring for the measurement. Kept so
        the finding is reproducible rather than asserted.
        """
        if not self.enabled or not self.scan_prompts:
            return ScanResult(outcome="skipped", score=0.0)
        return self._scan(content, self.injection_processor, ("user", "ai"))

    def scan_response(self, content: str,
                      extra_tokens: "frozenset[str] | None" = None) -> ScanResult:
        """Score a model response for identifiers before it reaches a human.

        This is the check the architecture lacked. A rationale is generated from tokenized
        evidence, but nothing verified that the model had not reconstructed, guessed, or been
        induced to emit something identifying.

        `extra_tokens` are protection tokens the CALLER minted this turn (e.g. the live-chat
        surrogates produced by tokenizing the user's own message) that may not be on disk in
        `data/protected/`. The serving container ships without the protected artifacts
        (`.dockerignore` strips `data/protected/`), so the on-disk token set is empty there and
        the surrogate-discount could never recognise a freshly-tokenized name — the service then
        rejects a safe reply (a `[PERSON]` surrogate mislabelled PASSWORD). Unioning the turn's
        own tokens restores the discount without shipping the artifacts. Rail 2 still subtracts
        every forbidden clear value, and `blocked` consults `leaked_values` first, so this cannot
        discount a real leak.
        """
        if not self.enabled:
            return ScanResult(outcome="skipped", score=0.0)
        return self._scan(content, self.pii_processor, ("ai", "user"),
                          extra_tokens=extra_tokens)
