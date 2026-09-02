"""Egress guard: every model reply is scanned before a human sees it. Fail closed.

Checks run in order on the user path (a public pharma bot, the bar is high):

1. **PII-leak check.** The reply must not contain any of the user's tokenized PII in the clear,
   and must not contain a raw token wrapper either (the model should have reasoned over tokens and
   produced prose; a leaked token is a plumbing bug). A leak blocks the reply.

2. **Medical-advice check.** The reply must not dose, diagnose, or direct treatment. Phrases that
   read as personal medical instruction ("you should take", "your dose", "stop taking", "I
   recommend you") are blocked and replaced with the safe redirect.

3. **Meta-language check.** A reply reciting its own instructions instead of answering ("I can only
   use the context", "as an AI...") is refused and swapped for the clean fallback.

4. **Groundedness check.** The reply must be supported by the retrieved context: (a) its substantive
   content terms must clear a lexical-overlap floor, and (b) every NUMBER it asserts must appear in
   the context, so a fabricated statistic can't ride along on on-topic words. This is the
   anti-hallucination gate for a domain where a wrong answer (especially a wrong number) is a safety
   issue. Below either bar -> refuse (not block): the bot answers only from the official material.

Guards BLOCK or REFUSE on the user path (fail closed), never silently pass a bad reply. Each
returns a structured verdict the API surfaces so the UI can explain what happened.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A leaked PII wrapper ([EMAIL], [NAME], ...) or a bare surrogate token (EM3f0a1b2c3d) must not
# survive to the visitor. Built from the protector's canonical PII-type list, so a legitimate
# bracketed acronym in an answer ("[FDA]") is NOT mistaken for a leaked token and blocked.
from app.protect.protector import TOKEN_RE as _TOKEN_TAG

# A bare REDACTION token that lost its wrapper. This matches ONLY our own redaction shape (the "RD"
# prefix + 12 hex chars, see protector._redaction_token), which is unambiguous and definitely a leak
# if it reaches a reply. We deliberately do NOT use a broad "alphanumeric run with a letter and a
# digit" heuristic, that false-blocked legitimate grounded content in a pharma corpus (ClinicalTrials
# IDs like "NCT01234567", product/CPT codes, "BOTOX100"). Real Protegrity surrogates are
# format-preserving and indistinguishable from such identifiers, so the wrapper check plus the
# per-turn `user_tokens` set (the actual issued tokens) are what catch a genuine leak; this covers the
# one bare shape we know we mint.
_SURROGATE_RE = re.compile(r"\bRD[0-9a-f]{12}\b")

# Personal medical-instruction patterns. Deliberately conservative: informational statements
# ("BOTOX is used to treat...") are fine; second-person directives are not.
_ADVICE = re.compile(
    # "you should/need to/can/must/ought to/you'll need to <verb>", covering the common directive
    # paraphrases a model produces, not just "you should". The "'ll" form attaches to "you" (no space).
    r"\byou(?:'ll| will)? (?:should|need to|must|can|ought to|could|may want to|want to|have to) "
    r"(?:take|use|stop|start|increase|decrease|inject|try|switch|adjust|change|reduce|raise)\b|"
    r"\byour (?:dose|dosage) (?:is|should|will be|of)\b|"
    r"\bi (?:recommend|advise|suggest|would recommend|'?d recommend) (?:you|that you|taking|using)\b|"
    r"\b(?:make sure to|be sure to|remember to) (?:take|use|inject|stop|start)\b|"
    r"\bstart (?:with|by taking) \d|\bstop taking\b|\bincrease your\b|\bthe right dose for you\b",
    re.I,
)

# Does the ANSWER actually discuss safety/risk? Used to decide whether to attach the Important
# Safety Information note. Deliberately keyed on substantive risk language, NOT the bare word
# "warning" (which appears in boilerplate on every source page), the note should reflect what the
# answer is about, not what the source page happened to footer.
_SAFETY = re.compile(
    r"side effect|adverse|serious (?:reaction|symptom|problem)|distant spread|spread of toxin|"
    r"boxed warning|important safety information|difficulty (?:swallowing|breathing)|"
    r"trouble (?:swallowing|breathing)|contraindicat|do not (?:use|receive)|"
    r"\brisk(?:s|y)?\b|allergic|overdose|not safe|avoid", re.I)

# Meta-language: the model echoing its own instructions instead of answering ("use only the
# context", "never use outside knowledge", "I can only share information from my sources"). A reply
# that talks ABOUT how it is allowed to answer, rather than answering, is refused and replaced with
# the clean fallback, the visitor should never see the plumbing.
#
# CAUTION: "context" and "sources" are ordinary words a real answer uses ("in the context of
# migraine", "discuss the sources of your headaches"). So we match them ONLY inside first-person,
# self-referential framing ("I can only use the context", "based on the context provided, I don't
# have…", "from my sources"), never the bare noun, otherwise good answers are silently refused.
_META_LEAK = re.compile(
    r"\bbased on the (?:provided |given |above )?context\b|"
    r"\bi (?:can only|only) (?:use|rely on|draw from|answer from) (?:the |my |provided )?"
    r"(?:context|sources|information)\b|"
    r"\bfrom (?:the|my|provided|given) (?:context|sources) (?:above|provided|given)?\b|"
    r"\boutside knowledge\b|\bthese instructions\b|\bmy instructions\b|\bthe system prompt\b|"
    r"\bi am (?:instructed|programmed|designed) to\b|\bas an ai\b|"
    r"\bi can only (?:share|provide|answer) (?:general )?information\b|\bi'?m only able to\b|"
    r"\bnever use (?:outside|external|general) (?:knowledge|information)\b",
    re.I,
)

_STOPWORDS = set("a an the and or of to in is are was were be been being for on with as at by "
                 "this that these those it its from can could may might will would should you your "
                 "i we they he she them his her our their about into over under more most".split())

# Numeric-grounding: a fabricated statistic ("a 92 percent success rate", "200 mg") is the concrete
# safety risk lexical overlap misses, the surrounding words are on-topic, so the reply passes the
# overlap floor while asserting a number that was never in the context. We extract each numeric
# claim AS A (number, unit) PAIR and refuse if any is absent from the retrieved context.
#
# Capturing the UNIT is what makes this safe for a dosing bot: "200 mg" and "200 units" are
# DIFFERENT claims, so a reply saying "200 mg" is not grounded by a context that says "200 units".
# The unit is the token immediately following the number (mg, units, %, days, months, mm), or the
# empty unit for a bare number. We deliberately DROP bare spelled-out cardinals used as ordinary
# words ("one option", "two of the") — they are not measured claims and caused false refusals.
_UNIT_WORDS = ("milligrams?", "micrograms?", "milliliters?", "mg", "mcg", "ug", "ml", "cc",
               "units?", "iu", "u", "percent", "%", "mm", "cm",
               "days?", "weeks?", "months?", "years?", "hours?", "hrs?", "minutes?", "mins?",
               "times", "injections?", "sites?", "doses?", "mg/kg")
# A number (digits, optional thousands separators / decimal / trailing %) followed by an optional
# unit word. The number must have a DIGIT, spelled-out numbers only count when a unit follows.
_NUM_UNIT_RE = re.compile(
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|" + "|".join(u for u in _UNIT_WORDS if u != "%") + r")?",
    re.I,
)
# Spelled-out small numbers, counted ONLY when immediately followed by a unit ("three days",
# "two weeks"), never bare ("one option"). Maps the word to its digit for canonical comparison.
_WORD_NUMS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
              "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}
_WORD_NUM_UNIT_RE = re.compile(
    r"\b(?P<word>one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?P<unit>" + "|".join(u for u in _UNIT_WORDS if u not in ("%", "u")) + r")\b",
    re.I,
)


# Dosage/measurement units where the number is only meaningful WITH its exact unit, "200 mg" and
# "200 units" are different doses, and confusing them is a safety error. For these, numeric grounding
# requires the same (number, unit) pair in the context; for all other units a number match suffices.
_DOSAGE_UNITS = {"mg", "ug", "ml", "iu", "unit", "%"}


def _canon_unit(unit: str | None) -> str:
    """Canonicalise a unit token so trivial variants compare equal ('unit'/'units'->'unit',
    'percent'/'%'->'%', 'day'/'days'->'day'). Empty/None -> '' (a bare number)."""
    if not unit:
        return ""
    u = unit.lower().rstrip(".")
    if u in ("percent", "%"):
        return "%"
    u = u.rstrip("s")                      # units->unit, days->day, sites->site, milligrams->milligram
    # Fold spelled-out and abbreviated forms of the same unit onto one canonical token, so
    # "200 milligrams" and "200 mg" compare equal (and both differ from "200 units").
    return {"hr": "hour", "min": "minute", "mcg": "ug", "cc": "ml",
            "milligram": "mg", "microgram": "ug", "milliliter": "ml"}.get(u, u)


def _claimed_numbers(text: str) -> set[tuple[str, str]]:
    """Numeric claims in `text` as normalised (number, canonical-unit) pairs. '1,200'/'1200' compare
    equal; '92%' == ('92','%'); '200 mg' == ('200','mg') which is DISTINCT from ('200','unit'). A
    bare number is ('200',''). Spelled-out numbers count only with a unit ('three days')."""
    claims: set[tuple[str, str]] = set()
    for m in _NUM_UNIT_RE.finditer(text):
        raw = m.group("num").replace(",", "")
        if raw.endswith(".0"):
            raw = raw[:-2]
        claims.add((raw, _canon_unit(m.group("unit"))))
    for m in _WORD_NUM_UNIT_RE.finditer(text):
        claims.add((_WORD_NUMS[m.group("word").lower()], _canon_unit(m.group("unit"))))
    return claims


@dataclass
class Verdict:
    ok: bool
    outcome: str                      # "passed" | "blocked" | "refused"
    reason: str = ""
    grounding_score: float = 0.0
    safety_relevant: bool = False
    leaked: list[str] = field(default_factory=list)


def _content_terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS}


def scan_reply(reply: str, *, context: str, user_tokens: set[str],
               min_grounding: float = 0.20, raw_reply: str | None = None) -> Verdict:
    """Scan one re-identified-ready reply. `context` is the retrieved chunk text the model was
    given; `user_tokens` are the token strings from the user's protected message.

    `raw_reply`, when given, is the PRE-scrub text the leak check runs against (the caller may have
    scrubbed the user's own benign token echo out of `reply` for presentation, but the leak check
    must still see the original so a leaked wrapper/token cannot be laundered past it). Defaults to
    `reply` when not supplied."""
    # Safety note reflects the ANSWER, not the sources: attach it only when the reply itself
    # discusses risk/side-effects/warnings. The source pages carry an ISI footer on every page, so
    # keying on context would flag every reply (which is exactly the "ISI on everything" bug).
    safety = bool(_SAFETY.search(reply))

    # 1. PII leak. Two parts, on purpose:
    #   (a) FOREIGN leak, checked on the PRE-scrub draft: any [TYPE] wrapper or a bare surrogate-shaped
    #       token that survived to the draft is a leak of something we did NOT put in (indexed content,
    #       a hallucinated token). Checking the un-scrubbed draft means scrubbing the user's own benign
    #       echo cannot launder a foreign token past this gate.
    #   (b) OWN leak, checked on the PRESENTED (scrubbed) text: if the user's own token still survives
    #       AFTER scrubbing, block it. A successfully-scrubbed own echo leaves nothing here and passes,
    #       which is the whole point of scrubbing (don't block an otherwise-good answer for it).
    leak_src = raw_reply if raw_reply is not None else reply
    # Exclude the user's OWN tokens (and their wrappers) from the foreign-leak scan, so a benign
    # self-echo the caller intends to scrub is not itself flagged as a foreign surrogate by shape.
    foreign_src = leak_src
    for t in user_tokens:
        if t:
            foreign_src = re.sub(rf"\[[A-Z_]+\]\s*{re.escape(t)}\s*\[/[A-Z_]+\]", " ", foreign_src)
            foreign_src = foreign_src.replace(t, " ")
    leaked = [t for t in user_tokens if t in reply]   # own token surviving the scrub -> block
    if _TOKEN_TAG.search(foreign_src) or _SURROGATE_RE.search(foreign_src) or leaked:
        return Verdict(ok=False, outcome="blocked",
                       reason="reply contained protected tokens or user PII", leaked=leaked,
                       safety_relevant=safety)

    # 2. Medical advice: block personal directives.
    if _ADVICE.search(reply):
        return Verdict(ok=False, outcome="blocked",
                       reason="reply read as personal medical advice", safety_relevant=safety)

    # 2b. Meta-language leak: the model reciting its instructions instead of answering. Refuse and
    # let the caller swap in the clean fallback, the visitor should never see this plumbing.
    if _META_LEAK.search(reply):
        return Verdict(ok=False, outcome="refused",
                       reason="reply echoed instructions/meta-language instead of answering",
                       safety_relevant=False)

    # 3. Groundedness: the reply's content terms must overlap the retrieved context.
    r_terms = _content_terms(reply)
    if not r_terms:
        return Verdict(ok=False, outcome="refused", reason="empty or non-substantive reply",
                       grounding_score=0.0, safety_relevant=safety)
    c_terms = _content_terms(context)
    overlap = len(r_terms & c_terms) / len(r_terms)
    if overlap < min_grounding:
        return Verdict(ok=False, outcome="refused",
                       reason=f"reply not grounded in retrieved content (overlap {overlap:.2f})",
                       grounding_score=overlap, safety_relevant=safety)

    # 3b. Numeric grounding: every number the reply ASSERTS must appear in the retrieved context.
    # Word overlap can be high while a fabricated statistic ("a 92% success rate in three days")
    # rides along on on-topic words; for a pharma bot an invented number is a safety issue, so a
    # reply that states a figure absent from the context is refused (anti-hallucination on specifics).
    reply_nums = _claimed_numbers(reply)
    if reply_nums:
        context_nums = _claimed_numbers(context)
        context_values = {n for n, _u in context_nums}
        ungrounded: list[tuple[str, str]] = []
        for n, u in reply_nums:
            if (n, u) in context_nums:
                continue                               # exact number+unit match: grounded
            if u in _DOSAGE_UNITS:
                # Dosage precision is safety-critical: a dose number must appear with its EXACT unit
                # in the context ("200 mg" is not grounded by "200 units"). No number-only fallback.
                ungrounded.append((n, u))
            elif n not in context_values:
                # Non-dosage unit (sites, days, %, months): the number itself must appear in the
                # context, but a compound-unit mismatch ("31 sites" vs "31 injection sites") is fine.
                ungrounded.append((n, u))
        if ungrounded:
            pretty = [f"{n}{(' ' + u) if u else ''}" for n, u in sorted(ungrounded)]
            return Verdict(
                ok=False, outcome="refused",
                reason=f"reply asserts numbers not in retrieved content: {pretty}",
                grounding_score=overlap, safety_relevant=safety)

    return Verdict(ok=True, outcome="passed", grounding_score=overlap, safety_relevant=safety)
