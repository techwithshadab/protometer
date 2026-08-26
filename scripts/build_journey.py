"""Build the per-domain 'data journey' artifact for the UI: one representative record traced
through THAT DOMAIN'S pipeline so the batch view shows the CONCRETE input/output at each stage,
not just metrics. Each domain gets stages matching its own flow (AML's investigation curve,
healthcare's HIPAA de-id, support's dual-gate) and copy drawn from its own committed artifacts.

The corpus is fully synthetic (`corpus/generate.py`), so showing ONE record's clear form for the
demo is safe: no real person is depicted. The clear side is included ONLY in this small, curated
artifact (never via a bulk endpoint), paired with its protected form.

    python scripts/build_journey.py            # writes data/eval/<domain>/journey.json per domain

No LLM calls. Reads the committed clear corpus + the protected/eval artifacts on disk, and makes
Protegrity unprotect calls (not billed like the hosted model) to re-identify the spans each role
may see for the Present-stage role views.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amlguard.persist import atomic_write_json  # noqa: E402
from amlguard.protect import Protector  # noqa: E402
from amlguard.reidentify import (  # noqa: E402
    ANALYST,
    AUDITOR,
    BILLING,
    INVESTIGATOR,
    RESEARCHER,
    SUPERVISOR,
    SUPPORT_AGENT,
    TREATING_CLINICIAN,
    reidentify,
)

# The AML subject whose narrative we trace (rich narrative, reused across the chat demo).
_SUBJECT_ID = "P02463"
_IDENTITY_TAGS = {"PERSON", "SOCIAL_SECURITY_ID", "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS",
                  "LOCATION", "CREDIT_CARD", "BANK_ACCOUNT", "ACCOUNT_NUMBER"}

# Which roles are relevant to each domain's Present/re-identification step, most-restricted first.
_DOMAIN_ROLES = {
    "aml": [AUDITOR, ANALYST, INVESTIGATOR],
    "healthcare": [RESEARCHER, BILLING, TREATING_CLINICIAN],  # clinical access model, not AML roles
    "customer-support": [SUPPORT_AGENT, SUPERVISOR],
}

_TAG_RE = re.compile(r"\[([A-Z_]+)\]([^\[]*?)\[/\1\]")


def _role_view(protected_text: str, clear_text: str, role, protector) -> dict:
    """Render `protected_text` as ROLE would see it, using the SAME re-identification the analyst
    path uses: `reidentify()` unprotects (Protegrity unprotect, not an LLM call) exactly the spans
    whose entity type the role may see, and leaves the rest tokenized; we then mask those remaining
    tokens to `[type: masked]`.

    So a partial-reveal role shows real cleartext for its permitted identifiers (e.g. the analyst
    sees the organization name in the clear while the person stays masked), not the internal token
    markup. The fully-entitled role sees the whole clear record. Counts reveals vs withholds."""
    fully = role.may_unprotect >= _IDENTITY_TAGS  # entitled to everything identity

    if fully:
        revealed = sum(1 for _ in _TAG_RE.finditer(protected_text))
        return {"name": role.name, "label": role.label,
                "may_unprotect": sorted(role.may_unprotect),
                "sees": clear_text, "revealed": revealed, "withheld": 0, "fully": True}

    # Reveal the permitted spans for real; reidentify keeps the wrapper tags (strip_tags=False) so a
    # revealed span reads `[TYPE]Clear Value[/TYPE]` while a withheld span keeps its raw token.
    res = reidentify(protected_text, protector, role=role, strip_tags=False)
    withheld = 0

    def render(m: "re.Match[str]") -> str:
        # reidentify only unprotects the role's permitted types, so mask by the allow-list: a
        # permitted span shows its (now cleartext) inner value with the wrapper tag stripped; any
        # other span is one this role may not see and is masked.
        nonlocal withheld
        etype, inner = m.group(1), m.group(2)
        if etype in role.may_unprotect:
            return inner  # revealed: show the cleartext value, drop the wrapper tag
        withheld += 1
        return f"[{etype}: masked]"

    shown = _TAG_RE.sub(render, res.text)
    return {"name": role.name, "label": role.label, "may_unprotect": sorted(role.may_unprotect),
            "sees": shown, "revealed": res.revealed, "withheld": withheld, "fully": False}


def _role_views(domain: str, protected_text: str, clear_text: str, protector) -> list[dict]:
    return [_role_view(protected_text, clear_text, r, protector)
            for r in _DOMAIN_ROLES.get(domain, _DOMAIN_ROLES["aml"])]


def _first(records: list[dict], subject_id: str, key: str = "subject_party_id") -> dict | None:
    return next((r for r in records if r.get(key) == subject_id), None)


def _count_tokens(protected_text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for tag in re.findall(r"\[([A-Z_]+)\][^\[]+\[/\1\]", protected_text):
        out[tag] = out.get(tag, 0) + 1
    return out


def _load(path: str):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else None


# ── AML: the full investigation curve, traced over a tokenized case note ───────────────────────
def _aml_journey() -> dict:
    clear_parties = _load("data/corpus/parties.json") or []
    cn = _first(_load("data/corpus/narratives.json") or [], _SUBJECT_ID) or {}
    pn = _first(_load("data/protected/direct/narratives.json") or [], _SUBJECT_ID) or {}
    party = _first(clear_parties, _SUBJECT_ID, key="party_id") or {}
    clear_text, prot_text = cn.get("text", ""), pn.get("text", "")
    tok = _count_tokens(prot_text)
    identity_tokens = sum(v for k, v in tok.items() if k in _IDENTITY_TAGS)

    stages = [
        {"id": "ingest", "title": "Ingest",
         "in_label": "Clear case note (as received)", "in": clear_text,
         "out_label": "Protected case note (tokenized in place)", "out": prot_text,
         "caption": f"{identity_tokens} identity spans tokenized. Amounts, dates and the pattern "
                    f"stay intact."},
        {"id": "train", "title": "Train",
         "in_label": "Protected ledger row (features over tokens)",
         "in": "amount=$83,472 · 9 sub-threshold deposits · counterparty graph · temporal window",
         "out_label": "Classifier score (no plaintext identifier)",
         "out": "Risk score from graph-walk + temporal features; identity fields are tokens and "
                "contribute nothing.",
         "caption": "The model trains on the protected ledger: identity is a token, the signal is "
                    "behaviour."},
        {"id": "retrieve", "title": "Retrieve",
         "in_label": "Query over the protected index",
         "in": "behavioural: 'structuring, sub-threshold cash deposits'   |   "
               "identity: 'find records for this person'",
         "out_label": "What survives protection",
         "out": "A behavioural query still finds the record (the pattern lives in the tokens). An "
                "identity query collapses (a token has no link to the name).",
         "caption": "Retrieval survives or dies by what the query asks for."},
        {"id": "infer", "title": "Infer",
         "in_label": "Prompt the model actually received (tokens only)", "in": prot_text,
         "out_label": "Model's reply, written over tokens",
         "out": "Summarises the case citing the token subject and the deposit pattern; never emits "
                "a real identifier, because it never saw one.",
         "caption": "The language model reasons entirely over pseudonymous tokens."},
        {"id": "egress", "title": "Egress",
         "in_label": "Model reply, pre-release",
         "in": "reply over tokens (e.g. [PERSON]…[/PERSON], [SOCIAL_SECURITY_ID]…)",
         "out_label": "Guardrail verdict",
         "out": "Scanned before a human sees it. A real identifier that slipped through is blocked; "
                "a reply written only over protection tokens is released, because a token carries no "
                "real identity.",
         "caption": "Every response is scanned at the boundary."},
        {"id": "present", "title": "Present",
         "in_label": "Protected reply",
         "in": "[PERSON]2S3y A47Vmilfi[/PERSON] … flagged for structuring …",
         "out_label": "Role-gated view",
         "out": "Auditor: nothing · Analyst: structure only · Investigator: full re-identification "
                "(the entitled role).",
         "caption": "Plaintext only for the role entitled to it: the single presentation boundary."},
    ]
    return {"subject_name_clear": party.get("full_name", ""), "subject_party_id": _SUBJECT_ID,
            "token_counts": tok, "identity_tokens": identity_tokens,
            "clear_narrative": clear_text, "protected_narrative": prot_text, "stages": stages}


# ── Healthcare: Safe Harbor -> Expert Determination, over a clinical record ────────────────────
def _healthcare_journey() -> dict:
    d = _load("data/eval/healthcare/deidentify.json") or {}
    sh, ed = d.get("safe_harbor", {}), d.get("expert_determination", {})
    b, a = ed.get("before", {}), ed.get("after", {})
    present = sh.get("identifiers_present", {})
    cats = "; ".join(k.replace("_", " ") for k in present)
    # An illustrative clinical record (synthetic; the schema is the real one from the artifact).
    clear = ("Patient: Sana Choudhury · MRN 88431 · ZIP 91344 · region West · "
             "diagnosis_date 2025-03-14 · Dx: type-2 diabetes, hypertension.")
    prot = ("Patient: [PERSON]2S3y A47Vmilfi[/PERSON] · MRN [MED_REC]MRx88[/MED_REC] · "
            "ZIP [GEO]913xx[/GEO] · region [GEO]West[/GEO] · diagnosis_date [DATE]2025[/DATE] · "
            "Dx: type-2 diabetes, hypertension.")
    stages = [
        {"id": "safe_harbor", "title": "Safe Harbor",
         "in_label": "Clear clinical record", "in": clear,
         "out_label": "De-identified record (HIPAA Safe Harbor)", "out": prot,
         "caption": f"{len(present)}/18 Safe-Harbor identifier categories present are removed or "
                    f"tokenized ({cats}); the clinical facts remain."},
        {"id": "expert_determination", "title": "Expert Determination",
         "in_label": "Residual re-identification risk (before k-anonymization)",
         "in": f"prosecutor {b.get('prosecutor_risk')} · journalist {b.get('journalist_risk')} · "
               f"marketer {b.get('marketer_risk')} · k={b.get('k_anonymity')}",
         "out_label": "Risk after k=5 generalization",
         "out": f"prosecutor {a.get('prosecutor_risk')} · journalist {a.get('journalist_risk')} · "
                f"marketer {a.get('marketer_risk')} · k={a.get('k_anonymity')}"
                + (": Expert Determination MET" if ed.get("expert_determination_met")
                   else ": NOT certified on this sample (worst-case risk unchanged)"),
         "caption": "The residual risk is quantified and reported honestly, not asserted away."},
    ]
    return {"subject_name_clear": "Sana Choudhury", "subject_party_id": "(clinical sample)",
            "token_counts": {}, "identity_tokens": len(present),
            "clear_narrative": clear, "protected_narrative": prot, "stages": stages}


# ── Customer support: Gate 1 (protect) -> Gate 2 (role dual-gate) ──────────────────────────────
def _support_journey() -> dict:
    d = _load("data/eval/support/dual_gate.json") or {}
    roles = {r["name"]: r for r in d.get("roles", [])}
    agent, sup = roles.get("support_agent", {}), roles.get("supervisor", {})
    stages = [
        {"id": "gate1", "title": "Gate 1 · Protect",
         "in_label": "Customer message (as received)", "in": d.get("inbound", ""),
         "out_label": "Tokenized before anything downstream", "out": d.get("protected_inbound", ""),
         "caption": f"{d.get('entities_protected', 0)} PII entities protected at the door."},
        {"id": "gate2", "title": "Gate 2 · Role Dual-Gate",
         "in_label": "Support Agent view (least privilege)", "in": agent.get("sees", ""),
         "out_label": "Supervisor view (entitled to re-identify)", "out": sup.get("sees", ""),
         "caption": "The same protected reply, re-identified differently per role, enforced at the "
                    "presentation boundary."},
    ]
    return {"subject_name_clear": "Sarah Chen", "subject_party_id": "(support sample)",
            "token_counts": {}, "identity_tokens": d.get("entities_protected", 0),
            "clear_narrative": d.get("inbound", ""),
            "protected_narrative": d.get("protected_inbound", ""), "stages": stages}


_BUILDERS = {"aml": _aml_journey, "healthcare": _healthcare_journey,
             "customer-support": _support_journey}
_SUBTITLE = {
    "aml": "The same synthetic case, stage by stage: cleartext identifiers in, tokens through, "
           "plaintext only at the gated end.",
    "healthcare": "One synthetic patient record: Safe Harbor removes the direct identifiers, "
                  "Expert Determination quantifies what risk remains.",
    "customer-support": "One customer message, protected at the door, then re-identified "
                        "differently for a front-line agent and a supervisor.",
}


def build_journey(domain: str, protector: Protector) -> dict:
    core = _BUILDERS.get(domain, _aml_journey)()
    # The SAME protected record, re-identified for each role relevant to this domain: the concrete
    # "same data, different eyes" view for the Present stage and the chatbot role toggle.
    role_views = _role_views(domain, core["protected_narrative"], core["clear_narrative"], protector)
    return {
        "domain": domain,
        "subtitle": _SUBTITLE.get(domain, _SUBTITLE["aml"]),
        "note": "Fully synthetic record. No real person is depicted. This is the only place a clear "
                "is the only place a clear value is shown, paired with its protected form, to "
                "illustrate the transformation.",
        "role_views": role_views,
        **core,
    }


def main() -> int:
    # One Protegrity login for the whole run (the role views unprotect permitted spans for real).
    protector = Protector()
    for d in ("aml", "healthcare", "customer-support"):
        j = build_journey(d, protector)
        dest = ROOT / "data" / "eval" / d / "journey.json"
        atomic_write_json(dest, j)
        print(f"  {d}: {len(j['stages'])} stages, {j['identity_tokens']} identity spans -> "
              f"{dest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
