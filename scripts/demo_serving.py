"""Live serving demo: the same protection boundary on a chatbot and a multi-agent hop.

Shows Parts A (domain config) and B (serving wrapper) together, with NO live services by
default (fakes stand in for discovery/protect/LLM) so it runs on a clean checkout:

    python scripts/demo_serving.py                    # fakes, any domain
    python scripts/demo_serving.py --domain healthcare
    python scripts/demo_serving.py --live --model bedrock-sonnet-5   # real services (paid)

The point: one `ConversationSession` gives a chatbot turn the batch pipeline's guarantees, and
because tokens are stable, a value tokenized by agent A survives verbatim to agent B and only
the final role-gated presentation re-identifies it. No agent in the middle holds plaintext.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protometer.env import load_dotenv  # noqa: E402

load_dotenv(ROOT)

from protometer.domains import domain_names, get_domain  # noqa: E402
from protometer.reidentify import ANALYST, INVESTIGATOR  # noqa: E402
from protometer.serving import ConversationSession, protect_text  # noqa: E402


class _FakeProtector:
    """Deterministic, reversible tokenizer, so the demo needs no hosted API."""

    def __init__(self):
        self._fwd, self._rev, self._n = {}, {}, 0

    def protect_values(self, values, element):
        out = []
        for v in values:
            if v not in self._fwd:
                self._n += 1
                tok = f"{element[:2].upper()}x{self._n:03d}"
                self._fwd[v], self._rev[(element, tok)] = tok, v
            out.append(self._fwd[v])
        return out

    def protect_value(self, value, element):
        return self.protect_values([value], element)[0]

    def unprotect_values(self, tokens, element):
        return [self._rev.get((element, t), t) for t in tokens]


class _FakeLLM:
    """Echoes tokens back tagged, as a well-behaved token-reasoning model would."""

    trace_session = ""
    trace_component = ""

    def complete(self, system, prompt, max_tokens=None):
        import re
        toks = re.findall(r"\[[A-Z_]+\][^\[]+\[/[A-Z_]+\]", prompt)
        who = toks[0] if toks else "the subject"
        return f"Reviewed {who}: activity is consistent with the stated concern."


def _fake_discovery_for(names_in_text, text):
    """A stand-in discovery result: mark each known name span as PERSON."""
    ents = []
    for name in names_in_text:
        i = text.find(name)
        if i >= 0:
            ents.append({"entity_type": "PERSON", "start": i, "end": i + len(name)})
    return ents


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="aml", choices=domain_names())
    ap.add_argument("--live", action="store_true", help="use real services (paid)")
    ap.add_argument("--model", default="bedrock-sonnet-5")
    args = ap.parse_args()

    domain = get_domain(args.domain)
    print(f"\n=== Domain: {domain.name} ({domain.label}) ===")
    print(f"    guardrail prompt model: {domain.injection_processor}")
    print(f"    system prompt: {domain.investigation_prompt}\n")

    live_guardrail = None
    live_roster = None
    if args.live:
        from protometer.llm import get_llm
        from protometer.protect import Protector
        protector = Protector()
        # allow_fallback=False, matching ui/api/app.py and run_hybrid: a throttle must fail
        # VISIBLY, not silently answer as a different (possibly PAID) fallback model while the
        # demo labels the reply as `args.model` — the model-attribution + unbudgeted-spend failure
        # the spend-cap rails exist to prevent.
        llm = get_llm(args.model, trace_component="serving", enable_cache=False,
                      allow_fallback=False)
        # Build the egress guardrail AND the hybrid roster, matching the UI serving boundary
        # (ui/api/app.py). Without the guardrail the fail-closed rule withholds every reply, so a
        # live demo that omits it can never show an answer. Best-effort: if the sidecar is down we
        # note it and drop to require_guardrail=False so the demo still shows the reply, honestly
        # labelled as unscanned.
        import json as _json

        import protometer.ingest  # noqa: F401  (real discovery is used)
        from protometer.guardrail import Guardrail
        from protometer.roster import roster_from_parties
        parties_path = ROOT / "data" / "corpus" / "parties.json"
        try:
            live_guardrail = Guardrail.for_corpus(parties_path, probe=False, domain=domain)
        except Exception as exc:  # noqa: BLE001
            print(f"    (guardrail unavailable: {type(exc).__name__}; reply will be shown UNSCANNED)")
        try:
            live_roster = roster_from_parties(_json.loads(parties_path.read_text()))
        except Exception:  # noqa: BLE001
            live_roster = None
    else:
        import protometer.ingest as ingest
        names = ["Leila Rahman", "Marcus Chen"]
        # Patch discovery to our fake for the demo text.
        _orig = ingest.discover_entities
        ingest.discover_entities = lambda text, **kw: _fake_discovery_for(names, text)
        protector, llm = _FakeProtector(), _FakeLLM()

    # ---- Chatbot: one protected turn, role-gated reply --------------------------------------
    print("--- Chatbot turn (role = Investigator, may re-identify) ---")
    session = ConversationSession(
        protector=protector, llm=llm, conversation_id="demo-conv-1",
        role=INVESTIGATOR, domain=domain,
        system_prompt="You review case material over stable pseudonymous tokens.",
        guardrail=live_guardrail, roster=live_roster,
        # If the live guardrail could not be built, do not fail closed (this is a demo, not the
        # analyst path): show the reply and label it unscanned. Fakes need no guardrail.
        require_guardrail=bool(live_guardrail) if args.live else False,
    )
    user_msg = "Please look into Leila Rahman regarding recent activity."
    print(f"  user     : {user_msg}")
    result = session.turn(user_msg)
    print(f"  tokenized: {result.protected_input}   ({result.entities_protected} PII protected)")
    print("  model saw: tokens only (never 'Leila Rahman')")
    print(f"  reply     : {result.reply}   (revealed={result.revealed}, ok={result.ok})")

    # ---- Multi-agent hop: token survives A -> B, only B's presentation re-identifies --------
    print("\n--- Multi-agent hop (token stability across agents) ---")
    agent_a_text = "Escalation: Marcus Chen flagged by monitoring."
    protected, n = protect_text(agent_a_text, protector, domain)
    print(f"  agent A tokenizes: {protected}")
    print("  agent A -> agent B passes the TOKEN, never the name")
    # Agent B (analyst role: individuals stay tokenized) presents its result.
    from protometer.reidentify import reidentify
    b_analyst = reidentify(protected, protector, ANALYST)
    b_investigator = reidentify(protected, protector, INVESTIGATOR)
    print(f"  agent B as Analyst      : {b_analyst.text}   (revealed={b_analyst.revealed})")
    print(f"  agent B as Investigator : {b_investigator.text}   (revealed={b_investigator.revealed})")
    print("\n  => same token, two role-gated views; no middle agent held plaintext.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
