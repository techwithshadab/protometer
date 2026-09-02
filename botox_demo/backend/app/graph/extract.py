"""Extract a BOTOX knowledge graph (entities + relationships) from chunks, deterministically.

Why a curated lexicon, not an NER model: the domain is small, closed, and safety-critical. A
general NER model would miss "cervical dystonia" as an indication, mislabel "BOTOX Complete" (a
savings program) as a drug, and, worst for a pharma bot, hallucinate relationships. A curated
lexicon seeded from the ACTUAL crawled content gives us precision we can audit line by line, is
fully deterministic and offline ($0, no LLM), and every node/edge is traceable to the chunk it came
from. The FDA-approved indication list, the ISI side effects, and the savings-program vocabulary
below are taken verbatim from the botox.com pages this pipeline crawls.

Relationships are extracted by co-occurrence within a chunk, then typed by the pair of entity
types involved (an Indication + a BodyArea that co-occur -> AFFECTS; a SafetyWarning +
anything -> WARNS_ABOUT). Co-occurrence is a deliberate, conservative choice: within one ~650-char
passage two clinical entities appearing together is a real semantic link on this corpus, and it
cannot fabricate a relationship that the source text does not support.

    python -m app.graph.extract      # data/processed/chunks.jsonl -> data/processed/graph.json

Output graph.json: {nodes:[{id,label,type,sources:[chunk_id]}], edges:[{src,dst,rel,sources:[...]}]}
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from app.paths import PROCESSED

# ── The curated BOTOX lexicon, seeded from the crawled botox.com content ──────────────────────
# Each entry: canonical label -> list of surface forms (regex-escaped, matched case-insensitively
# on word boundaries). Canonical label is the node id's basis; surface forms are how it appears in
# text. Ordered by type so relationship typing can look up an entity's type in O(1).

# FDA-approved therapeutic indications (the 12 conditions BOTOX treats, per the ISI/sitemap).
INDICATIONS = {
    "Chronic Migraine": ["chronic migraine", "chronic migraines"],
    "Adult Spasticity": ["adult spasticity", "spasticity in adults"],
    "Pediatric Spasticity": ["pediatric spasticity", "spasticity in children", "spasticity in people 2 years"],
    "Spasticity": ["spasticity", "muscle stiffness"],
    "Overactive Bladder": ["overactive bladder", "\\bOAB\\b", "urge urinary incontinence", "urinary incontinence"],
    "Cervical Dystonia": ["cervical dystonia", "\\bCD\\b", "abnormal head position"],
    "Blepharospasm": ["blepharospasm"],
    "Strabismus": ["strabismus", "eye muscle problem"],
    "Severe Underarm Sweating": ["severe underarm sweating", "hyperhidrosis", "axillary hyperhidrosis"],
    "Detrusor Overactivity": ["detrusor overactivity", "neurologic condition"],
}

# Symptoms the indications present with (used for TREATS/AFFECTS context).
SYMPTOMS = {
    "Headache": ["headache", "headaches"],
    "Urinary Urgency": ["urgency", "strong need to urinate", "leaking", "wetting accidents"],
    "Urinary Frequency": ["urinating often", "frequency"],
    "Neck Pain": ["neck pain"],
    "Muscle Stiffness": ["muscle stiffness", "increased muscle stiffness"],
}

# Side effects, verbatim from the Important Safety Information.
SIDE_EFFECTS = {
    "Dry Mouth": ["dry mouth"],
    "Injection Site Pain": ["discomfort or pain at the injection site", "injection site pain", "injection-site"],
    "Tiredness": ["tiredness", "fatigue"],
    "Neck Pain (side effect)": ["neck pain"],
    "Double Vision": ["double vision"],
    "Blurred Vision": ["blurred vision"],
    "Drooping Eyelids": ["drooping eyelids", "drooping eyelid", "ptosis"],
    "Dry Eyes": ["dry eyes"],
    "Drooping Eyebrows": ["drooping eyebrows"],
    "Upper Respiratory Infection": ["upper respiratory tract infection", "upper respiratory infection"],
    "Urinary Tract Infection": ["\\bUTI\\b", "urinary tract infection"],
    "Painful Urination": ["painful urination", "dysuria"],
    "Inability to Empty Bladder": ["inability to empty your bladder", "difficulty fully emptying your bladder"],
}

# Safety warnings, the boxed-warning / distant-spread language.
SAFETY_WARNINGS = {
    "Distant Spread of Toxin": ["spread of toxin", "distant spread", "spread of botulinum toxin"],
    "Swallowing/Breathing Problems": [
        "problems swallowing", "trouble swallowing", "trouble breathing",
        "difficulty swallowing", "problems.*breathing", "trouble speaking",
    ],
    "Allergic Reaction": ["allergic reaction", "serious allergic"],
    "Boxed Warning": ["boxed warning"],
    "Serious Side Effects": ["serious side effects", "life threatening", "life-threatening"],
}

# Cost / patient-support programs.
COST_PROGRAMS = {
    "BOTOX Complete": ["BOTOX® Complete", "BOTOX Complete"],
    "Savings Card": ["savings card", "savings program", "co-pay savings", "copay savings"],
    "Insurance Coverage": ["insurance coverage", "commercial insurance", "private insurance", "coverage"],
    "Out-of-Pocket Cost": ["out-of-pocket", "out of pocket", "co-pay", "copay", "co-insurance", "deductible"],
}

# Body areas affected / injected.
BODY_AREAS = {
    "Bladder": ["bladder"],
    "Neck": ["\\bneck\\b"],
    "Eye": ["\\beye\\b", "eyes", "eyelid"],
    "Head": ["\\bhead\\b"],
    "Underarm": ["underarm", "armpit", "axillary"],
    "Muscle": ["muscle", "muscles"],
}

# Who administers / prescribes.
PROVIDERS = {
    "BOTOX Specialist": ["BOTOX® specialist", "BOTOX specialist", "specialist"],
    "Healthcare Provider": ["healthcare provider", "doctor", "physician", "prescriber"],
}

# The drug itself (the hub node most things connect to).
DRUGS = {
    "BOTOX": ["BOTOX®", "\\bBOTOX\\b", "onabotulinumtoxinA"],
    "BOTOX Cosmetic": ["BOTOX® Cosmetic", "BOTOX Cosmetic"],
}

# type name -> lexicon. Order matters only for display; lookup is by canonical label.
LEXICONS: dict[str, dict[str, list[str]]] = {
    "Drug": DRUGS,
    "Indication": INDICATIONS,
    "Symptom": SYMPTOMS,
    "SideEffect": SIDE_EFFECTS,
    "SafetyWarning": SAFETY_WARNINGS,
    "CostProgram": COST_PROGRAMS,
    "BodyArea": BODY_AREAS,
    "Provider": PROVIDERS,
}

# Precompiled (type, canonical, compiled_regex) for a single pass over each chunk.
_COMPILED: list[tuple[str, str, re.Pattern]] = []
for _type, _lex in LEXICONS.items():
    for _canon, _forms in _lex.items():
        _pat = re.compile(r"\b(?:" + "|".join(_forms) + r")", re.IGNORECASE)
        _COMPILED.append((_type, _canon, _pat))

# canonical label -> type, for relationship typing.
_TYPE_OF: dict[str, str] = {canon: t for t, lex in LEXICONS.items() for canon in lex}


def _node_id(canonical: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", canonical.lower()).strip("_")


# Relationship type is decided by the (typeA, typeB) pair. Symmetric pairs are normalised so
# (Drug, Indication) and (Indication, Drug) yield the same directed TREATS edge (Drug->Indication).
def _relation(type_a: str, canon_a: str, type_b: str, canon_b: str) -> tuple[str, str, str] | None:
    """Return (src_canonical, dst_canonical, REL) or None if the pair isn't a meaningful link."""
    pair = frozenset((type_a, type_b))

    if pair == frozenset(("Drug", "Indication")):
        drug = canon_a if type_a == "Drug" else canon_b
        ind = canon_b if type_a == "Drug" else canon_a
        return (drug, ind, "TREATS")
    if pair == frozenset(("Drug", "SideEffect")):
        drug = canon_a if type_a == "Drug" else canon_b
        se = canon_b if type_a == "Drug" else canon_a
        return (drug, se, "HAS_SIDE_EFFECT")
    if pair == frozenset(("Drug", "SafetyWarning")):
        drug = canon_a if type_a == "Drug" else canon_b
        w = canon_b if type_a == "Drug" else canon_a
        return (drug, w, "WARNS_ABOUT")
    if pair == frozenset(("Indication", "Symptom")):
        ind = canon_a if type_a == "Indication" else canon_b
        sym = canon_b if type_a == "Indication" else canon_a
        return (ind, sym, "PRESENTS_WITH")
    if pair == frozenset(("Indication", "BodyArea")):
        ind = canon_a if type_a == "Indication" else canon_b
        ba = canon_b if type_a == "Indication" else canon_a
        return (ind, ba, "AFFECTS")
    if pair == frozenset(("Indication", "CostProgram")):
        ind = canon_a if type_a == "Indication" else canon_b
        cp = canon_b if type_a == "Indication" else canon_a
        return (cp, ind, "COVERS")
    if pair == frozenset(("CostProgram", "CostProgram")) and canon_a != canon_b:
        return (canon_a, canon_b, "RELATED_TO")
    if pair == frozenset(("Indication", "Provider")):
        ind = canon_a if type_a == "Indication" else canon_b
        pr = canon_b if type_a == "Indication" else canon_a
        return (pr, ind, "TREATS_WITH_BOTOX")
    if pair == frozenset(("SideEffect", "BodyArea")):
        se = canon_a if type_a == "SideEffect" else canon_b
        ba = canon_b if type_a == "SideEffect" else canon_a
        return (se, ba, "AFFECTS")
    return None


def extract(chunks_path: Path | None = None, out_path: Path | None = None) -> dict:
    chunks_path = chunks_path or (PROCESSED / "chunks.jsonl")
    out_path = out_path or (PROCESSED / "graph.json")
    if not chunks_path.exists():
        raise FileNotFoundError(f"{chunks_path} not found, run `python -m app.ingest.chunk` first.")

    # node canonical -> {type, sources:set}
    nodes: dict[str, dict] = {}
    # (src,dst,rel) -> sources:set
    edges: dict[tuple[str, str, str], set] = defaultdict(set)
    # chunk_id -> set of (type, canonical) it mentions (for the :MENTIONS graph + expansion)
    chunk_mentions: dict[str, list[str]] = {}

    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ch = json.loads(line)
        cid = ch["id"]
        found: list[tuple[str, str]] = []
        for etype, canon, pat in _COMPILED:
            if pat.search(ch["text"]):
                found.append((etype, canon))
                node = nodes.setdefault(canon, {"type": etype, "sources": set()})
                node["sources"].add(cid)
        # record mentions (deduped canonical list) for this chunk
        chunk_mentions[cid] = sorted({canon for _t, canon in found})

        # Relationships from co-occurrence within this chunk.
        for (ta, ca), (tb, cb) in combinations(found, 2):
            if ca == cb:
                continue
            rel = _relation(ta, ca, tb, cb)
            if rel:
                src, dst, r = rel
                edges[(src, dst, r)].add(cid)

    graph = {
        "nodes": [
            {"id": _node_id(canon), "label": canon, "type": data["type"],
             "sources": sorted(data["sources"])}
            for canon, data in sorted(nodes.items())
        ],
        "edges": [
            {"src": _node_id(s), "dst": _node_id(d), "rel": r, "sources": sorted(srcs)}
            for (s, d, r), srcs in sorted(edges.items())
        ],
        "chunk_mentions": {cid: [_node_id(c) for c in cs] for cid, cs in chunk_mentions.items()},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(graph, indent=1, ensure_ascii=False))
    print(
        f"extracted {len(graph['nodes'])} entities, {len(graph['edges'])} relationships "
        f"from {len(chunk_mentions)} chunks -> {out_path}"
    )
    return graph


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.parse_args()
    extract()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
