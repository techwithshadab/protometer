"""AMLGuard grouped by concern — the deck's "Protected Pipeline" view.

Four concern-columns left to right: Enterprise Data -> the dashed Protected Pipeline (tokens only)
-> Governance -> State & Observability.
Regenerate:  python docs/diagrams/diagram_concerns.py
"""
from _svglib import SVG, T, wrap

W, H = 1180, 620
s = SVG(W, H)

s.text(W / 2, 38, "AMLGuard Architecture by Concern", size=22, weight="bold", anchor="middle")
s.text(W / 2, 61, "One protected zone, with governance and durable state as its neighbours",
       size=12.5, fill=T["inkSoft"], anchor="middle")

TOP = 96
COL_H = 470

# ── column 1: Enterprise Data (green) ──────────────────────────────────────────────────────────
c1x, c1w = 40, 200
s.band(c1x, TOP, c1w, 250, "green", "Enterprise Data")
s.text(c1x + 20, TOP + 46, "clear, left of ingest", size=10, fill=T["inkSoft"], italic=True)
inputs = ["Parties + Profiles", "Transactions", "Alerts + Labels", "Case Narratives", "Approved Content"]
for i, t in enumerate(inputs):
    yy = TOP + 62 + i * 36
    s.rect(c1x + 16, yy, c1w - 32, 28, rx=7, fill=T["card"], stroke=T["hair"], sw=1.1)
    s.icon("database", c1x + 26, yy + 5, T["bands"]["green"]["head"])
    s.text(c1x + 50, yy + 18, t, size=10.5, fill=T["ink"])

# ── column 2: Protected Pipeline (blue band, dashed inner "tokens only" frame) ──────────────────
c2x, c2w = 300, 340
s.band(c2x, TOP, c2w, COL_H, "blue", "Protected Pipeline")
s.rect(c2x + 14, TOP + 44, c2w - 28, COL_H - 60, rx=12, fill="none", stroke=T["accent"], sw=1.4, dash="6 5")
s.rect(c2x + 24, TOP + 34, 128, 20, rx=7, fill=T["accentSoft"], stroke="none")
s.text(c2x + 34, TOP + 48, "TOKENS ONLY", size=10, fill=T["accent"], weight="bold")

stages = [
    (1, "gear",   "Ingest",          "discover + tokenize per scope; leak-verified"),
    (2, "chart",  "Train + Embed",   "RF + graph, SHAP, MLflow; MiniLM to ChromaDB"),
    (3, "route",  "Retrieve",        "surrogate-key open; 2-hop + top-k chunks"),
    (4, "brain",  "Infer",           "Bedrock or local; temp 0, cached, traced"),
    (5, "shield", "Egress + Present", "guardrail scan; role-gated re-identify"),
]
sx, sw = c2x + 28, c2w - 56
sy = TOP + 62
sh = 66
sgap = 18                        # gap between stage cards: room for a visible arrow + padding
for n, icon, title, body in stages:
    s.rect(sx, sy, sw, sh, rx=9, fill=T["card"], stroke=T["hair"], sw=1.1)
    s.badge(sx + 20, sy + 22, n, r=10)
    s.icon(icon, sx + 38, sy + 13, T["bands"]["blue"]["head"])
    s.text(sx + 62, sy + 25, title, size=12, fill=T["bands"]["blue"]["head"], weight="600")
    for i, ln in enumerate(wrap(body, sw - 74, 10)):
        s.text(sx + 62, sy + 42 + i * 13, ln, size=10, fill=T["inkSoft"])
    if n < 5:
        s.arrow(sx + 20, sy + sh + 3, sx + 20, sy + sh + sgap - 4)   # stops short of the next card
    sy += sh + sgap

# ── column 3: Governance (purple) ──────────────────────────────────────────────────────────────
c3x, c3w = 668, 210
s.band(c3x, TOP, c3w, 300, "purple", "Governance")
s.text(c3x + 20, TOP + 46, "enforced every unprotect", size=10, fill=T["inkSoft"], italic=True)
gov = [
    ("lock",  "Role & Territory",   "scope at unprotect"),
    ("shield","Approved-Content",   "groundedness gate"),
    ("user",  "Human-in-the-loop",  "fail-closed analyst"),
    ("doc",   "Audit & Provenance", "trace per request"),
]
for i, (icon, t, b) in enumerate(gov):
    yy = TOP + 60 + i * 58
    s.rect(c3x + 16, yy, c3w - 32, 48, rx=8, fill=T["card"], stroke=T["hair"], sw=1.1)
    s.icon(icon, c3x + 28, yy + 9, T["bands"]["purple"]["head"])
    s.text(c3x + 52, yy + 20, t, size=11, fill=T["ink"], weight="600")
    s.text(c3x + 52, yy + 35, b, size=9.5, fill=T["inkSoft"])

# ── column 4: State & Observability (slate) ────────────────────────────────────────────────────
c4x, c4w = 908, W - 908 - 40
s.band(c4x, TOP, c4w, 300, "slate", "State & Observability")
s.text(c4x + 20, TOP + 46, "durable + provable", size=10, fill=T["inkSoft"], italic=True)
state = [
    ("database", "Durable State Store", "Postgres mirror + protected corpus per scope"),
    ("chart",    "MLflow Registry",     "runs, champion by AP, + corpus fingerprint"),
    ("eye",      "Langfuse Traces",     "PII scrubbed at rest"),
    ("bell",     "Prometheus / Grafana","ops series (scope + corpus fingerprint)"),
]
for i, (icon, t, b) in enumerate(state):
    yy = TOP + 60 + i * 58
    s.rect(c4x + 16, yy, c4w - 32, 48, rx=8, fill=T["card"], stroke=T["hair"], sw=1.1)
    s.icon(icon, c4x + 28, yy + 9, T["bands"]["slate"]["head"])
    s.text(c4x + 52, yy + 20, t, size=11, fill=T["ink"], weight="600")
    for j, ln in enumerate(wrap(b, c4w - 76, 9.5)):
        s.text(c4x + 52, yy + 34 + j * 11, ln, size=9.5, fill=T["inkSoft"])

# ── inter-column arrows (data flows left to right) ─────────────────────────────────────────────
midy = TOP + 150
s.arrow(c1x + c1w, midy, c2x, midy)                     # data -> pipeline
s.arrow(c2x + c2w, midy, c3x, midy)                     # pipeline -> governance
s.arrow(c3x + c3w, midy, c4x, midy)                     # governance -> state

# join-keys callout
s.text(c4x + 16, TOP + 322, "All planes join on:", size=10, fill=T["inkSoft"], weight="600")
s.text(c4x + 16, TOP + 338, "run_id · corpus_fingerprint · scope", size=10, fill=T["accent"], weight="600")

s.save(__file__.replace("diagram_concerns.py", "architecture-concerns.svg"))
print("wrote architecture-concerns.svg")
