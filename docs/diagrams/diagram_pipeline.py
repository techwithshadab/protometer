"""End-to-end AMLGuard pipeline — the 7 protected stages, trust boundary, and observability.

Reference-style: pastel stage bands, white detail, numbered flow badges, sync/async arrows.
Regenerate:  python docs/diagrams/diagram_pipeline.py
"""
from _svglib import SVG, T, wrap

W, H = 1180, 1074
s = SVG(W, H)

# ── title ────────────────────────────────────────────────────────────────────────────────────
s.text(W / 2, 40, "AMLGuard Protected Pipeline, End to End", size=22, weight="bold", anchor="middle")
s.text(W / 2, 64, "Plaintext exists only left of Ingestion and right of Presentation · everything between is tokens",
       size=12.5, fill=T["inkSoft"], anchor="middle")

# Column geometry: a main flow column on the left, an observability/cross-cutting column on the right.
# MAIN_X leaves room so inter-stage flow labels (placed to the RIGHT of the arrow) never overflow.
MAIN_X, MAIN_W = 60, 720
OBS_X = 820
OBS_W = W - OBS_X - 40

# ── trust-boundary frame around the protected zone ─────────────────────────────────────────────
# tb_h sized to the flow: 7 bands (six at 76, Inference at 80) + six 34px gaps + a 30px top inset.
tb_y, tb_h = 156, 30 + (6 * 76 + 80) + 6 * 34 + 16
s.rect(MAIN_X - 16, tb_y, MAIN_W + 32, tb_h, rx=16, fill="none", stroke=T["accent"], sw=1.4, dash="7 6")
# label pill sits ON the top frame line, positioned to the RIGHT of the flow/arrow column (which runs
# down x = MAIN_X+24) so the corpus->ingestion arrow never crosses it.
PILL_X = MAIN_X + 70
s.rect(PILL_X, tb_y - 12, 232, 22, rx=8, fill=T["accentSoft"], stroke="none")
s.text(PILL_X + 14, tb_y + 3, "TRUST BOUNDARY · tokens only", size=10.5, fill=T["accent"], weight="bold")

# ── input: clear corpus (above the boundary) ───────────────────────────────────────────────────
y = 96
s.card(MAIN_X, y, MAIN_W, 44, band="green", icon="database",
       title="Clear corpus (synthetic; every identifier is generated)")
s.text(MAIN_X + 40, y + 34, "parties · transactions · alerts · ground truth · case narratives",
       size=10.5, fill=T["inkSoft"])

# helper to place a full-width stage band-card in the main flow, returns its bottom y
def stage(y, num, band, icon, title, body, h=76):
    s.rect(MAIN_X, y, MAIN_W, h, rx=12, fill=T["bands"][band]["fill"], stroke=T["bands"][band]["border"], sw=1.3)
    s.badge(MAIN_X + 24, y + 24, num)
    s.icon(icon, MAIN_X + 44, y + 15, T["bands"][band]["head"])
    s.text(MAIN_X + 70, y + 27, title, size=13, fill=T["bands"][band]["head"], weight="bold")
    for i, ln in enumerate(wrap(body, MAIN_W - 90, 10.5)):
        s.text(MAIN_X + 70, y + 46 + i * 14, ln, size=10.5, fill=T["inkSoft"])
    return y + h


GAP = 34         # vertical space between stage bands: room for the arrow + padding
HEAD = 7         # arrow stops this far above the next box, so the head sits in a clean gap

def vflow(y1, label=None):
    # a full connector living in the GAP below a band: arrow down the badge column, stopping short of
    # the next band so the head has padding; the label sits to its RIGHT, left-anchored.
    s.arrow(MAIN_X + 24, y1 + 4, MAIN_X + 24, y1 + GAP - HEAD)
    if label:
        s.text(MAIN_X + 42, y1 + GAP / 2 + 3, label, size=9.5, fill=T["inkSoft"])
    return y1 + GAP

# 1 Ingestion
s.arrow(MAIN_X + 24, y + 46, MAIN_X + 24, tb_y + 30 - HEAD)
y = stage(tb_y + 30, 1, "green", "search", "Ingestion",
          "Data Discovery classifies each narrative; a roster fills its measured gaps; entities batched "
          "by data element, then Protegrity /protect. No-op tokens redacted, leak-verified.")
y = vflow(y, "protected corpus · one copy per scope (8 scopes)")
# 2 Training
y = stage(y, 2, "blue", "chart", "Training",
          "RandomForest + graph features, fitted PER SCOPE on the protected ledger; SHAP reliance; "
          "logged to the MLflow registry.")
y = vflow(y)
# 3 Embedding
y = stage(y, 3, "teal", "layers", "Embedding",
          "Local MiniLM vectors over the protected narratives into a ChromaDB index (tokens only, no plaintext embedded).")
y = vflow(y)
# 4 Retrieval
y = stage(y, 4, "teal", "route", "Retrieval",
          "A case opens on a surrogate key; a 2-hop transaction network + top-k token chunks form the prompt.")
y = vflow(y, "prompt = tokens + computed candidates")
# 5 Inference
y = stage(y, 5, "purple", "brain", "Inference",
          "Model reasons over tokens: hosted (Bedrock/Claude) when credentials are present, else a local "
          "open-source model (Ollama). Temp 0, cached, spend-capped, every call traced.", h=80)
y = vflow(y)
# 6 Egress
y = stage(y, 6, "red", "shield", "Egress",
          "Semantic Guardrail scan + forbidden-value leak check + groundedness gate. Fail-closed on the analyst path.")
y = vflow(y)
# 7 Presentation
y = stage(y, 7, "green", "user", "Presentation",
          "Role-gated /unprotect. Auditor: nothing · Analyst: structure · Investigator: everything.")

# output card (below the boundary)
oy = tb_y + tb_h + 24
s.arrow(MAIN_X + 24, y + 4, MAIN_X + 24, oy - HEAD)
s.card(MAIN_X, oy, MAIN_W, 40, band="green", icon="eye",
       title="Analyst screen: the only plaintext surface")

# ── observability / cross-cutting column (right) ───────────────────────────────────────────────
# A side panel that measures itself from its rows (so its height always fits its content, and the
# feedback arrow can enter cleanly below the last row). Returns (top, bottom, first_row_y).
def side_panel(y, band, title, subtitle, rows):
    head = T["bands"][band]["head"]
    ry = y + 46
    if subtitle:
        ry += 14
    first = ry
    for icon, t, b in rows:
        lines = wrap(b, OBS_W - 56, 9.5) if b else []
        ry += 24 + 12 * max(len(lines) - 1, 0) + 8
    bottom = ry + 6
    # draw band behind, then content
    s.band(OBS_X, y, OBS_W, bottom - y, band, title, title_size=13)
    if subtitle:
        s.text(OBS_X + 20, y + 40, subtitle, size=9.5, fill=T["inkSoft"], italic=True)
    ry = first
    for icon, t, b in rows:
        s.icon(icon, OBS_X + 18, ry - 8, head)
        s.text(OBS_X + 44, ry, t, size=11, fill=T["ink"], weight="600")
        lines = wrap(b, OBS_W - 56, 9.5) if b else []
        for i, ln in enumerate(lines):
            s.text(OBS_X + 44, ry + 14 + i * 12, ln, size=9.5, fill=T["inkSoft"])
        ry += 24 + 12 * max(len(lines) - 1, 0) + 8
    return y, bottom

_, ob_bottom = side_panel(150, "red", "Observability", "loopback-only, optional", [
    ("chart", "MLflow", "Experiment ledger + model registry: runs, params, metrics, signatures."),
    ("eye", "Langfuse", "Per-generation record: prompt, completion, tokens, cost, latency, scores."),
    ("bell", "Prometheus + Grafana", "Ingest ops as a time-series; PII scrubbed before storage."),
])
_, cc_bottom = side_panel(ob_bottom + 28, "slate", "Cross-cutting", None, [
    ("key", "Key rotation", "/reprotect migrates tokens server-side; plaintext never transits the app."),
    ("database", "Postgres", "The app's queryable source of truth (corpus mirror)."),
    ("lock", "Spend + auth rails", "Pre-call spend cap; shared-secret + turn ceiling on billed endpoints."),
])
# (The "planes join on run_id · corpus_fingerprint · scope" note lives on the concerns diagram, which
#  is the one that groups by state & observability; not repeated here to avoid duplication.)

# Every stage feeds the observability plane (dashed = async). Shown as a labelled dashed connector
# in the channel between the flow and the panels; the label sits in the open space below the panels
# so it crosses nothing. The connector runs from just outside the trust-boundary frame to the
# Observability panel's left edge.
tb_right = MAIN_X - 16 + MAIN_W + 32        # trust-boundary frame's right edge
fb_y = 234                                  # aligned with the Langfuse row, a calm mid-panel point
s.arrow(tb_right + 4, fb_y, OBS_X - 2, fb_y, dashed=True)
s.text(OBS_X, cc_bottom + 72, "Every stage feeds observability (async).",
       size=9.5, fill=T["inkSoft"], italic=True)

# ── flow legend ────────────────────────────────────────────────────────────────────────────────
ly = H - 34
s.text(60, ly + 4, "Flow:", size=11, fill=T["inkSoft"], weight="600")
s.arrow(110, ly, 150, ly)
s.text(158, ly + 4, "Sync / data flow", size=10.5, fill=T["inkSoft"])
s.arrow(300, ly, 340, ly, dashed=True)
s.text(348, ly + 4, "Async / event flow", size=10.5, fill=T["inkSoft"])

s.save(__file__.replace("diagram_pipeline.py", "architecture-pipeline.svg"))
print("wrote architecture-pipeline.svg")
