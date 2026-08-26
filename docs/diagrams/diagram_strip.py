"""Compact horizontal pipeline strip for the README top: corpus -> 7 stages -> analyst, with a
dashed 'protected zone (tokens only)' bracket over the tokenized middle.

Regenerate:  python docs/diagrams/diagram_strip.py
"""
from _svglib import SVG, T

W, H = 1180, 176
s = SVG(W, H)

# stages: (label, sublabel, band): corpus + 7 stages + analyst as terminal chips
stages = [
    ("Ingest",   "discover + tokenize", "green"),
    ("Train",    "RF on tokens",        "blue"),
    ("Embed",    "local vectors",       "teal"),
    ("Retrieve", "tokens only",         "teal"),
    ("Infer",    "model on tokens",     "purple"),
    ("Egress",   "guardrail + leakcheck","red"),
    ("Present",  "role-gated unprotect","green"),
]

# Layout, computed up front so it is exactly justified: one row of 9 chips,
#   [corpus]  <GROUP_GAP>  ( <PAD> stage stage ... stage <PAD> )  <GROUP_GAP>  [analyst]
# The dashed zone bracket hugs the 7 stages with equal PAD inside; the two GROUP_GAPs (corpus->zone
# and zone->analyst) are equal, so left and right spacing match.
STAGE_GAP = 12          # between adjacent stages
GROUP_GAP = 40          # corpus<->zone and zone<->analyst (the arrow spans this)
PAD = 14                # bracket inner padding around the stage row
MARGIN = 30             # page margin at far left / right
bh = 54
y = 88

# Chip width so the whole row fits W exactly.
n_stage = len(stages)
# widths: 2 terminal chips + n_stage stage chips, all equal `bw`.
# horizontal budget consumed by gaps/padding (everything that is NOT a chip):
gaps = 2 * GROUP_GAP + 2 * PAD + STAGE_GAP * (n_stage - 1)
bw = (W - 2 * MARGIN - gaps) / (n_stage + 2)
cy = y + bh / 2

def chip(x, label, sub, band, terminal=False):
    fill = T["card"] if terminal else T["bands"][band]["fill"]
    stroke = T["hair"] if terminal else T["bands"][band]["border"]
    s.rect(x, y, bw, bh, rx=9, fill=fill, stroke=stroke, sw=1.3)
    head = T["ink"] if terminal else T["bands"][band]["head"]
    s.text(x + bw / 2, y + 24, label, size=12.5, fill=head, weight="bold", anchor="middle")
    s.text(x + bw / 2, y + 41, sub, size=9, fill=T["inkSoft"], anchor="middle")

# corpus
cx = MARGIN
chip(cx, "corpus", "clear input", "green", terminal=True)
corpus_right = cx + bw

# zone bracket geometry
zx0 = corpus_right + GROUP_GAP
first_stage_x = zx0 + PAD
# stages
x = first_stage_x
prev_right = None
for label, sub, band in stages:
    chip(x, label, sub, band)
    if prev_right is not None:
        s.arrow(prev_right, cy, x, cy)          # arrow between adjacent stages
    prev_right = x + bw
    x += bw + STAGE_GAP
last_stage_right = prev_right                   # right edge of the last stage
zx1 = last_stage_right + PAD

# analyst
ax = zx1 + GROUP_GAP
chip(ax, "analyst", "plaintext", "green", terminal=True)

# arrows crossing the zone boundary: corpus -> first stage, last stage -> analyst
s.arrow(corpus_right, cy, first_stage_x, cy)
s.arrow(last_stage_right, cy, ax, cy)

# ── dashed protected-zone bracket, hugging the stage row with equal PAD each side ───────────────
s.rect(zx0, y - 30, zx1 - zx0, bh + 48, rx=12, fill="none", stroke=T["accent"], sw=1.4, dash="6 5")
s.rect((zx0 + zx1) / 2 - 118, y - 40, 236, 20, rx=7, fill=T["accentSoft"], stroke="none")
s.text((zx0 + zx1) / 2, y - 26, "PROTECTED ZONE · TOKENS ONLY", size=10.5,
       fill=T["accent"], weight="bold", anchor="middle")

s.save(__file__.replace("diagram_strip.py", "pipeline-strip.svg"))
print("wrote pipeline-strip.svg")
