"""Live-chat model selection: the UI picks a model at request time so a fork runs with
or without a cloud account. A decision flow in the shared reference style.

Regenerate:  python docs/diagrams/diagram_model_selection.py
"""
from _svglib import SVG, T, wrap

W, H = 1000, 570
s = SVG(W, H)

s.text(W / 2, 38, "Live-Chat Model Selection", size=22, weight="bold", anchor="middle")
s.text(W / 2, 61, "The UI picks a model per request, so the demo runs with or without a cloud account",
       size=12.5, fill=T["inkSoft"], anchor="middle")

# helper: a decision diamond
def diamond(cx, cy, w, h, label):
    s.raw(f'<path d="M{cx:.1f},{cy - h/2:.1f} L{cx + w/2:.1f},{cy:.1f} '
          f'L{cx:.1f},{cy + h/2:.1f} L{cx - w/2:.1f},{cy:.1f} Z" '
          f'fill="{T["accentSoft"]}" stroke="{T["accent"]}" stroke-width="1.4"/>')
    for i, ln in enumerate(wrap(label, w - 30, 10.5)):
        s.text(cx, cy - (len(wrap(label, w - 30, 10.5)) - 1) * 7 + i * 14 + 4, ln,
               size=10.5, fill=T["ink"], weight="600", anchor="middle")

# helper: an outcome card (a resolved model)
def model_card(x, y, w, band, icon, title, body):
    h = 84
    s.rect(x, y, w, h, rx=10, fill=T["bands"][band]["fill"], stroke=T["bands"][band]["border"], sw=1.4)
    s.icon(icon, x + 16, y + 15, T["bands"][band]["head"])
    s.text(x + 42, y + 27, title, size=12.5, fill=T["bands"][band]["head"], weight="bold")
    for i, ln in enumerate(wrap(body, w - 30, 10)):
        s.text(x + 16, y + 46 + i * 13, ln, size=10, fill=T["inkSoft"])
    return h

# start
sx, sy = 90, 165
s.rect(sx - 70, sy - 20, 140, 40, rx=20, fill=T["card"], stroke=T["hair"], sw=1.3)
s.icon("chat", sx - 58, sy - 9, T["accent"])
s.text(sx - 30, sy + 4, "Live turn", size=11.5, fill=T["ink"], weight="600")

# decision 1: explicit override?
d1x, d1y = 320, 165
DHW = 85                                   # diamond half-width (right/left points)
BOX_X = 560
s.arrow(sx + 70, sy, d1x - DHW, d1y)       # Live turn -> diamond 1 (into its left point)
diamond(d1x, d1y, 170, 90, "AMLGUARD_UI_MODEL set?")

BOXH = 84
# YES -> override card. The box is vertically CENTRED on the diamond, so the arrow is a single
# STRAIGHT horizontal line from the diamond's right point into the box's left edge.
s.arrow(d1x + DHW, d1y, BOX_X, d1y)
s.text(d1x + DHW + 12, d1y - 8, "yes", size=10, fill=T["accent"], weight="600")
model_card(BOX_X, d1y - BOXH / 2, 380, "purple", "list", "Use the named model (explicit override)",
           "AMLGUARD_UI_MODEL wins unconditionally · validated against config/models.yaml")

# NO -> decision 2 (straight down into its top point)
d2x, d2y = 320, 330
s.arrow(d1x, d1y + 45, d2x, d2y - 45)
s.text(d1x + 10, (d1y + 45 + d2y - 45) / 2 + 3, "no", size=10, fill=T["inkSoft"], weight="600")
diamond(d2x, d2y, 170, 90, "AWS credentials present?")

# YES -> hosted card, likewise centred on the diamond -> straight horizontal arrow.
s.arrow(d2x + DHW, d2y, BOX_X, d2y)
s.text(d2x + DHW + 12, d2y - 8, "yes", size=10, fill=T["accent"], weight="600")
model_card(BOX_X, d2y - BOXH / 2, 380, "blue", "cloud", "Hosted model: bedrock-sonnet-5",
           "Matches the committed evaluation artifacts · a real billed call")

# NO -> local card. This branch exits the diamond's BOTTOM, so it elbows down-then-right; the box
# sits below the hosted one and its centre is where the elbow's horizontal leg meets it.
lo_cy = d2y + 120
s.elbow(d2x, d2y + 45, BOX_X, lo_cy)
s.text(d2x + 10, d2y + 62, "no", size=10, fill=T["inkSoft"], weight="600")
model_card(BOX_X, lo_cy - BOXH / 2, 380, "teal", "cpu", "Local open-source model: llama3.2 (Ollama)",
           "No cloud account · $0 per turn · make setup-local-model / auto-pull")

# footer band (below all cards): tokenization constant + the health-report note, side by side
fy = 508
s.rect(90, fy, 850, 50, rx=10, fill=T["bands"]["green"]["fill"], stroke=T["bands"]["green"]["border"], sw=1.3)
s.icon("shield", 106, fy + 16, T["bands"]["green"]["head"])
s.text(132, fy + 22, "Tokenization is always Protegrity.", size=11, fill=T["bands"]["green"]["head"], weight="bold")
s.text(330, fy + 22, "Only the reasoning model changes; the protected pipeline is identical either way.",
       size=10.5, fill=T["inkSoft"])
s.text(132, fy + 40, "GET /api/health reports the resolved model, its provider, and whether it is ready.",
       size=10.5, fill=T["inkSoft"])

s.save(__file__.replace("diagram_model_selection.py", "model-selection.svg"))
print("wrote model-selection.svg")
