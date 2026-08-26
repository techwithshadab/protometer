"""Tiny SVG builder for the AMLGuard architecture diagrams.

One place for the palette (from _tokens.json), text wrapping (so nothing overflows a box), bands,
cards, edge-to-edge arrows, numbered flow badges, and a small line-icon set. Every diagram source
(diagram_*.py) imports this, so the look is identical and the palette is defined once.

No external deps; renders plain SVG 1.1 that GitHub and browsers display natively.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = json.loads((HERE / "_tokens.json").read_text())
FONT = T["font"]

# Approximate character width per font-size for wrapping (Segoe UI-ish, proportional). Slightly
# generous so wrapped text never touches the card edge.
_CHAR_W = 0.545


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def wrap(text: str, max_width: float, size: float) -> list[str]:
    """Greedy word-wrap to fit `max_width` px at font `size`. Respects explicit '\n'."""
    out: list[str] = []
    for para in str(text).split("\n"):
        words, line = para.split(" "), ""
        for w in words:
            trial = (line + " " + w).strip()
            if len(trial) * size * _CHAR_W <= max_width or not line:
                line = trial
            else:
                out.append(line)
                line = w
        out.append(line)
    return out


def text_width(s: str, size: float) -> float:
    return len(s) * size * _CHAR_W


class SVG:
    def __init__(self, w: float, h: float):
        self.w, self.h, self.parts = w, h, []

    def raw(self, s: str):
        self.parts.append(s)

    # ── primitives ────────────────────────────────────────────────────────────────────────────
    def rect(self, x, y, w, h, rx=10, fill="#fff", stroke="none", sw=1, dash=None, opacity=1):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} opacity="{opacity}"/>')

    def text(self, x, y, s, size=11, fill=None, weight="normal", anchor="start",
             spacing=None, italic=False):
        fill = fill or T["ink"]
        ls = f' letter-spacing="{spacing}"' if spacing else ""
        it = ' font-style="italic"' if italic else ""
        self.raw(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
                 f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"{ls}{it}>{esc(s)}</text>')

    def multiline(self, x, y, lines, size=10.5, fill=None, lh=1.32, weight="normal", anchor="start"):
        for i, ln in enumerate(lines):
            self.text(x, y + i * size * lh, ln, size=size, fill=fill, weight=weight, anchor=anchor)
        return y + max(len(lines) - 1, 0) * size * lh  # baseline of the last line

    # ── composites ────────────────────────────────────────────────────────────────────────────
    def band(self, x, y, w, h, band, title, title_size=14, num=None):
        """A pastel layer container with a colored bold header (optionally a leading number badge)."""
        c = T["bands"][band]
        self.rect(x, y, w, h, rx=14, fill=c["fill"], stroke=c["border"], sw=1.4)
        tx = x + 20
        if num is not None:
            self.badge(x + 20, y + 20, num)
            tx = x + 44
        self.text(tx, y + 25, title, size=title_size, fill=c["head"], weight="bold")

    def card(self, x, y, w, h, title=None, body=None, band=None, icon=None,
             title_size=12.5, body_size=10.5, center=False):
        """White rounded card with optional icon, semibold title, wrapped body.

        center=False (default) lays the icon at the left and the title to its right on the same line;
        center=True stacks a centered icon over a centered title (for grid cards)."""
        head = T["bands"][band]["head"] if band else T["ink"]
        self.rect(x, y, w, h, rx=10, fill=T["card"], stroke=T["hair"], sw=1.2)
        pad = 12
        if center:
            cy = y + pad + 6
            if icon:
                self.icon(icon, x + w / 2 - 9, y + pad, head)
                cy = y + pad + 30
            if title:
                self.text(x + w / 2, cy, title, size=title_size, fill=head, weight="600", anchor="middle")
                cy += title_size + 3
            if body:
                self.multiline(x + w / 2, cy + 2, wrap(body, w - 2 * pad, body_size),
                               size=body_size, fill=T["inkSoft"], anchor="middle")
            return
        # left-aligned: icon on the left, title on the same baseline to its right
        tx = x + pad
        if icon:
            self.icon(icon, x + pad, y + pad, head)
            tx = x + pad + 24                    # clear of the 18px icon + gap
        cy = y + pad + title_size
        if title:
            self.text(tx, cy, title, size=title_size, fill=head, weight="600")
            cy += 5
        if body:
            # body wraps under the full width (below title), never under the icon column
            lines = wrap(body, w - 2 * pad, body_size)
            self.multiline(x + pad, cy + body_size, lines, size=body_size, fill=T["inkSoft"])

    def badge(self, cx, cy, n, r=11):
        """Numbered flow badge: filled accent circle with a white number."""
        self.raw(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{T["accent"]}"/>')
        self.text(cx, cy + r * 0.38, str(n), size=r * 1.05, fill="#fff", weight="bold",
                  anchor="middle")

    def _head(self, x, y, dx, dy, color, size=9):
        """Draw a filled triangular arrowhead at (x,y) pointing in direction (dx,dy). Explicit
        polygon (not an SVG marker) so it renders IDENTICALLY in every viewer — browsers, GitHub,
        and macOS Preview/Quick Look, which mishandle marker scaling."""
        import math
        a = math.atan2(dy, dx)
        # tip at (x,y); two back corners fanned out by ~26 degrees, `size` long.
        p1 = (x - size * math.cos(a - 0.45), y - size * math.sin(a - 0.45))
        p2 = (x - size * math.cos(a + 0.45), y - size * math.sin(a + 0.45))
        self.raw(f'<polygon points="{x:.1f},{y:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" '
                 f'fill="{color}"/>')

    def arrow(self, x1, y1, x2, y2, dashed=False, label=None, color=None, lx=None, ly=None):
        color = color or T["ink"]
        d = ' stroke-dasharray="5 4"' if dashed else ""
        import math
        a = math.atan2(y2 - y1, x2 - x1)
        # shorten the shaft so it meets the BACK of the head, not the tip (no double-drawn overlap)
        bx, by = x2 - 8 * math.cos(a), y2 - 8 * math.sin(a)
        self.raw(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
                 f'stroke="{color}" stroke-width="1.7"{d}/>')
        self._head(x2, y2, x2 - x1, y2 - y1, color)
        if label:
            self.text(lx if lx is not None else (x1 + x2) / 2,
                      ly if ly is not None else (y1 + y2) / 2 - 5, label,
                      size=9.5, fill=T["inkSoft"], anchor="middle")

    def elbow(self, x1, y1, x2, y2, dashed=False, color=None):
        """Right-angle connector: VERTICAL first (from (x1,y1) down/up to y2), then HORIZONTAL across
        to (x2,y2), ending in an explicit arrowhead pointing horizontally."""
        color = color or T["ink"]
        d = ' stroke-dasharray="5 4"' if dashed else ""
        hx = x2 - (8 if x2 > x1 else -8)          # stop short so the head sits at the end cleanly
        self.raw(f'<path d="M{x1:.1f},{y1:.1f} L{x1:.1f},{y2:.1f} L{hx:.1f},{y2:.1f}" '
                 f'fill="none" stroke="{color}" stroke-width="1.7"{d}/>')
        self._head(x2, y2, (1 if x2 > x1 else -1), 0, color)

    def elbow_hv(self, x1, y1, x2, y2, dashed=False, color=None):
        """Right-angle connector: HORIZONTAL first (from (x1,y1) across to x2), then VERTICAL to
        (x2,y2), ending in an explicit arrowhead pointing DOWN or UP into the target. Use this when a
        source exits to the side and the target sits above/below it (reads more naturally than
        vertical-first for decision branches)."""
        color = color or T["ink"]
        d = ' stroke-dasharray="5 4"' if dashed else ""
        vy = y2 - (8 if y2 > y1 else -8)          # stop short so the head sits at the end cleanly
        self.raw(f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y1:.1f} L{x2:.1f},{vy:.1f}" '
                 f'fill="none" stroke="{color}" stroke-width="1.7"{d}/>')
        self._head(x2, y2, 0, (1 if y2 > y1 else -1), color)

    # ── icons (18px, simple line glyphs, colored to the band head) ─────────────────────────────
    def icon(self, name, x, y, color):
        g = _ICONS.get(name)
        if not g:
            return
        self.raw(f'<g transform="translate({x:.1f},{y:.1f})" fill="none" stroke="{color}" '
                 f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{g}</g>')

    def render(self) -> str:
        # No <marker> defs: arrowheads are explicit polygons (see _head), which every renderer draws
        # the same — including macOS Preview/Quick Look, which mishandle SVG markers. A <style> block
        # sets the default font so text is consistent even where the presentation attribute is ignored.
        style = f'<style>text{{font-family:{FONT};}}</style>'
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w:.0f} {self.h:.0f}" '
                f'width="{self.w:.0f}" height="{self.h:.0f}">'
                f'{style}'
                f'<rect width="{self.w:.0f}" height="{self.h:.0f}" fill="{T["page"]}"/>'
                f'{"".join(self.parts)}</svg>')

    def save(self, path: str | Path):
        Path(path).write_text(self.render())


# 18x18 line icons (viewBox 0..18). Kept minimal and legible at small sizes.
_ICONS = {
    "globe": '<circle cx="9" cy="9" r="7"/><path d="M2 9h14M9 2c2.5 2 2.5 12 0 14M9 2c-2.5 2-2.5 12 0 14"/>',
    "database": '<ellipse cx="9" cy="4.5" rx="6" ry="2.3"/><path d="M3 4.5v9c0 1.3 2.7 2.3 6 2.3s6-1 6-2.3v-9"/><path d="M3 9c0 1.3 2.7 2.3 6 2.3s6-1 6-2.3"/>',
    "search": '<circle cx="8" cy="8" r="5"/><path d="M12 12l4 4"/>',
    "brain": '<path d="M7 3a3 3 0 00-3 3 3 3 0 00-1 5 3 3 0 003 4 3 3 0 001 0V3zM11 3a3 3 0 013 3 3 3 0 011 5 3 3 0 01-3 4 3 3 0 01-1 0V3z"/>',
    "gear": '<circle cx="9" cy="9" r="2.6"/><path d="M9 2v2M9 14v2M2 9h2M14 9h2M4 4l1.5 1.5M12.5 12.5L14 14M14 4l-1.5 1.5M5.5 12.5L4 14"/>',
    "shield": '<path d="M9 2l6 2.4v4c0 4-2.7 6.4-6 7.6-3.3-1.2-6-3.6-6-7.6v-4z"/>',
    "chat": '<path d="M3 4h12v8H8l-3 3v-3H3z"/>',
    "book": '<path d="M3 3h5a2 2 0 012 2v10a2 2 0 00-2-1.5H3zM15 3h-5a2 2 0 00-2 2v10a2 2 0 012-1.5h5z"/>',
    "clock": '<circle cx="9" cy="9" r="7"/><path d="M9 5v4l3 2"/>',
    "user": '<circle cx="9" cy="6" r="3"/><path d="M3.5 16a5.5 5.5 0 0111 0"/>',
    "doc": '<path d="M5 2h6l3 3v11H5zM11 2v3h3"/>',
    "route": '<circle cx="4" cy="4" r="2"/><circle cx="14" cy="14" r="2"/><path d="M6 4h5a3 3 0 013 3v5"/>',
    "list": '<path d="M6 4h9M6 9h9M6 14h9M3 4h.01M3 9h.01M3 14h.01"/>',
    "lock": '<rect x="4" y="8" width="10" height="7" rx="1.5"/><path d="M6 8V6a3 3 0 016 0v2"/>',
    "chart": '<path d="M3 15V3M3 15h12M6 12v-3M9.5 12V7M13 12V5"/>',
    "bell": '<path d="M9 3a4 4 0 014 4c0 4 2 5 2 5H3s2-1 2-5a4 4 0 014-4zM7.5 15a1.6 1.6 0 003 0"/>',
    "retry": '<path d="M14 5a6 6 0 10.6 6"/><path d="M14 2v3h-3"/>',
    "eye": '<path d="M1.5 9S4 4 9 4s7.5 5 7.5 5-2.5 5-7.5 5S1.5 9 1.5 9z"/><circle cx="9" cy="9" r="2.2"/>',
    "key": '<circle cx="6" cy="9" r="3"/><path d="M9 9h7M14 9v3M16 9v2"/>',
    "cloud": '<path d="M5 13a3 3 0 010-6 4 4 0 017.7-1A3.2 3.2 0 0114 13z"/>',
    "cpu": '<rect x="5" y="5" width="8" height="8" rx="1.5"/><path d="M8 2v3M10 2v3M8 13v3M10 13v3M2 8h3M2 10h3M13 8h3M13 10h3"/>',
    "flow": '<rect x="3" y="3" width="5" height="4" rx="1"/><rect x="10" y="11" width="5" height="4" rx="1"/><path d="M5.5 7v3a2 2 0 002 2h2.5"/>',
    "split": '<circle cx="4" cy="9" r="2"/><circle cx="14" cy="4" r="2"/><circle cx="14" cy="14" r="2"/><path d="M6 9l6-4M6 9l6 4"/>',
    "layers": '<path d="M9 2l7 3.5-7 3.5-7-3.5zM2 9l7 3.5L16 9M2 12.5L9 16l7-3.5"/>',
}
