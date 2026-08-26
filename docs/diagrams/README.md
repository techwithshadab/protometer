# Diagrams

Reference-style architecture diagrams, generated as SVG so they render on GitHub and in any browser,
scale without blur, and stay diffable. All of them share one palette — see [PALETTE.md](PALETTE.md).

| File | Diagram | Source |
|---|---|---|
| `pipeline-strip.svg` | compact horizontal pipeline (README header) | `diagram_strip.py` |
| `architecture-pipeline.svg` | end-to-end 7-stage pipeline + observability | `diagram_pipeline.py` |
| `architecture-concerns.svg` | grouped by concern (deck view) | `diagram_concerns.py` |
| `model-selection.svg` | live-chat model selection | `diagram_model_selection.py` |

## Regenerate

```bash
cd docs/diagrams
for d in diagram_*.py; do python3 "$d"; done
```

Each `diagram_*.py` writes its `.svg` next to it. No dependencies — `_svglib.py` emits plain SVG 1.1.

## Editing

- The palette (colors, fonts) lives once in `_tokens.json`; change it there and regenerate.
- `_svglib.py` has the shared primitives: `band`, `card`, `badge`, `arrow`, `elbow`, `icon`, and a
  `wrap()` that fits text to a box so nothing overflows. Add icons to the `_ICONS` dict.
- Keep the color scheme consistent with `_tokens.json` for any new diagram, flowchart, or table image
  so the whole doc set reads as one system.

The palette is deliberately the single source of truth: define a value once, use it everywhere.
