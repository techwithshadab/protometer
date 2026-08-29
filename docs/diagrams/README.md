# Diagrams

Every figure the docs embed, rendered from a committed source so a diagram can never drift
from something regenerable.

| Figure | Embedded in | Shows |
|---|---|---|
| `architecture-overview.png` | README, architecture.md | The full system: plaintext sources, ingest-time protection, the trust boundary, the tokens-only protected zone, the egress guard chain, the role gate, and the assurance plane. |
| `pipeline-strip.png` | README (top) | The same story as a compact strip: sources → ingest → tokens-only band → role-gated presentation. |
| `architecture-concerns.png` | architecture.md | The identical system regrouped by concern: enterprise data, the protected pipeline, governance, state & observability. |
| `model-selection.png` | architecture.md | Live-chat model precedence: explicit override → hosted when AWS credentials exist → local Ollama at $0. |

## How they are built

Each figure is a small HTML page under [`src/`](src/) sharing one design system
([`src/_design.css`](src/_design.css): the palette, column/card/step components, trust-boundary
and plane styles). Rendering is a headless-Chrome screenshot at 2x:

```bash
sh docs/diagrams/src/render.sh     # regenerates every PNG in this directory
```

Conventions the design system enforces:

- One palette everywhere (cream ground, ink text, purple accents, red for trust boundaries).
- Straight orthogonal connectors only; arrowheads are explicit polygons.
- Trust boundaries are red dashed lines; plaintext zones are grey dashed containers.
- Columns fill their full height (content is distributed, not top-packed).
- No footers, logos, or slide chrome inside a figure; the embedding doc provides context.

To change a figure, edit its `src/*.html`, re-run `render.sh`, and commit both the source and
the PNG together.
