# Architecture diagrams

Designed figures for the BOTOX® Assistant, in the same system as the AMLGuard diagrams: each is an
HTML source under `src/` styled by the shared `src/_design.css` (navy botox.com palette, Geist font,
a reusable component set), rendered to a 2× PNG with headless Chrome.

| PNG | Source | Used in |
|---|---|---|
| `architecture-overview.png` | `src/architecture-overview.html` | README, `../architecture.md` (hero) |
| `system-topology.png` | `src/system-topology.html` | `../architecture.md` §1 |
| `ingest-pipeline.png` | `src/ingest-pipeline.html` | `../architecture.md` §2a |
| `query-pipeline.png` | `src/query-pipeline.html` | `../architecture.md` §2b |
| `graphrag-retrieval.png` | `src/graphrag-retrieval.html` | `../architecture.md` §3 |
| `protection-boundary.png` | `src/protection-boundary.html` | `../architecture.md` §4 |
| `egress-guards.png` | `src/egress-guards.html` | `../architecture.md` §5 |
| `model-tiering.png` | `src/model-tiering.html` | `../architecture.md` §6 |
| `observability.png` | `src/observability.html` | `../architecture.md` §7 |

## Regenerate

```sh
sh docs/diagrams/src/render.sh      # requires Google Chrome
```

Each `render` line names the figure and its logical (1×) canvas size; the script renders at
`--force-device-scale-factor=2`, so a 1440×788 canvas becomes a 2880×1576 PNG. Edit the `.html`
(and `_design.css` for shared styling), then re-run to refresh the committed PNGs.
