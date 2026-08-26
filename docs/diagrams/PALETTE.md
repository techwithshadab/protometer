# Diagram conventions & color scheme

All generated diagrams, flowcharts, and doc tables use this single palette (adapted from the
reference architecture style: soft pastel layer bands, saturated headers, white rounded cards, mono
line icons, a blue accent for flow). Keep it consistent across every diagram so the docs read as one
system.

## Palette

| Token | Hex | Use |
|---|---|---|
| `ink` | `#1F2937` | primary text |
| `ink-soft` | `#5A6784` | secondary/label text |
| `page` | `#FFFFFF` | diagram background |
| `card` | `#FFFFFF` | inner card fill |
| `hair` | `#D8DEEC` | card / band borders |
| **Layer bands (fill / header / border)** | | |
| green | `#EAF3EC` / `#3E9E6B` / `#CFE6D6` | client / input / governance layers |
| purple | `#EFEAF7` / `#7A5AB6` / `#DED3EF` | orchestration / control |
| blue | `#E8F0FB` / `#3D74C9` / `#CFE0F5` | agent / core compute |
| teal | `#E7F4F5` / `#2FB6C4` / `#C7E8EC` | tools / integrations |
| amber | `#FBF3E2` / `#E0A43B` / `#F0E2C0` | memory / knowledge |
| red | `#FBECEC` / `#D25C5C` / `#F2D6D6` | monitoring / risk |
| slate | `#EEF1F6` / `#5A6784` / `#DBE1EC` | foundation / infra (cross-cutting) |
| **Accents** | | |
| accent (flow) | `#2F6FDB` | numbered flow badges, sync arrows |
| accent-soft | `#DCE7FA` | badge fill / highlight |

Sync / data flow = solid arrow. Async / event flow = dashed arrow.

## Fonts

`Segoe UI, system-ui, -apple-system, Helvetica, Arial, sans-serif` for everything; the diagram
**title** is semibold, layer **headers** are bold, card **titles** are semibold, card **body** is
regular in `ink-soft`. Sizes: title 22, layer header 14, card title 12.5, body 10.5.

## Rules that keep them clean

- Every text block fits inside its box with padding; wrap long lines rather than overflow.
- Cards in a row are equal width and top-aligned; bands are equal padding.
- Arrows connect box edges (not centers), never cross a box, and carry an optional label.
- Icons are simple monochrome line glyphs in the layer's header color.

The machine-readable version of these tokens lives in `docs/diagrams/_tokens.json` (used by the
diagram sources) so a value is defined once.
