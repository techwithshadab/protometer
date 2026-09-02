// A tiny, XSS-safe markdown-lite renderer.
//
// The assistant's answer is LLM output over scraped web content, untrusted. We NEVER use
// dangerouslySetInnerHTML. Instead we parse a small, fixed subset and return React elements.
// Because every span of text becomes a React text node, it is escaped by construction; no markup
// in the source string can inject DOM.
//
// Supported:
//   - blank-line-separated paragraphs, with single newlines inside a paragraph as <br/>
//   - bullet lists: lines starting with "-", "*", "•", or "+" (any of the markers a model emits)
//   - NESTED bullets by indentation: a more-indented bullet becomes a child <ul> of the one above
//   - inline **bold**
//   - a standalone "**Heading**" line (a bold-only line) renders as a small heading, since models
//     love to structure answers as "**Side Effects:**" followed by a list
// Everything else renders as literal text. Citations are structured data, not markdown, so there
// is no link/URL sink to sanitize here.

import type { ReactElement } from "react";

const BULLET_RE = /^(\s*)([-*•+])\s+(.*)$/; // indent, marker, content

// Split a line into React nodes, honoring only **bold**. Any stray "*" renders literally.
function renderInline(text: string, keyPrefix: string): (string | ReactElement)[] {
  const nodes: (string | ReactElement)[] = [];
  const re = /\*\*([^*]+)\*\*/g; // paired ** ... ** (non-greedy, no nesting)
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
    nodes.push(<strong key={`${keyPrefix}-b${i++}`}>{m[1]}</strong>);
    last = m.index + m[0].length;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes.length ? nodes : [text];
}

// True if the whole line is a single **bold** span (a pseudo-heading like "**Side Effects:**").
function isBoldHeading(line: string): boolean {
  const t = line.trim();
  const m = t.match(/^\*\*([^*]+)\*\*:?$/);
  return !!m;
}

type BulletItem = { indent: number; content: string };

// Render a run of bullet lines as a nested <ul>. Items whose indent is greater than the current
// level become a child list of the preceding item. Consumes items[start..] at `level` indent and
// returns [element, nextIndex].
function renderList(
  items: BulletItem[],
  start: number,
  level: number,
  keyPrefix: string,
): [ReactElement, number] {
  const children: ReactElement[] = [];
  let i = start;
  let li = 0;
  while (i < items.length && items[i].indent >= level) {
    if (items[i].indent > level) {
      // Deeper than expected without a parent at this exact level: attach to the last item, or
      // treat as this level if there's no parent yet.
      if (children.length > 0) {
        const [sub, next] = renderList(items, i, items[i].indent, `${keyPrefix}-s${li}`);
        // Graft the sublist onto the previous <li>.
        const prev = children[children.length - 1];
        children[children.length - 1] = (
          <li key={prev.key}>
            {prev.props.children}
            {sub}
          </li>
        );
        i = next;
        continue;
      }
      // No parent: normalize this deeper run to the current level.
    }
    const item = items[i];
    const key = `${keyPrefix}-i${li++}`;
    // Look ahead: does a deeper run follow this item? If so, render it as a nested <ul> child.
    let childList: ReactElement | null = null;
    let next = i + 1;
    if (next < items.length && items[next].indent > item.indent) {
      const [sub, after] = renderList(items, next, items[next].indent, `${key}-c`);
      childList = sub;
      next = after;
    }
    children.push(
      <li key={key}>
        {renderInline(item.content, key)}
        {childList}
      </li>,
    );
    i = next;
  }
  return [
    <ul key={`${keyPrefix}-ul`} className={`bm-list${level > 0 ? " bm-list--nested" : ""}`}>
      {children}
    </ul>,
    i,
  ];
}

export function SafeMarkdown({ text }: { text: string }): ReactElement {
  const normalized = text.replace(/\r\n/g, "\n").trim();
  const blocks = normalized.split(/\n{2,}/);
  const out: ReactElement[] = [];

  blocks.forEach((block, bi) => {
    const lines = block.split("\n");
    // Segment the block into consecutive runs of bullet lines vs non-bullet lines.
    type Run = { bullet: boolean; lines: string[] };
    const runs: Run[] = [];
    lines.forEach((l) => {
      const bullet = BULLET_RE.test(l);
      const tail = runs[runs.length - 1];
      if (tail && tail.bullet === bullet) tail.lines.push(l);
      else runs.push({ bullet, lines: [l] });
    });

    runs.forEach((run, ri) => {
      const kp = `b${bi}-r${ri}`;
      if (run.bullet) {
        // Parse indentation into levels. Normalize the smallest indent to 0 and step by 2-space (or
        // per-marker) units so ragged model indentation still nests sensibly.
        const raw = run.lines.map((l) => {
          const m = l.match(BULLET_RE)!;
          return { rawIndent: m[1].replace(/\t/g, "  ").length, content: m[3] };
        });
        const indents = Array.from(new Set(raw.map((r) => r.rawIndent))).sort((a, b) => a - b);
        const levelOf = new Map(indents.map((v, idx) => [v, idx]));
        const items: BulletItem[] = raw.map((r) => ({
          indent: levelOf.get(r.rawIndent) ?? 0,
          content: r.content,
        }));
        const [list] = renderList(items, 0, 0, kp);
        out.push(<div key={kp}>{list}</div>);
        return;
      }

      // Non-bullet run: each line is either a bold-only heading or part of a paragraph.
      let para: (string | ReactElement)[] = [];
      let pi = 0;
      const flush = () => {
        if (para.length) {
          out.push(
            <p key={`${kp}-p${pi++}`} className="bm-para">
              {para}
            </p>,
          );
          para = [];
        }
      };
      run.lines.forEach((l, li) => {
        if (isBoldHeading(l)) {
          flush();
          const label = l.trim().replace(/^\*\*|\*\*:?$/g, "");
          out.push(
            <p key={`${kp}-h${li}`} className="bm-heading">
              {label}
            </p>,
          );
          return;
        }
        if (l.trim() === "") return;
        if (para.length) para.push(<br key={`${kp}-br${li}`} />);
        para.push(...renderInline(l, `${kp}-l${li}`));
      });
      flush();
    });
  });

  return <>{out}</>;
}
