import type { ReactElement } from "react";

// Sticky top navigation. Wordmark left, anchor links to the page sections, and a primary CTA.
// Links are plain in-page anchors; on narrow screens the link row hides and only the wordmark +
// CTA remain (handled in CSS).
const LINKS = [
  { href: "#treats", label: "What BOTOX® Treats" },
  { href: "#how", label: "How It Works" },
  { href: "#cost", label: "Cost & Coverage" },
];

export function SiteNav(): ReactElement {
  return (
    <header className="lp-nav">
      <div className="lp-nav-inner">
        <a className="lp-nav-brand" href="#top" aria-label="BOTOX home">
          <span className="lp-wordmark">
            BOTOX<sup>&reg;</sup>
          </span>
          <span className="lp-wordmark-sub">onabotulinumtoxinA</span>
        </a>

        <nav className="lp-nav-links" aria-label="Primary">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} className="lp-nav-link">
              {l.label}
            </a>
          ))}
        </nav>

        <a href="#find" className="lp-btn lp-btn--primary lp-nav-cta">
          Find a Specialist
        </a>
      </div>
    </header>
  );
}
