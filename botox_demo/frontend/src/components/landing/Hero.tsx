import type { ReactElement } from "react";
import { ChevronRight } from "./landingIcons";

// Hero: original headline + subhead, two CTAs, and an abstract branded visual (inline SVG, a
// clean geometric "cell / molecule" motif in the brand palette, not a copied image).
export function Hero(): ReactElement {
  return (
    <section className="lp-hero" id="top">
      <div className="lp-hero-inner">
        <div className="lp-hero-copy">
          <p className="lp-eyebrow">FDA-approved prescription treatment</p>
          <h1 className="lp-hero-title">
            One treatment,
            <br />
            many conditions.
          </h1>
          <p className="lp-hero-sub">
            BOTOX<sup>&reg;</sup> (onabotulinumtoxinA) is a prescription medicine used to treat a
            range of medical conditions, from chronic migraine to certain muscle and bladder
            disorders. Talk with a licensed specialist to learn whether it may be right for you.
          </p>
          <div className="lp-hero-cta">
            <a href="#find" className="lp-btn lp-btn--primary lp-btn--lg">
              Find a Specialist
              <ChevronRight />
            </a>
            <a href="#treats" className="lp-btn lp-btn--ghost lp-btn--lg">
              Learn More
            </a>
          </div>
          <p className="lp-hero-note">
            Questions? Our assistant, bottom-right, answers general questions using information from
            the official site.
          </p>
        </div>

        <div className="lp-hero-art" aria-hidden="true">
          <HeroArt />
        </div>
      </div>
    </section>
  );
}

// A calm, medical, abstract motif: concentric rings with orbiting nodes over a soft gradient , 
// evokes a molecule / targeted treatment without depicting anything clinical or copied.
function HeroArt(): ReactElement {
  return (
    <svg viewBox="0 0 440 400" width="100%" height="100%" role="img" aria-label="Abstract branded illustration">
      <defs>
        <linearGradient id="lpg1" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#23417b" />
          <stop offset="1" stopColor="#071d49" />
        </linearGradient>
        <radialGradient id="lpg2" cx="0.5" cy="0.42" r="0.6">
          <stop offset="0" stopColor="#eaf0f9" />
          <stop offset="1" stopColor="#dbe6f6" />
        </radialGradient>
      </defs>

      <rect x="20" y="20" width="400" height="360" rx="28" fill="url(#lpg2)" />

      <g fill="none" stroke="#23417b" strokeOpacity="0.25" strokeWidth="1.5">
        <circle cx="220" cy="200" r="140" />
        <circle cx="220" cy="200" r="104" />
        <circle cx="220" cy="200" r="66" />
      </g>

      <circle cx="220" cy="200" r="40" fill="url(#lpg1)" />
      <circle cx="220" cy="200" r="40" fill="none" stroke="#fff" strokeOpacity="0.5" strokeWidth="2" />

      {/* orbiting nodes */}
      <g>
        <circle cx="220" cy="60" r="12" fill="#23417b" />
        <circle cx="360" cy="200" r="9" fill="#96999e" />
        <circle cx="140" cy="304" r="14" fill="#071d49" />
        <circle cx="316" cy="296" r="8" fill="#23417b" />
        <circle cx="120" cy="128" r="10" fill="#c0bdc2" />
      </g>

      {/* connective strokes */}
      <g stroke="#23417b" strokeOpacity="0.35" strokeWidth="1.5">
        <line x1="220" y1="200" x2="220" y2="60" />
        <line x1="220" y1="200" x2="360" y2="200" />
        <line x1="220" y1="200" x2="140" y2="304" />
        <line x1="220" y1="200" x2="120" y2="128" />
      </g>
    </svg>
  );
}
