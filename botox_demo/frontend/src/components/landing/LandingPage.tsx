import type { ReactElement } from "react";
import { SiteNav } from "./SiteNav";
import { Hero } from "./Hero";
import { ConditionsGrid } from "./ConditionsGrid";
import { HowItWorks } from "./HowItWorks";
import { CostBand } from "./CostBand";
import { ISIFooter } from "./ISIFooter";

// The demo host page: an original, brand-aligned BOTOX(R)-style marketing landing page that sits
// BEHIND the floating chat widget. It is intentionally NOT a copy of the real botox.com, the copy
// is original and a demo disclaimer is shown in the footer. The chat widget mounts separately at a
// very high z-index, so this page is purely the backdrop.
export function LandingPage(): ReactElement {
  return (
    <div className="lp">
      <SiteNav />
      <main id="lp-main">
        <Hero />
        <ConditionsGrid />
        <HowItWorks />
        <CostBand />
      </main>
      <ISIFooter />
    </div>
  );
}
