import type { ReactElement } from "react";
import { SavingsIcon, PinIcon, ChevronRight } from "./landingIcons";

// Cost & coverage callout band and a "find a specialist" anchor. Original copy; no invented
// dollar figures, coverage and savings vary and are directed to the provider / program.
export function CostBand(): ReactElement {
  return (
    <section className="lp-cost" id="cost">
      <div className="lp-cost-inner">
        <div className="lp-cost-card">
          <span className="lp-cost-icon" aria-hidden="true">
            <SavingsIcon />
          </span>
          <div className="lp-cost-body">
            <h2 className="lp-h2 lp-h2--onDark">Cost &amp; coverage</h2>
            <p className="lp-cost-text">
              What you pay depends on your insurance and the condition being treated. Savings
              programs may help eligible, commercially-insured patients lower out-of-pocket costs.
              Your provider's office can walk you through coverage and any programs you may qualify
              for.
            </p>
            <a href="#find" className="lp-btn lp-btn--onDark">
              Explore coverage options
              <ChevronRight />
            </a>
          </div>
        </div>

        <div className="lp-find" id="find">
          <span className="lp-find-icon" aria-hidden="true">
            <PinIcon />
          </span>
          <div>
            <h3 className="lp-find-title">Find a specialist near you</h3>
            <p className="lp-find-text">
              Treatment begins with a licensed provider. Ask your doctor whether BOTOX&reg; may be
              appropriate for your condition.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
