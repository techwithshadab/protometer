import type { ReactElement } from "react";
import { ShieldInfoIcon } from "../icons";

// Prominent Important Safety Information block + a plain site footer. The ISI language mirrors the
// summary used in the widget's SafetyCallout. The footer makes the DEMO nature explicit.
export function ISIFooter(): ReactElement {
  return (
    <footer className="lp-footer">
      <section className="lp-isi" aria-labelledby="isi-heading">
        <div className="lp-isi-inner">
          <div className="lp-isi-head">
            <ShieldInfoIcon className="lp-isi-icon" />
            <h2 id="isi-heading" className="lp-isi-title">
              Important Safety Information
            </h2>
          </div>

          <div className="lp-isi-body">
            <p>
              <strong>
                BOTOX<sup>&reg;</sup> may cause serious side effects that can be life threatening.
              </strong>{" "}
              These include problems with swallowing, speaking, or breathing due to the spread of
              toxin effects, which can happen hours to weeks after an injection.
            </p>
            <p>
              BOTOX<sup>&reg;</sup> should not be used if you have an infection at the planned
              injection site or are allergic to any of its ingredients. Tell your doctor about all
              your medical conditions and medications before treatment.
            </p>
            <p>
              This page is an informational demonstration and is <strong>not medical advice</strong>,
              not a substitute for a consultation with a licensed healthcare provider, and not a
              complete list of risks. Please read the full Important Safety Information and
              Prescribing Information, and talk to your doctor about whether treatment is right for
              you.
            </p>
          </div>
        </div>
      </section>

      <div className="lp-footer-bottom">
        <div className="lp-footer-inner">
          <span className="lp-wordmark lp-wordmark--sm">
            BOTOX<sup>&reg;</sup>
          </span>
          <p className="lp-footer-disclaimer">
            Demo experience for illustration only, not affiliated with, endorsed by, or the
            official site of the BOTOX<sup>&reg;</sup> brand owner. Informational only; not medical
            advice. BOTOX<sup>&reg;</sup> is a registered trademark of its respective owner.
          </p>
        </div>
      </div>
    </footer>
  );
}
