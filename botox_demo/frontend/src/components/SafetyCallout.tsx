import type { ReactElement } from "react";
import { ShieldInfoIcon } from "./icons";

// An amber "Important Safety Information" callout shown beneath safety-relevant answers. Static,
// non-clinical copy that points the user to authoritative sources, never generated advice.
export function SafetyCallout(): ReactElement {
  return (
    <div className="safety-callout" role="note" aria-label="Important Safety Information">
      <div className="safety-head">
        <ShieldInfoIcon className="safety-icon" />
        <span>Important Safety Information</span>
      </div>
      <p className="safety-body">
        BOTOX<sup>®</sup> may cause serious side effects, including the spread of toxin effects. This
        summary is informational and not a substitute for professional medical advice. Please read
        the full Important Safety Information and Prescribing Information, and talk to your doctor.
      </p>
    </div>
  );
}
