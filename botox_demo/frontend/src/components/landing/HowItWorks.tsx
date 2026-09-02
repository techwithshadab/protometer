import type { ReactElement, SVGProps } from "react";
import { ConsultIcon, PlanIcon, TreatIcon, FollowUpIcon } from "./landingIcons";

type Step = {
  n: number;
  title: string;
  blurb: string;
  Icon: (p: SVGProps<SVGSVGElement>) => ReactElement;
};

// General, non-clinical framing of a typical treatment journey. Original copy.
const STEPS: Step[] = [
  {
    n: 1,
    title: "Consult a specialist",
    blurb: "Talk through your symptoms and history with a licensed healthcare provider.",
    Icon: ConsultIcon,
  },
  {
    n: 2,
    title: "Review your plan",
    blurb: "Your provider explains whether treatment is appropriate and what to expect.",
    Icon: PlanIcon,
  },
  {
    n: 3,
    title: "Receive treatment",
    blurb: "Treatment is administered in-office by a trained professional.",
    Icon: TreatIcon,
  },
  {
    n: 4,
    title: "Follow up",
    blurb: "Your provider tracks your response and plans any next steps with you.",
    Icon: FollowUpIcon,
  },
];

export function HowItWorks(): ReactElement {
  return (
    <section className="lp-section lp-section--alt" id="how">
      <div className="lp-section-inner">
        <div className="lp-section-head">
          <p className="lp-eyebrow lp-eyebrow--center">What to expect</p>
          <h2 className="lp-h2">How it works</h2>
          <p className="lp-lead">
            Treatment is always guided by a licensed provider. Here's the general path most patients
            follow, your provider will tailor it to you.
          </p>
        </div>

        <ol className="lp-steps" role="list">
          {STEPS.map(({ n, title, blurb, Icon }) => (
            <li key={n} className="lp-step">
              <span className="lp-step-badge" aria-hidden="true">
                <Icon />
                <span className="lp-step-n">{n}</span>
              </span>
              <h3 className="lp-step-title">{title}</h3>
              <p className="lp-step-blurb">{blurb}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
