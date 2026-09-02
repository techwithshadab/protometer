import type { ReactElement, SVGProps } from "react";
import {
  MigraineIcon,
  DystoniaIcon,
  SpasticityIcon,
  BladderIcon,
  SweatingIcon,
  EyeIcon,
} from "./landingIcons";

// The FDA-approved therapeutic areas (naming the indications is factual). Descriptions are our
// own plain-language summaries, not copied marketing copy.
type Condition = { name: string; blurb: string; Icon: (p: SVGProps<SVGSVGElement>) => ReactElement };

const CONDITIONS: Condition[] = [
  {
    name: "Chronic Migraine",
    blurb: "For adults who experience headaches on 15 or more days each month.",
    Icon: MigraineIcon,
  },
  {
    name: "Cervical Dystonia",
    blurb: "Helps reduce the abnormal head position and neck pain it can cause.",
    Icon: DystoniaIcon,
  },
  {
    name: "Spasticity",
    blurb: "Eases increased muscle stiffness in the arms and legs of eligible patients.",
    Icon: SpasticityIcon,
  },
  {
    name: "Overactive Bladder",
    blurb: "An option when other medicines haven't managed urinary symptoms well.",
    Icon: BladderIcon,
  },
  {
    name: "Severe Underarm Sweating",
    blurb: "For excessive sweating that topical treatments haven't controlled.",
    Icon: SweatingIcon,
  },
  {
    name: "Eye Muscle Conditions",
    blurb: "Used for certain eyelid spasms and eye-alignment conditions.",
    Icon: EyeIcon,
  },
];

export function ConditionsGrid(): ReactElement {
  return (
    <section className="lp-section" id="treats">
      <div className="lp-section-inner">
        <div className="lp-section-head">
          <p className="lp-eyebrow lp-eyebrow--center">Therapeutic areas</p>
          <h2 className="lp-h2">What BOTOX&reg; treats</h2>
          <p className="lp-lead">
            BOTOX&reg; is approved for several distinct medical conditions. A specialist can explain
            what each treatment involves and whether you may be a candidate.
          </p>
        </div>

        <ul className="lp-cards" role="list">
          {CONDITIONS.map(({ name, blurb, Icon }) => (
            <li key={name} className="lp-card">
              <span className="lp-card-icon" aria-hidden="true">
                <Icon />
              </span>
              <h3 className="lp-card-title">{name}</h3>
              <p className="lp-card-blurb">{blurb}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
