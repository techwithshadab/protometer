// Inline SVG icons for the landing page, no external icon library, no remote assets. All draw
// with currentColor so CSS controls color. Decorative, so aria-hidden.
import type { ReactElement, SVGProps } from "react";

type P = SVGProps<SVGSVGElement>;

const base = {
  viewBox: "0 0 24 24",
  width: 28,
  height: 28,
  fill: "none",
  "aria-hidden": true as const,
};

// ---- Condition icons ----

export function MigraineIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="1.6" />
      <path
        d="M12 2.5v2.4M12 19.1v2.4M2.5 12h2.4M19.1 12h2.4M5.2 5.2l1.7 1.7M17.1 17.1l1.7 1.7M18.8 5.2l-1.7 1.7M6.9 17.1l-1.7 1.7"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function DystoniaIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <path
        d="M9 3.5a3 3 0 1 1 5 2.2c-1.4 1.2-2 2-2 3.8V12"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M12 12c-3 0-5 1.8-5 4.6V21h10v-4.4C17 13.8 15 12 12 12Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function SpasticityIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <path
        d="M6 4v6.5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M6 20h12M9 16.5V20M15 16.5V20"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function BladderIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <path
        d="M12 3c-1 2.2-4.5 5.2-4.5 9.2A4.5 4.5 0 0 0 12 16.7a4.5 4.5 0 0 0 4.5-4.5C16.5 8.2 13 5.2 12 3Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path d="M12 16.7V21" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export function SweatingIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <path
        d="M12 2.5c-2.6 3.4-6 6.8-6 10.5a6 6 0 0 0 12 0c0-3.7-3.4-7.1-6-10.5Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path
        d="M9.5 13a2.5 2.5 0 0 0 2.5 2.5"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function EyeIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <path
        d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12" r="2.6" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

// ---- Step icons (How it works) ----

export function ConsultIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="8" r="3.4" stroke="currentColor" strokeWidth="1.6" />
      <path
        d="M5 20c0-3.3 3.1-5.5 7-5.5s7 2.2 7 5.5"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function PlanIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <rect x="4.5" y="4" width="15" height="16" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <path
        d="M8.5 9h7M8.5 12.5h7M8.5 16h4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function TreatIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <path
        d="M14.5 3.5 20.5 9.5M17.5 6.5 8 16c-.4.4-.6.6-1.4.8L3.5 18l.9-3.1c.2-.8.4-1 .8-1.4L15 4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function FollowUpIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <path
        d="M20 12a8 8 0 1 1-2.3-5.6"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M20 4v3.2h-3.2"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="m9 12 2.2 2.2L15 10.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ---- Misc ----

export function PinIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <path
        d="M12 21s6-5.3 6-10a6 6 0 1 0-12 0c0 4.7 6 10 6 10Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="11" r="2.2" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

export function SavingsIcon(props: P): ReactElement {
  return (
    <svg {...base} {...props}>
      <rect x="3" y="6.5" width="18" height="11" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="12" cy="12" r="2.6" stroke="currentColor" strokeWidth="1.6" />
      <path d="M6.5 9.5h.01M17.5 14.5h.01" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

export function ChevronRight(props: P): ReactElement {
  return (
    <svg viewBox="0 0 24 24" width={18} height={18} fill="none" aria-hidden {...props}>
      <path d="m9 6 6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
