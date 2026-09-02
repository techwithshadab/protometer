// Inline SVG icons, no external icon library. All are stroke/fill via currentColor so CSS
// controls color. Decorative icons are aria-hidden; interactive labeling lives on the buttons.
import type { ReactElement, SVGProps } from "react";

export function ChatIcon(props: SVGProps<SVGSVGElement>): ReactElement {
  return (
    <svg viewBox="0 0 24 24" width="26" height="26" fill="none" aria-hidden="true" {...props}>
      <path
        d="M4 5.5A2.5 2.5 0 0 1 6.5 3h11A2.5 2.5 0 0 1 20 5.5v8A2.5 2.5 0 0 1 17.5 16H9l-4.2 3.4A0.6 0.6 0 0 1 4 18.9V5.5Z"
        fill="currentColor"
      />
      <circle cx="9" cy="9.5" r="1.15" fill="#fff" />
      <circle cx="12.5" cy="9.5" r="1.15" fill="#fff" />
      <circle cx="16" cy="9.5" r="1.15" fill="#fff" />
    </svg>
  );
}

export function CloseIcon(props: SVGProps<SVGSVGElement>): ReactElement {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" aria-hidden="true" {...props}>
      <path
        d="M6 6l12 12M18 6L6 18"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function SendIcon(props: SVGProps<SVGSVGElement>): ReactElement {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" aria-hidden="true" {...props}>
      <path
        d="M4 11.9 20 4l-4.4 16-4-6.2L4 11.9Z"
        fill="currentColor"
        stroke="currentColor"
        strokeWidth="1"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function ThumbUpIcon(props: SVGProps<SVGSVGElement>): ReactElement {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" aria-hidden="true" {...props}>
      <path
        d="M7 10.5 11 3c1 0 2 .8 2 2v3.5h4.6c1.1 0 1.9 1 1.6 2.1l-1.5 6c-.2.9-1 1.4-1.9 1.4H7m0-9.5H4.6c-.6 0-1.1.5-1.1 1.1v7.3c0 .6.5 1.1 1.1 1.1H7m0-9.5v9.5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function ThumbDownIcon(props: SVGProps<SVGSVGElement>): ReactElement {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" aria-hidden="true" {...props}>
      <path
        d="M17 13.5 13 21c-1 0-2-.8-2-2v-3.5H6.4c-1.1 0-1.9-1-1.6-2.1l1.5-6c.2-.9 1-1.4 1.9-1.4H17m0 9.5h2.4c.6 0 1.1-.5 1.1-1.1V6.1c0-.6-.5-1.1-1.1-1.1H17m0 9.5V4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function ShieldInfoIcon(props: SVGProps<SVGSVGElement>): ReactElement {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" aria-hidden="true" {...props}>
      <path
        d="M12 3l7 2.5v5c0 4.4-3 8.1-7 9.3-4-1.2-7-4.9-7-9.3v-5L12 3Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="8.6" r="0.95" fill="currentColor" />
      <path d="M12 11v5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}
