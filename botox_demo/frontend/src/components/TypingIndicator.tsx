import type { ReactElement } from "react";

// Three-dot "assistant is typing" animation. Purely decorative; the surrounding bubble carries the
// accessible status text.
export function TypingIndicator(): ReactElement {
  return (
    <span className="typing" aria-hidden="true">
      <span className="typing-dot" />
      <span className="typing-dot" />
      <span className="typing-dot" />
    </span>
  );
}
