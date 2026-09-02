import { useState } from "react";
import type { ReactElement } from "react";
import { sendFeedback } from "../lib/api";
import { ThumbDownIcon, ThumbUpIcon } from "./icons";

// A small "Was this helpful?" thumbs-up/down control shown under an assistant answer. The visitor
// can change their vote (click the other thumb) or undo it (click the same thumb again); the
// current choice shows as an active button. Each change posts the rating (best-effort, a failure
// is silent). Rendered only when the turn has a trace_id, so the feedback can attach to it.
export function FeedbackButtons({
  traceId,
  initial,
  onRated,
}: {
  traceId: string;
  initial?: 1 | -1;
  onRated?: (rating: 1 | -1) => void;
}): ReactElement {
  const [rating, setRating] = useState<1 | -1 | undefined>(initial);

  function toggle(value: 1 | -1) {
    const next = rating === value ? undefined : value; // same thumb again = undo
    setRating(next);
    if (next !== undefined) {
      onRated?.(next);
      void sendFeedback(traceId, next); // fire-and-forget
    }
    // Undo is local-only: langfuse scores are immutable, so we simply stop showing a selection.
  }

  return (
    <div className="feedback">
      <span className="feedback-label">
        {rating ? "Thanks for your feedback." : "Was this helpful?"}
      </span>
      <button
        type="button"
        className={`feedback-btn${rating === 1 ? " feedback-btn--active" : ""}`}
        aria-label="Yes, this was helpful"
        aria-pressed={rating === 1}
        onClick={() => toggle(1)}
      >
        <ThumbUpIcon />
      </button>
      <button
        type="button"
        className={`feedback-btn${rating === -1 ? " feedback-btn--active" : ""}`}
        aria-label="No, this was not helpful"
        aria-pressed={rating === -1}
        onClick={() => toggle(-1)}
      >
        <ThumbDownIcon />
      </button>
    </div>
  );
}
