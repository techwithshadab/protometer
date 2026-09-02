import type { ReactElement } from "react";
import type { ChatMessage } from "../lib/types";
import { SafeMarkdown } from "../lib/safeMarkdown";
import { FeedbackButtons } from "./FeedbackButtons";
import { SafetyCallout } from "./SafetyCallout";
import { TypingIndicator } from "./TypingIndicator";

// One transcript row. User messages are plain text (right-aligned). Assistant messages render the
// safe-markdown answer plus an optional Safety callout. (Per-chunk sources are operator-facing only:
// they live in the Langfuse trace, never shown to the visitor, so no citations are rendered here.)
// `refused`/`blocked`/`errored` get a distinct calm style. `pending` shows the typing indicator.
export function MessageBubble({
  message,
  onFeedback,
  onRetry,
}: {
  message: ChatMessage;
  onFeedback?: (messageId: string, rating: 1 | -1) => void;
  onRetry?: (messageId: string) => void;
}): ReactElement {
  const isUser = message.role === "user";
  const special = message.refused || message.blocked || message.errored;
  // Offer feedback only on a real, delivered answer (not the typing state, not a refusal/block/error
  // fallback) that tracing recorded, a trace_id is needed to attach the score.
  const canRate = !isUser && !message.pending && !special && !!message.traceId;

  const cls = [
    "msg",
    isUser ? "msg--user" : "msg--assistant",
    special ? "msg--special" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={cls}>
      <div className="msg-bubble">
        {message.pending ? (
          <>
            <TypingIndicator />
            <span className="sr-only">Assistant is typing</span>
          </>
        ) : isUser ? (
          // User text is a plain string; React escapes it. No markdown parsing for user input.
          <span className="msg-usertext">{message.text}</span>
        ) : (
          <>
            <div className="msg-answer">
              <SafeMarkdown text={message.text} />
            </div>
            {message.safety ? <SafetyCallout /> : null}
            {(message.errored || message.retryable) && onRetry ? (
              <button
                type="button"
                className="msg-retry"
                onClick={() => onRetry(message.id)}
              >
                Try again
              </button>
            ) : null}
            {canRate ? (
              <FeedbackButtons
                traceId={message.traceId as string}
                initial={message.feedback}
                onRated={(rating) => onFeedback?.(message.id, rating)}
              />
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}
