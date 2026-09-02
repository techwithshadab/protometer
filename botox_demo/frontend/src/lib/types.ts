// Shared types for the BOTOX® Assistant widget.

// The backend /api/chat response contract. Note: per-chunk sources are operator-facing only (they
// live in the Langfuse trace), so the response deliberately carries NO citations to the public UI.
export interface ChatResponse {
  answer: string;
  safety: boolean;
  refused: boolean;
  blocked: boolean;
  // True when the refusal is transient (protection service down) and re-trying may work.
  retryable?: boolean;
  conversation_id: string;
  // Opaque handle for this turn's trace; echoed back on /api/feedback. Null when tracing is off.
  trace_id: string | null;
}

export type Role = "user" | "assistant";

// A rendered message in the transcript. `pending` marks the assistant's typing placeholder.
export interface ChatMessage {
  id: string;
  role: Role;
  text: string;
  safety?: boolean;
  refused?: boolean;
  blocked?: boolean;
  retryable?: boolean;
  pending?: boolean;
  errored?: boolean;
  // Feedback affordance: the trace this answer belongs to, and the visitor's rating once given.
  traceId?: string | null;
  feedback?: 1 | -1;
}
