// API client for the BOTOX® Assistant backend.
//
// One endpoint: POST {API_BASE}/chat with { message, conversation_id }.
// In production nginx proxies /api to the backend; in dev Vite proxies it. So the default base is
// the relative "/api". An explicit VITE_API_BASE overrides it (e.g. for a separately hosted API).

import type { ChatMessage, ChatResponse } from "./types";

const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") || "/api";

export interface ChatRequest {
  message: string;
  conversation_id: string;
  visitor_id?: string;
}

// A fresh random id. crypto.randomUUID is available in all modern browsers over https/localhost;
// fall back to a timestamp+random token if it is somehow absent.
function randomId(prefix: string): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return `${prefix}${crypto.randomUUID()}`;
    }
  } catch {
    /* fall through */
  }
  return `${prefix}${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

// A stable per-session (per page load) conversation id.
export function newConversationId(): string {
  return randomId("conv-");
}

const VISITOR_KEY = "botox_visitor_id";

// A stable, ANONYMOUS visitor id that persists across sessions and return visits, stored in
// localStorage. It contains NO personal data, it's an opaque random token, the privacy-safe way to
// recognise a returning visitor without any device/IP/PII. localStorage can throw or be unavailable
// (private windows, blocked site data), so every access is guarded and we fall back to a
// session-only id rather than break. `isReturning` reports whether we found an existing id.
export function getVisitorId(): { id: string; isReturning: boolean } {
  try {
    const existing = localStorage.getItem(VISITOR_KEY);
    if (existing) return { id: existing, isReturning: true };
    const fresh = randomId("v-");
    localStorage.setItem(VISITOR_KEY, fresh);
    return { id: fresh, isReturning: false };
  } catch {
    // Storage unavailable, use a volatile per-load id so a turn still carries a visitor tag.
    return { id: randomId("v-ephemeral-"), isReturning: false };
  }
}

// ── Chat-history persistence (per-tab session) ─────────────────────────────────────────────────
// Persist the transcript to sessionStorage so a page refresh keeps the conversation. sessionStorage
// (not localStorage) scopes it to the tab session, matching the per-page-load conversation id. Every
// access is guarded exactly like getVisitorId, it must never throw (private windows, blocked site
// data) and must degrade gracefully.
const HISTORY_KEY = "botox_chat_history";

interface PersistedHistory {
  conversationId: string;
  messages: ChatMessage[];
}

export function saveHistory(conversationId: string, messages: ChatMessage[]): void {
  try {
    // Never persist transient state: drop the typing placeholder and any errored bubble so a
    // restored transcript only contains settled turns.
    const settled = messages.filter((m) => !m.pending && !m.errored);
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify({ conversationId, messages: settled }));
  } catch {
    /* storage unavailable, history just won't survive a refresh */
  }
}

export function loadHistory(): PersistedHistory | null {
  try {
    const raw = sessionStorage.getItem(HISTORY_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PersistedHistory>;
    if (
      !parsed ||
      typeof parsed.conversationId !== "string" ||
      !Array.isArray(parsed.messages) ||
      parsed.messages.length === 0
    ) {
      return null;
    }
    // Defensive: keep only well-formed, settled messages.
    const messages = parsed.messages.filter(
      (m): m is ChatMessage =>
        !!m && typeof m.id === "string" && typeof m.text === "string" &&
        (m.role === "user" || m.role === "assistant") && !m.pending && !m.errored,
    );
    if (messages.length === 0) return null;
    return { conversationId: parsed.conversationId, messages };
  } catch {
    return null;
  }
}

export async function sendChat(req: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });

  if (!res.ok) {
    // Surface a clean, user-safe error; the widget renders this as an assistant error bubble.
    throw new Error(`The assistant is temporarily unavailable (status ${res.status}).`);
  }

  const data = (await res.json()) as Partial<ChatResponse>;

  // Defensive normalization: never trust the shape blindly, so the UI can't crash on a bad field.
  return {
    answer: typeof data.answer === "string" ? data.answer : "",
    safety: Boolean(data.safety),
    refused: Boolean(data.refused),
    blocked: Boolean(data.blocked),
    retryable: Boolean(data.retryable),
    conversation_id: typeof data.conversation_id === "string" ? data.conversation_id : req.conversation_id,
    trace_id: typeof data.trace_id === "string" ? data.trace_id : null,
  };
}

// The `final` SSE event: the guard-approved result after the full reply was validated. `replaced`
// means the streamed draft was retracted (blocked/refused/scrubbed) and `answer` is the safe text.
export interface StreamFinal {
  answer: string;
  safety: boolean;
  refused: boolean;
  blocked: boolean;
  retryable: boolean;
  replaced: boolean;
  conversation_id: string;
  trace_id: string | null;
}

// How long to wait for ANY further bytes from the stream before giving up. A backend that accepts
// the connection then stalls (no tokens, no `final`) must not leave the UI stuck in "typing"
// forever, so we abort on this idle gap and surface an error the caller renders as a retry bubble.
const STREAM_IDLE_TIMEOUT_MS = 45_000;

// Stream an answer via Server-Sent Events. Calls onToken(text) for each draft chunk as it arrives,
// then onFinal(result) once with the authoritative guard-approved result. Reads the response body
// as a stream and parses "data: {json}\n\n" frames. Throws on a non-OK response, a network error,
// or an idle-timeout (the caller renders that as an error bubble). Honors the AbortSignal.
export async function streamChat(
  req: ChatRequest,
  onToken: (text: string) => void,
  onFinal: (result: StreamFinal) => void,
  signal?: AbortSignal,
): Promise<void> {
  // Own controller so we can abort on our idle timeout; also forward the caller's external signal.
  const controller = new AbortController();
  const onExternalAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", onExternalAbort, { once: true });
  }

  let timedOut = false;
  let idleTimer: ReturnType<typeof setTimeout> | undefined;
  const armIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, STREAM_IDLE_TIMEOUT_MS);
  };

  try {
    armIdleTimer();
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) {
      throw new Error(`The assistant is temporarily unavailable (status ${res.status}).`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    // Once the authoritative `final` event arrives, the turn is settled. Ignore any stray `token`
    // that follows it (a reordering proxy, a buggy server): a late token must never clobber the
    // guard-approved answer and resurrect retracted/unsafe draft text.
    let finalReceived = false;

    const handle = (jsonText: string) => {
      let evt: Record<string, unknown>;
      try {
        evt = JSON.parse(jsonText);
      } catch {
        return; // ignore a malformed frame rather than break the stream
      }
      if (evt.type === "token" && typeof evt.text === "string") {
        if (finalReceived) return; // no tokens after final
        onToken(evt.text);
      } else if (evt.type === "final") {
        if (finalReceived) return; // only the first final counts
        finalReceived = true;
        onFinal({
          answer: typeof evt.answer === "string" ? evt.answer : "",
          safety: Boolean(evt.safety),
          refused: Boolean(evt.refused),
          blocked: Boolean(evt.blocked),
          retryable: Boolean(evt.retryable),
          replaced: Boolean(evt.replaced),
          conversation_id:
            typeof evt.conversation_id === "string" ? evt.conversation_id : req.conversation_id,
          trace_id: typeof evt.trace_id === "string" ? evt.trace_id : null,
        });
      }
    };

    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      armIdleTimer(); // fresh bytes arrived, reset the idle clock
      buffer += decoder.decode(value, { stream: true });
      // SSE frames are separated by a blank line; each frame's payload lines start with "data: ".
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        const dataLine = frame
          .split("\n")
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.slice(5).trim())
          .join("");
        if (dataLine) handle(dataLine);
      }
      if (finalReceived) break; // settled; stop reading even if the server holds the connection
    }
  } catch (err) {
    // An idle-timeout aborts the controller; report it as a clear timeout rather than a bare abort,
    // so the caller shows a retry bubble instead of silently swallowing it (as it does for a
    // user-initiated abort, which keeps the original external signal's `aborted` flag).
    if (timedOut) {
      throw new Error("The assistant took too long to respond. Please try again.");
    }
    throw err;
  } finally {
    if (idleTimer) clearTimeout(idleTimer);
    if (signal) signal.removeEventListener("abort", onExternalAbort);
  }
}

// Send a thumbs-up (+1) / thumbs-down (-1) on an answer. Fire-and-forget from the UI's point of
// view: feedback is best-effort telemetry, so a failure is swallowed and never blocks the user.
export async function sendFeedback(traceId: string, rating: 1 | -1): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trace_id: traceId, rating }),
    });
    return res.ok;
  } catch {
    return false;
  }
}
