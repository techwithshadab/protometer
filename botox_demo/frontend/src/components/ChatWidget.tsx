import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactElement } from "react";
import type { ChatMessage } from "../lib/types";
import {
  getVisitorId,
  loadHistory,
  newConversationId,
  saveHistory,
  streamChat,
} from "../lib/api";
import { ChatIcon, CloseIcon, SendIcon } from "./icons";
import { Wordmark } from "./Wordmark";
import { MessageBubble } from "./MessageBubble";

const GREETING =
  "Hi! I can answer questions about BOTOX® treatments using information from botox.com. " +
  "What would you like to know?";

const SUGGESTIONS = [
  "What is BOTOX® used to treat?",
  "What are the most common side effects?",
  "How much does BOTOX® cost?",
];

const REFUSAL_FALLBACK =
  "I can only help with general BOTOX® information from the official site. For anything about " +
  "your specific situation, please consult your healthcare provider.";

let msgSeq = 0;
function nextId(): string {
  msgSeq += 1;
  return `m${msgSeq}-${Date.now()}`;
}

interface ChatWidgetProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ChatWidget({ open, onOpenChange }: ChatWidgetProps): ReactElement {
  // Anonymous, persistent visitor id (localStorage). No PII, an opaque token so returning visitors
  // can be recognised for analytics without any device/IP/personal data.
  const visitor = useMemo(() => getVisitorId(), []);
  // Restore a prior conversation from this tab session (survives a page refresh); otherwise start
  // fresh with a new id and the greeting. Computed once on mount.
  const restored = useMemo(() => loadHistory(), []);
  const conversationId = useMemo(
    () => restored?.conversationId ?? newConversationId(),
    [restored],
  );
  const [messages, setMessages] = useState<ChatMessage[]>(
    () => restored?.messages ?? [{ id: nextId(), role: "assistant", text: GREETING }],
  );
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  // A short cue announced to screen readers when a turn settles (the pending→answer swap reuses one
  // node id, which some SRs don't re-announce).
  const [announce, setAnnounce] = useState("");

  const listRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const launcherRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLElement | null>(null);

  // Auto-scroll to the newest message whenever the transcript or busy state changes.
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  // Focus the input when the panel opens.
  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Persist the transcript to sessionStorage whenever it settles, so a refresh keeps the
  // conversation. Transient (pending/errored) bubbles are dropped by saveHistory itself.
  useEffect(() => {
    saveHistory(conversationId, messages);
  }, [conversationId, messages]);

  // Abort any in-flight request if the component unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  // Close on Escape (returning focus to the launcher), and trap Tab focus within the panel while
  // it's open so keyboard/SR users can't wander into the landing page behind it.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onOpenChange(false);
        return;
      }
      if (e.key !== "Tab" || !panelRef.current) return;
      const focusables = panelRef.current.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), textarea:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      const active = document.activeElement as HTMLElement | null;
      if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onOpenChange]);

  // When the panel closes, return focus to the launcher so keyboard users keep their place.
  const wasOpen = useRef(open);
  useEffect(() => {
    if (wasOpen.current && !open) launcherRef.current?.focus();
    wasOpen.current = open;
  }, [open]);

  const onlyGreeting = messages.length === 1;

  // Persist a visitor's thumbs rating on the message so the control stays "voted" after re-render.
  function handleFeedback(messageId: string, rating: 1 | -1) {
    setMessages((prev) => prev.map((m) => (m.id === messageId ? { ...m, feedback: rating } : m)));
  }

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;

    const userMsg: ChatMessage = { id: nextId(), role: "user", text: trimmed };
    const pendingId = nextId();
    const pendingMsg: ChatMessage = { id: pendingId, role: "assistant", text: "", pending: true };

    setMessages((prev) => [...prev, userMsg, pendingMsg]);
    setInput("");
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;

    // Accumulate streamed draft text locally so each token appends without a stale-closure race.
    let streamed = "";
    try {
      await streamChat(
        { message: trimmed, conversation_id: conversationId, visitor_id: visitor.id },
        (token) => {
          // First token flips the bubble out of the "typing" state; subsequent tokens append.
          streamed += token;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pendingId ? { ...m, pending: false, text: streamed } : m,
            ),
          );
        },
        (final) => {
          // The guard-approved result. If `replaced`, the streamed draft was retracted (blocked/
          // refused/scrubbed), swap in the safe text. Otherwise keep the streamed text but resync
          // to the canonical answer (handles minor scrubbing differences) and attach flags.
          const answerText =
            (final.refused || final.blocked) && (!final.answer || !final.answer.trim())
              ? REFUSAL_FALLBACK
              : final.answer || streamed;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === pendingId
                ? {
                    ...m,
                    pending: false,
                    text: answerText,
                    safety: final.safety,
                    refused: final.refused,
                    blocked: final.blocked,
                    retryable: final.retryable,
                    traceId: final.trace_id,
                  }
                : m,
            ),
          );
          setAnnounce(`Assistant replied. ${answerText}`);
        },
        controller.signal,
      );
    } catch (err) {
      if (controller.signal.aborted) return;
      const message =
        err instanceof Error ? err.message : "The assistant is temporarily unavailable.";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === pendingId ? { ...m, pending: false, errored: true, text: message } : m,
        ),
      );
      setAnnounce("The assistant could not respond. You can try again.");
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  // Retry a failed turn: drop the errored bubble and its preceding user message, then re-submit the
  // original text (same conversation + visitor id).
  function handleRetry(erroredId: string) {
    if (busy) return;
    let retryText = "";
    setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === erroredId);
      if (idx <= 0) return prev;
      const userMsg = prev[idx - 1];
      if (userMsg?.role === "user") retryText = userMsg.text;
      // Remove the user message + the errored assistant bubble; submit() re-adds them.
      return prev.filter((_, i) => i !== idx && i !== idx - 1);
    });
    if (retryText) void submit(retryText);
  }

  return (
    <div className="botox-widget">
      {/* Launcher */}
      <button
        ref={launcherRef}
        type="button"
        className={`launcher ${open ? "launcher--hidden" : ""}`}
        aria-label="Open the BOTOX Assistant chat"
        aria-expanded={open}
        onClick={() => onOpenChange(true)}
      >
        <ChatIcon />
      </button>

      {/* Screen-reader announcement of settled turns. The transcript reuses one node id for the
          pending→answer swap, which some SRs don't re-announce, so we mirror the completed text to
          a dedicated polite live region. */}
      <div className="sr-only" role="status" aria-live="polite">
        {announce}
      </div>

      {/* Panel */}
      {open ? (
        <section
          ref={panelRef}
          className="panel"
          role="dialog"
          aria-modal="false"
          aria-label="BOTOX Assistant chat"
        >
          <header className="panel-header">
            <div className="panel-title">
              <Wordmark suffix="Assistant" />
              <p className="panel-subtitle">Ask about BOTOX® treatments</p>
            </div>
            <button
              type="button"
              className="icon-btn panel-close"
              aria-label="Close the chat"
              onClick={() => onOpenChange(false)}
            >
              <CloseIcon />
            </button>
          </header>

          <div className="disclaimer-strip">
            Informational only, not medical advice. Talk to your doctor.
          </div>

          <div className="messages" role="log" ref={listRef}>
            {messages.map((m) => (
              <MessageBubble
                key={m.id}
                message={m}
                onFeedback={handleFeedback}
                onRetry={handleRetry}
              />
            ))}

            {onlyGreeting ? (
              <div className="suggestions" aria-label="Suggested questions">
                {SUGGESTIONS.map((q) => (
                  <button
                    key={q}
                    type="button"
                    className="suggestion-chip"
                    disabled={busy}
                    onClick={() => submit(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            ) : null}
          </div>

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              submit(input);
            }}
          >
            <input
              ref={inputRef}
              className="composer-input"
              type="text"
              value={input}
              placeholder="Type your question…"
              aria-label="Type your question"
              autoComplete="off"
              disabled={busy}
              onChange={(e) => setInput(e.target.value)}
            />
            <button
              type="submit"
              className="icon-btn send-btn"
              aria-label="Send message"
              disabled={busy || input.trim().length === 0}
            >
              <SendIcon />
            </button>
          </form>
        </section>
      ) : null}
    </div>
  );
}
