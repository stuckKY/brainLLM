"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const INFERENCE_LABELS: Record<number, string> = {
  1: "Strict",
  2: "Conservative",
  3: "Balanced",
  4: "Inferential",
  5: "Speculative",
};

const EVIDENCE_LABELS: Record<number, string> = {
  1: "Narrow (3)",
  2: "Focused (6)",
  3: "Moderate (10)",
  4: "Broad (15)",
  5: "Exhaustive (25)",
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  chunks_used: number;
  sources: string[];
  inference_level: number;
  evidence_depth: number;
  created_at: string;
}

interface ConversationSummary {
  id: string;
  title: string;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

function useTheme() {
  const [dark, setDark] = useState(false);

  useEffect(() => {
    setDark(document.documentElement.classList.contains("dark"));
  }, []);

  function toggle() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    localStorage.setItem("theme", next ? "dark" : "light");
  }

  return { dark, toggle };
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex justify-end mb-6">
      <div className="max-w-[80%] bg-neutral-100 dark:bg-neutral-800 px-5 py-3 text-sm leading-relaxed">
        {content}
      </div>
    </div>
  );
}

function AssistantMessage({
  content,
  chunksUsed,
  sources,
}: {
  content: string;
  chunksUsed: number;
  sources: string[];
}) {
  return (
    <div className="mb-8">
      <div className="border border-neutral-200 dark:border-neutral-800 p-6">
        <div className="flex items-center gap-3 mb-4 pb-4 border-b border-neutral-100 dark:border-neutral-800">
          <span className="text-xs tracking-widest uppercase text-neutral-400 dark:text-neutral-500">
            Response
          </span>
          {chunksUsed > 0 && (
            <span className="text-xs text-neutral-400 dark:text-neutral-500">
              &middot; {chunksUsed} chunks
            </span>
          )}
        </div>
        <div className="prose prose-neutral dark:prose-invert max-w-none prose-headings:font-black prose-headings:uppercase prose-headings:tracking-tight prose-h2:text-lg prose-h3:text-base prose-p:leading-relaxed">
          <ReactMarkdown>{content}</ReactMarkdown>
        </div>
        {sources.length > 0 && (
          <div className="mt-6 pt-4 border-t border-neutral-100 dark:border-neutral-800">
            <span className="text-xs tracking-widest uppercase text-neutral-400 dark:text-neutral-500">
              Sources
            </span>
            <div className="mt-2 flex flex-wrap gap-2">
              {sources.map((source) => (
                <span
                  key={source}
                  className="text-xs bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400 px-2 py-1"
                >
                  {source}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  open,
  onToggle,
}: {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  open: boolean;
  onToggle: () => void;
}) {
  function formatTime(iso: string) {
    const d = new Date(iso);
    const now = new Date();
    const diff = now.getTime() - d.getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    if (days < 7) return `${days}d ago`;
    return d.toLocaleDateString();
  }

  return (
    <>
      {/* Mobile overlay */}
      {open && (
        <div
          className="fixed inset-0 bg-black/20 z-20 lg:hidden"
          onClick={onToggle}
        />
      )}

      <aside
        className={`
          fixed top-0 left-0 z-30 h-full w-64 border-r border-neutral-200 dark:border-neutral-800
          bg-white dark:bg-neutral-950 flex flex-col transition-transform duration-200
          lg:relative lg:translate-x-0
          ${open ? "translate-x-0" : "-translate-x-full"}
        `}
      >
        {/* New conversation button */}
        <div className="p-4 border-b border-neutral-200 dark:border-neutral-800">
          <button
            onClick={onNew}
            className="w-full py-2 border border-neutral-200 dark:border-neutral-700 text-xs tracking-wide uppercase font-semibold text-neutral-600 dark:text-neutral-400 hover:bg-neutral-50 dark:hover:bg-neutral-900 transition-colors"
          >
            + New conversation
          </button>
        </div>

        {/* Conversation list */}
        <div className="flex-1 overflow-y-auto">
          {conversations.length === 0 && (
            <p className="text-xs text-neutral-400 dark:text-neutral-600 text-center mt-8">
              No conversations yet
            </p>
          )}
          {conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => onSelect(conv.id)}
              className={`
                group flex items-start gap-2 px-4 py-3 cursor-pointer border-b border-neutral-100 dark:border-neutral-900
                hover:bg-neutral-50 dark:hover:bg-neutral-900 transition-colors
                ${conv.id === activeId ? "bg-neutral-100 dark:bg-neutral-800" : ""}
              `}
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm truncate text-neutral-800 dark:text-neutral-200">
                  {conv.title}
                </p>
                <p className="text-xs text-neutral-400 dark:text-neutral-600 mt-0.5">
                  {formatTime(conv.updated_at)}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(conv.id);
                }}
                className="opacity-0 group-hover:opacity-100 text-neutral-400 hover:text-red-500 dark:text-neutral-600 dark:hover:text-red-400 text-xs mt-0.5 transition-opacity"
                aria-label="Delete conversation"
              >
                &times;
              </button>
            </div>
          ))}
        </div>
      </aside>
    </>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function Home() {
  const { dark, toggle } = useTheme();

  // Sidebar
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Current conversation
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);

  // Input
  const [question, setQuestion] = useState("");
  const [inferenceLevel, setInferenceLevel] = useState(3);
  const [evidenceDepth, setEvidenceDepth] = useState(3);
  const [showSliders, setShowSliders] = useState(true);

  // Status
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");

  // Scroll ref
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load conversations on mount
  const loadConversations = useCallback(async () => {
    try {
      const res = await fetch("/api/conversations");
      if (res.ok) {
        setConversations(await res.json());
      }
    } catch {
      // Silently fail — sidebar just stays empty
    }
  }, []);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  // Select a conversation
  async function selectConversation(id: string) {
    setActiveConversationId(id);
    setMessages([]);
    setError("");
    setSidebarOpen(false);

    try {
      const res = await fetch(`/api/conversations/${id}`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages);
      }
    } catch {
      setError("Failed to load conversation");
    }
  }

  // Start new conversation
  function startNewConversation() {
    setActiveConversationId(null);
    setMessages([]);
    setQuestion("");
    setError("");
    setSidebarOpen(false);
  }

  // Delete a conversation
  async function deleteConversation(id: string) {
    try {
      await fetch(`/api/conversations/${id}`, { method: "DELETE" });
      if (id === activeConversationId) {
        startNewConversation();
      }
      loadConversations();
    } catch {
      setError("Failed to delete conversation");
    }
  }

  // Ask a question
  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setError("");

    const userContent = question;
    setQuestion("");

    // Optimistic: show user message immediately
    const tempUserMsg: Message = {
      id: Date.now(),
      role: "user",
      content: userContent,
      chunks_used: 0,
      sources: [],
      inference_level: inferenceLevel,
      evidence_depth: evidenceDepth,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, tempUserMsg]);

    // Collapse sliders after first message
    if (messages.length === 0) {
      setShowSliders(false);
    }

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userContent,
          inference_level: inferenceLevel,
          evidence_depth: evidenceDepth,
          conversation_id: activeConversationId,
        }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Error ${res.status}`);
      }

      const data = await res.json();

      // Update conversation ID if this was a new conversation
      if (!activeConversationId) {
        setActiveConversationId(data.conversation_id);
      }

      // Add assistant message
      const assistantMsg: Message = {
        id: Date.now() + 1,
        role: "assistant",
        content: data.answer,
        chunks_used: data.chunks_used,
        sources: data.sources,
        inference_level: inferenceLevel,
        evidence_depth: evidenceDepth,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Refresh sidebar (title may have been generated)
      loadConversations();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      // Remove optimistic user message on error
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  }

  // Ingest documents
  async function handleIngest() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/ingest", { method: "POST" });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Error ${res.status}`);
      }
      const data = await res.json();

      const parts: string[] = [];
      if (data.files_new > 0) parts.push(`${data.files_new} new`);
      if (data.files_modified > 0) parts.push(`${data.files_modified} modified`);
      if (data.files_skipped > 0) parts.push(`${data.files_skipped} unchanged`);
      if (data.files_deleted > 0) parts.push(`${data.files_deleted} deleted`);

      const summary =
        parts.length > 0
          ? `Ingestion complete — ${parts.join(", ")}. ${data.chunks_stored} chunks stored.`
          : "Ingestion complete — no documents found.";

      setToast(summary);
      setTimeout(() => setToast(""), 6000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ingestion failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-neutral-200 dark:border-neutral-800 z-10">
        <div className="px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Sidebar toggle */}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-neutral-500 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100 transition-colors lg:hidden"
              aria-label="Toggle sidebar"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="3" y1="6" x2="21" y2="6" />
                <line x1="3" y1="12" x2="21" y2="12" />
                <line x1="3" y1="18" x2="21" y2="18" />
              </svg>
            </button>
            <span className="text-xs tracking-widest uppercase text-neutral-400 dark:text-neutral-500">
              Personal Knowledge Base
            </span>
          </div>
          <div className="flex items-center gap-5">
            <button
              onClick={handleIngest}
              disabled={loading}
              className="text-xs tracking-wide uppercase text-neutral-500 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100 disabled:text-neutral-300 dark:disabled:text-neutral-600 transition-colors"
            >
              Ingest documents
            </button>
            <button
              onClick={toggle}
              className="text-xs tracking-wide uppercase text-neutral-500 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100 transition-colors"
              aria-label="Toggle dark mode"
            >
              {dark ? "Light" : "Dark"}
            </button>
          </div>
        </div>
      </header>

      {/* Body: sidebar + chat */}
      <div className="flex flex-1 min-h-0">
        {/* Sidebar */}
        <Sidebar
          conversations={conversations}
          activeId={activeConversationId}
          onSelect={selectConversation}
          onNew={startNewConversation}
          onDelete={deleteConversation}
          open={sidebarOpen}
          onToggle={() => setSidebarOpen(false)}
        />

        {/* Chat area */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Scrollable messages */}
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto px-6">
              {/* Empty state */}
              {messages.length === 0 && (
                <div className="pt-16 pb-8">
                  <h1 className="text-5xl sm:text-6xl md:text-7xl font-black uppercase leading-[0.95] tracking-tight mb-6">
                    Ask Your
                    <br />
                    Brain.
                  </h1>
                  <p className="text-neutral-500 dark:text-neutral-400 text-base max-w-md">
                    Query your personal document corpus using natural language.
                    Adjust inference and evidence depth to control how the AI
                    reasons.
                  </p>
                </div>
              )}

              {/* Messages */}
              {messages.length > 0 && (
                <div className="pt-8 pb-4">
                  {messages.map((msg) =>
                    msg.role === "user" ? (
                      <UserMessage key={msg.id} content={msg.content} />
                    ) : (
                      <AssistantMessage
                        key={msg.id}
                        content={msg.content}
                        chunksUsed={msg.chunks_used}
                        sources={msg.sources}
                      />
                    )
                  )}

                  {/* Thinking indicator */}
                  {loading && (
                    <div className="mb-8">
                      <div className="border border-neutral-200 dark:border-neutral-800 p-6">
                        <span className="text-xs tracking-widest uppercase text-neutral-400 dark:text-neutral-500 animate-pulse">
                          Thinking&hellip;
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div ref={bottomRef} />
            </div>
          </div>

          {/* Toast */}
          {toast && (
            <div className="mx-auto max-w-3xl px-6 pb-2">
              <div className="bg-neutral-100 dark:bg-neutral-800 text-neutral-700 dark:text-neutral-300 text-sm px-4 py-3 border border-neutral-200 dark:border-neutral-700">
                {toast}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mx-auto max-w-3xl px-6 pb-2">
              <div className="border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-400 p-4 text-sm">
                {error}
              </div>
            </div>
          )}

          {/* Input area */}
          <div className="flex-shrink-0 border-t border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-950">
            <div className="max-w-3xl mx-auto px-6 py-4">
              <div className="border border-neutral-200 dark:border-neutral-800 p-4">
                <textarea
                  className="w-full bg-transparent text-neutral-900 dark:text-neutral-100 placeholder-neutral-400 dark:placeholder-neutral-600 resize-none focus:outline-none text-sm leading-relaxed"
                  rows={2}
                  placeholder="What do your notes say about..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.metaKey || e.ctrlKey))
                      handleAsk();
                  }}
                />

                {/* Sliders (collapsible) */}
                <div className="mt-2 pt-2 border-t border-neutral-100 dark:border-neutral-800">
                  <button
                    onClick={() => setShowSliders(!showSliders)}
                    className="text-xs tracking-wide uppercase text-neutral-400 dark:text-neutral-500 hover:text-neutral-600 dark:hover:text-neutral-300 transition-colors mb-2"
                  >
                    {showSliders ? "Hide settings" : "Show settings"} &mdash;
                    Inference: {INFERENCE_LABELS[inferenceLevel]}, Evidence:{" "}
                    {EVIDENCE_LABELS[evidenceDepth]}
                  </button>

                  {showSliders && (
                    <div className="grid grid-cols-2 gap-8 mt-2">
                      <div>
                        <div className="flex items-baseline justify-between mb-2">
                          <span className="text-xs tracking-wide uppercase text-neutral-400 dark:text-neutral-500">
                            Inference
                          </span>
                          <span className="text-xs font-semibold text-neutral-700 dark:text-neutral-300">
                            {inferenceLevel} &mdash;{" "}
                            {INFERENCE_LABELS[inferenceLevel]}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={1}
                          max={5}
                          step={1}
                          value={inferenceLevel}
                          onChange={(e) =>
                            setInferenceLevel(Number(e.target.value))
                          }
                          className="w-full"
                        />
                      </div>
                      <div>
                        <div className="flex items-baseline justify-between mb-2">
                          <span className="text-xs tracking-wide uppercase text-neutral-400 dark:text-neutral-500">
                            Evidence
                          </span>
                          <span className="text-xs font-semibold text-neutral-700 dark:text-neutral-300">
                            {evidenceDepth} &mdash;{" "}
                            {EVIDENCE_LABELS[evidenceDepth]}
                          </span>
                        </div>
                        <input
                          type="range"
                          min={1}
                          max={5}
                          step={1}
                          value={evidenceDepth}
                          onChange={(e) =>
                            setEvidenceDepth(Number(e.target.value))
                          }
                          className="w-full"
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Submit row */}
                <div className="mt-3 pt-3 border-t border-neutral-100 dark:border-neutral-800 flex items-center justify-between">
                  <span className="text-xs text-neutral-400 dark:text-neutral-500">
                    {loading ? "" : "\u2318 + Enter to submit"}
                  </span>
                  <button
                    onClick={handleAsk}
                    disabled={loading || !question.trim()}
                    className="px-5 py-2 bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 text-xs tracking-wide uppercase font-semibold hover:bg-neutral-700 dark:hover:bg-neutral-300 disabled:bg-neutral-200 disabled:text-neutral-400 dark:disabled:bg-neutral-800 dark:disabled:text-neutral-600 transition-colors"
                  >
                    {loading ? "Thinking\u2026" : "Ask"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
