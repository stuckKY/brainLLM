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

const SUPPORTED_EXTENSIONS = [
  ".md", ".pdf", ".pptx", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp",
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ChunkReference {
  index: number;
  id: number;
  source: string;
  content: string;
  similarity: number;
}

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  chunks_used: number;
  sources: string[];
  chunks_data: ChunkReference[];
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

function CitationButton({
  num,
  source,
  onClick,
}: {
  num: number;
  source: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className="inline-flex items-center justify-center min-w-[1.25rem] h-5 text-[10px] font-bold bg-neutral-200 dark:bg-neutral-700 text-neutral-600 dark:text-neutral-300 rounded-full hover:bg-neutral-300 dark:hover:bg-neutral-600 cursor-pointer transition-colors mx-0.5 px-1 align-super"
      title={source}
    >
      {num}
    </button>
  );
}

// Regex to match citation patterns like [1], [2] etc.
// Negative lookahead avoids matching markdown link refs like [1]: or [1](
const CITATION_REGEX = /\[(\d{1,2})\](?![:(])/g;

function processChildren(
  children: React.ReactNode,
  chunks: ChunkReference[],
  toggle: (n: number) => void,
): React.ReactNode {
  if (typeof children === "string") {
    const parts = children.split(CITATION_REGEX);
    if (parts.length === 1) return children;

    // split produces: [text, match1, text, match2, text, ...]
    const result: React.ReactNode[] = [];
    for (let i = 0; i < parts.length; i++) {
      if (i % 2 === 0) {
        // Plain text segment
        if (parts[i]) result.push(parts[i]);
      } else {
        // Captured group — citation number
        const refNum = parseInt(parts[i]);
        const chunk = chunks.find((c) => c.index === refNum);
        if (chunk) {
          result.push(
            <CitationButton
              key={`cite-${i}`}
              num={refNum}
              source={chunk.source}
              onClick={() => toggle(refNum)}
            />,
          );
        } else {
          result.push(`[${parts[i]}]`);
        }
      }
    }
    return <>{result}</>;
  }

  if (Array.isArray(children)) {
    return children.map((child, i) => (
      <span key={i}>{processChildren(child, chunks, toggle)}</span>
    ));
  }

  return children;
}

function AssistantMessage({
  content,
  chunksUsed,
  sources,
  chunks,
}: {
  content: string;
  chunksUsed: number;
  sources: string[];
  chunks: ChunkReference[];
}) {
  const [expandedCitations, setExpandedCitations] = useState<Set<number>>(
    new Set(),
  );

  function toggleCitation(n: number) {
    setExpandedCitations((prev) => {
      const next = new Set(prev);
      if (next.has(n)) next.delete(n);
      else next.add(n);
      return next;
    });
  }

  // Custom components that process citation references in text
  const citationComponents = chunks.length > 0
    ? {
        p: ({ children }: { children?: React.ReactNode }) => (
          <p>{processChildren(children, chunks, toggleCitation)}</p>
        ),
        li: ({ children }: { children?: React.ReactNode }) => (
          <li>{processChildren(children, chunks, toggleCitation)}</li>
        ),
        td: ({ children }: { children?: React.ReactNode }) => (
          <td>{processChildren(children, chunks, toggleCitation)}</td>
        ),
        blockquote: ({ children }: { children?: React.ReactNode }) => (
          <blockquote>{processChildren(children, chunks, toggleCitation)}</blockquote>
        ),
      }
    : undefined;

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
          <ReactMarkdown components={citationComponents}>
            {content}
          </ReactMarkdown>
        </div>

        {/* Expanded citation panels */}
        {expandedCitations.size > 0 && (
          <div className="mt-4 space-y-2">
            {Array.from(expandedCitations)
              .sort((a, b) => a - b)
              .map((n) => {
                const chunk = chunks.find((c) => c.index === n);
                if (!chunk) return null;
                return (
                  <div
                    key={n}
                    className="border border-neutral-200 dark:border-neutral-700 p-3 text-xs bg-neutral-50 dark:bg-neutral-900"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-neutral-700 dark:text-neutral-300">
                        [{n}] {chunk.source}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="text-neutral-400 dark:text-neutral-500">
                          {(chunk.similarity * 100).toFixed(0)}% match
                        </span>
                        <button
                          onClick={() => toggleCitation(n)}
                          className="text-neutral-400 hover:text-neutral-600 dark:text-neutral-500 dark:hover:text-neutral-300"
                        >
                          &times;
                        </button>
                      </div>
                    </div>
                    <p className="text-neutral-600 dark:text-neutral-400 whitespace-pre-wrap leading-relaxed">
                      {chunk.content}
                    </p>
                  </div>
                );
              })}
          </div>
        )}

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
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

  // Upload
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploadFiles, setUploadFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Scroll ref
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll when messages change or streaming text updates
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

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

  // Ask a question (streaming)
  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setIsStreaming(true);
    setError("");
    setStreamingText("");

    const userContent = question;
    setQuestion("");

    // Optimistic: show user message immediately
    const tempUserMsg: Message = {
      id: Date.now(),
      role: "user",
      content: userContent,
      chunks_used: 0,
      sources: [],
      chunks_data: [],
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
      const res = await fetch("/api/ask/stream", {
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

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let accumulated = "";
      let metadata: {
        chunks_used: number;
        sources: string[];
        chunks: ChunkReference[];
      } | null = null;
      let newConversationId: string | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6);
          if (!jsonStr.trim()) continue;

          try {
            const event = JSON.parse(jsonStr);
            if (event.type === "delta") {
              accumulated += event.text;
              setStreamingText(accumulated);
            } else if (event.type === "metadata") {
              metadata = {
                chunks_used: event.chunks_used,
                sources: event.sources,
                chunks: event.chunks || [],
              };
            } else if (event.type === "conversation_id") {
              newConversationId = event.conversation_id;
            } else if (event.type === "error") {
              throw new Error(event.message);
            }
          } catch (parseErr) {
            if (
              parseErr instanceof Error &&
              parseErr.message !== "Unexpected end of JSON input"
            ) {
              throw parseErr;
            }
          }
        }
      }

      // Update conversation ID if this was a new conversation
      if (newConversationId && !activeConversationId) {
        setActiveConversationId(newConversationId);
      }

      // Add assistant message with final content
      const assistantMsg: Message = {
        id: Date.now() + 1,
        role: "assistant",
        content: accumulated,
        chunks_used: metadata?.chunks_used ?? 0,
        sources: metadata?.sources ?? [],
        chunks_data: metadata?.chunks ?? [],
        inference_level: inferenceLevel,
        evidence_depth: evidenceDepth,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setStreamingText("");

      // Refresh sidebar (title may have been generated)
      loadConversations();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      // Remove optimistic user message on error
      setMessages((prev) => prev.slice(0, -1));
      setStreamingText("");
    } finally {
      setLoading(false);
      setIsStreaming(false);
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

  // File upload handlers
  function handleFileSelect(selectedFiles: FileList | null) {
    if (!selectedFiles) return;
    const valid = Array.from(selectedFiles).filter((f) => {
      const ext = "." + f.name.split(".").pop()?.toLowerCase();
      return SUPPORTED_EXTENSIONS.includes(ext);
    });
    setUploadFiles((prev) => [...prev, ...valid]);
  }

  function removeFile(index: number) {
    setUploadFiles((prev) => prev.filter((_, i) => i !== index));
  }

  async function handleUpload() {
    if (uploadFiles.length === 0) return;
    setUploading(true);
    setUploadResult("");

    try {
      const formData = new FormData();
      for (const file of uploadFiles) {
        formData.append("files", file);
      }

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(
          Array.isArray(detail.detail)
            ? detail.detail.map((e: { error: string }) => e.error).join("; ")
            : detail.detail || `Error ${res.status}`
        );
      }

      const data = await res.json();
      let summary = `${data.count} file(s) uploaded successfully.`;
      if (data.errors?.length > 0) {
        summary += ` ${data.errors.length} file(s) failed.`;
      }

      setUploadResult(summary);
      setUploadFiles([]);
      setToast(summary + " Click 'Ingest documents' to process them.");
      setTimeout(() => setToast(""), 8000);

      setTimeout(() => {
        setShowUploadModal(false);
        setUploadResult("");
      }, 2000);
    } catch (e) {
      setUploadResult(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
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
              onClick={() => setShowUploadModal(true)}
              disabled={loading}
              className="flex items-center gap-1.5 text-xs tracking-wide uppercase text-neutral-500 hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100 disabled:text-neutral-300 dark:disabled:text-neutral-600 transition-colors"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Upload files
            </button>
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
                        chunks={msg.chunks_data || []}
                      />
                    )
                  )}

                  {/* Streaming response */}
                  {isStreaming && streamingText && (
                    <div className="mb-8">
                      <div className="border border-neutral-200 dark:border-neutral-800 p-6">
                        <div className="flex items-center gap-3 mb-4 pb-4 border-b border-neutral-100 dark:border-neutral-800">
                          <span className="text-xs tracking-widest uppercase text-neutral-400 dark:text-neutral-500">
                            Response
                          </span>
                          <span className="text-xs text-neutral-400 dark:text-neutral-500 animate-pulse">
                            streaming&hellip;
                          </span>
                        </div>
                        <div className="prose prose-neutral dark:prose-invert max-w-none prose-headings:font-black prose-headings:uppercase prose-headings:tracking-tight prose-h2:text-lg prose-h3:text-base prose-p:leading-relaxed">
                          <ReactMarkdown>{streamingText}</ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Thinking indicator (before first token) */}
                  {loading && !streamingText && (
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
                    {loading && !streamingText
                      ? "Thinking\u2026"
                      : isStreaming
                        ? "Streaming\u2026"
                        : "Ask"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Upload modal */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/30 dark:bg-black/50"
            onClick={() => {
              setShowUploadModal(false);
              setUploadFiles([]);
              setUploadResult("");
            }}
          />

          {/* Modal content */}
          <div className="relative bg-white dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 w-full max-w-lg mx-4 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-sm tracking-widest uppercase text-neutral-400 dark:text-neutral-500">
                Upload Documents
              </h2>
              <button
                onClick={() => {
                  setShowUploadModal(false);
                  setUploadFiles([]);
                  setUploadResult("");
                }}
                className="text-neutral-400 hover:text-neutral-900 dark:text-neutral-600 dark:hover:text-neutral-100 text-lg"
              >
                &times;
              </button>
            </div>

            {/* Drop zone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFileSelect(e.dataTransfer.files);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed p-8 text-center cursor-pointer transition-colors ${
                dragOver
                  ? "border-neutral-900 dark:border-neutral-100 bg-neutral-50 dark:bg-neutral-900"
                  : "border-neutral-300 dark:border-neutral-700 hover:border-neutral-400 dark:hover:border-neutral-600"
              }`}
            >
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                Drag and drop files here, or click to browse
              </p>
              <p className="text-xs text-neutral-400 dark:text-neutral-600 mt-2">
                Supports: {SUPPORTED_EXTENSIONS.join(", ")}
              </p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={SUPPORTED_EXTENSIONS.join(",")}
                className="hidden"
                onChange={(e) => {
                  handleFileSelect(e.target.files);
                  e.target.value = "";
                }}
              />
            </div>

            {/* Selected files list */}
            {uploadFiles.length > 0 && (
              <div className="mt-4 max-h-40 overflow-y-auto">
                {uploadFiles.map((file, i) => (
                  <div
                    key={`${file.name}-${i}`}
                    className="flex items-center justify-between py-1.5 border-b border-neutral-100 dark:border-neutral-900"
                  >
                    <span className="text-xs text-neutral-700 dark:text-neutral-300 truncate flex-1">
                      {file.name}
                    </span>
                    <span className="text-xs text-neutral-400 dark:text-neutral-600 mx-3">
                      {(file.size / 1024).toFixed(0)} KB
                    </span>
                    <button
                      onClick={() => removeFile(i)}
                      className="text-neutral-400 hover:text-red-500 dark:text-neutral-600 dark:hover:text-red-400 text-xs"
                    >
                      &times;
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Upload result */}
            {uploadResult && (
              <p className="mt-3 text-xs text-neutral-600 dark:text-neutral-400">
                {uploadResult}
              </p>
            )}

            {/* Actions */}
            <div className="mt-6 flex items-center justify-between">
              <span className="text-xs text-neutral-400 dark:text-neutral-600">
                {uploadFiles.length} file(s) selected
              </span>
              <button
                onClick={handleUpload}
                disabled={uploading || uploadFiles.length === 0}
                className="px-5 py-2 bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 text-xs tracking-wide uppercase font-semibold hover:bg-neutral-700 dark:hover:bg-neutral-300 disabled:bg-neutral-200 disabled:text-neutral-400 dark:disabled:bg-neutral-800 dark:disabled:text-neutral-600 transition-colors"
              >
                {uploading ? "Uploading\u2026" : "Upload"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
