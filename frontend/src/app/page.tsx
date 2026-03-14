"use client";

import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";

function getApiUrl() {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
}

const API_URL = getApiUrl();

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

interface AskResponse {
  answer: string;
  chunks_used: number;
  sources: string[];
}

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

export default function Home() {
  const { dark, toggle } = useTheme();
  const [question, setQuestion] = useState("");
  const [inferenceLevel, setInferenceLevel] = useState(3);
  const [evidenceDepth, setEvidenceDepth] = useState(3);
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAsk() {
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    setResponse(null);

    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          inference_level: inferenceLevel,
          evidence_depth: evidenceDepth,
        }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Error ${res.status}`);
      }

      setResponse(await res.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function handleIngest() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch(`${API_URL}/ingest`, { method: "POST" });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Error ${res.status}`);
      }
      const data = await res.json();
      setResponse({
        answer: `Ingestion complete — ${data.documents_loaded} documents loaded, ${data.chunks_stored} chunks stored.`,
        chunks_used: 0,
        sources: [],
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ingestion failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="border-b border-neutral-200 dark:border-neutral-800">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
          <span className="text-xs tracking-widest uppercase text-neutral-400 dark:text-neutral-500">
            Personal Knowledge Base
          </span>
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

      <main className="max-w-3xl mx-auto px-6 pt-16 pb-24">
        {/* Title */}
        <h1 className="text-5xl sm:text-6xl md:text-7xl font-black uppercase leading-[0.95] tracking-tight mb-6">
          Ask Your
          <br />
          Brain.
        </h1>
        <p className="text-neutral-500 dark:text-neutral-400 text-base mb-12 max-w-md">
          Query your personal document corpus using natural language.
          Adjust inference and evidence depth to control how the AI reasons.
        </p>

        {/* Query card */}
        <div className="border border-neutral-200 dark:border-neutral-800 p-6 mb-8">
          <textarea
            className="w-full bg-transparent text-neutral-900 dark:text-neutral-100 placeholder-neutral-400 dark:placeholder-neutral-600 resize-none focus:outline-none text-base leading-relaxed"
            rows={3}
            placeholder="What do your notes say about..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleAsk();
            }}
          />

          {/* Sliders */}
          <div className="grid grid-cols-2 gap-8 mt-4 pt-4 border-t border-neutral-100 dark:border-neutral-800">
            <div>
              <div className="flex items-baseline justify-between mb-2">
                <span className="text-xs tracking-wide uppercase text-neutral-400 dark:text-neutral-500">
                  Inference
                </span>
                <span className="text-xs font-semibold text-neutral-700 dark:text-neutral-300">
                  {inferenceLevel} &mdash; {INFERENCE_LABELS[inferenceLevel]}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={5}
                step={1}
                value={inferenceLevel}
                onChange={(e) => setInferenceLevel(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <div className="flex items-baseline justify-between mb-2">
                <span className="text-xs tracking-wide uppercase text-neutral-400 dark:text-neutral-500">
                  Evidence
                </span>
                <span className="text-xs font-semibold text-neutral-700 dark:text-neutral-300">
                  {evidenceDepth} &mdash; {EVIDENCE_LABELS[evidenceDepth]}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={5}
                step={1}
                value={evidenceDepth}
                onChange={(e) => setEvidenceDepth(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>

          {/* Ask button */}
          <div className="mt-5 pt-4 border-t border-neutral-100 dark:border-neutral-800 flex items-center justify-between">
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

        {/* Error */}
        {error && (
          <div className="border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950 text-red-700 dark:text-red-400 p-4 mb-8 text-sm">
            {error}
          </div>
        )}

        {/* Response */}
        {response && (
          <div className="border border-neutral-200 dark:border-neutral-800 p-6">
            <div className="flex items-center gap-3 mb-4 pb-4 border-b border-neutral-100 dark:border-neutral-800">
              <span className="text-xs tracking-widest uppercase text-neutral-400 dark:text-neutral-500">
                Response
              </span>
              {response.chunks_used > 0 && (
                <span className="text-xs text-neutral-400 dark:text-neutral-500">
                  &middot; {response.chunks_used} chunks used
                </span>
              )}
            </div>
            <div className="prose prose-neutral dark:prose-invert max-w-none prose-headings:font-black prose-headings:uppercase prose-headings:tracking-tight prose-h2:text-lg prose-h3:text-base prose-p:leading-relaxed">
              <ReactMarkdown>{response.answer}</ReactMarkdown>
            </div>
            {response.sources.length > 0 && (
              <div className="mt-6 pt-4 border-t border-neutral-100 dark:border-neutral-800">
                <span className="text-xs tracking-widest uppercase text-neutral-400 dark:text-neutral-500">
                  Sources
                </span>
                <div className="mt-2 flex flex-wrap gap-2">
                  {response.sources.map((source) => (
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
        )}
      </main>
    </div>
  );
}
