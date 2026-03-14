import os

import anthropic
from openai import OpenAI
from sqlalchemy import text

from backend.db import engine

EMBEDDING_MODEL = "text-embedding-3-small"
CLAUDE_MODEL = "claude-sonnet-4-20250514"

EVIDENCE_K = [3, 6, 10, 15, 25]

SYSTEM_PROMPTS = {
    1: (
        "You are a knowledge assistant. Answer strictly from the provided notes only. "
        "If the answer is not in the notes, say so. Do not speculate."
    ),
    2: (
        "You are a knowledge assistant. Answer strictly from the provided notes only. "
        "If the answer is not in the notes, say so. Do not speculate."
    ),
    3: (
        "You are a knowledge assistant. Answer primarily from the provided notes. "
        "You may make tactical inferences where the evidence strongly suggests something, "
        "but flag any uncertainty clearly."
    ),
    4: (
        "You are a knowledge assistant. Answer from the provided notes. "
        "You may make reasonable inferences and connect ideas across chunks, "
        "but flag uncertainty clearly."
    ),
    5: (
        "You are a knowledge assistant. Use the provided notes as a foundation, "
        "but feel free to extrapolate novel angles, draw broader connections, "
        "and explore speculative interpretations. Use speculative language when going "
        "beyond what the notes directly state."
    ),
}

openai_client = OpenAI()
anthropic_client = anthropic.Anthropic()


def embed_query(question: str) -> list[float]:
    """Embed a single query string."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )
    return response.data[0].embedding


def search_chunks(embedding: list[float], k: int) -> list[dict]:
    """Find the k nearest chunks by cosine distance."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, content, source, "
                "1 - (embedding <=> :embedding) AS similarity "
                "FROM chunks "
                "ORDER BY embedding <=> :embedding "
                "LIMIT :k"
            ),
            {"embedding": str(embedding), "k": k},
        ).fetchall()

    return [
        {"id": row.id, "content": row.content, "source": row.source, "similarity": row.similarity}
        for row in rows
    ]


def generate_title(question: str, answer: str) -> str:
    """Generate a short conversation title from the first Q&A exchange."""
    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=30,
        messages=[
            {
                "role": "user",
                "content": (
                    "Generate a short title (3-6 words, no quotes, no punctuation) "
                    "for a conversation that starts with this exchange:\n\n"
                    f"Q: {question}\nA: {answer[:300]}"
                ),
            }
        ],
    )
    return response.content[0].text.strip().strip('"').strip("'")


def ask(
    question: str,
    inference_level: int,
    evidence_depth: int,
    history: list[dict] | None = None,
) -> dict:
    """Full query pipeline: embed question → search → call Claude.

    If history is provided (list of {"role", "content"} dicts from prior
    messages), they are prepended to the messages array so Claude has
    multi-turn context.  RAG retrieval is always fresh for the current
    question.
    """
    k = EVIDENCE_K[evidence_depth - 1]
    system_prompt = SYSTEM_PROMPTS[inference_level]

    embedding = embed_query(question)
    chunks = search_chunks(embedding, k)

    if not chunks:
        return {
            "answer": "No documents have been ingested yet. Run /ingest first.",
            "chunks_used": 0,
            "sources": [],
        }

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['content']}" for c in chunks
    )

    # Build multi-turn messages array
    messages: list[dict] = []

    if history:
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Current turn: RAG context prepended only to the latest question
    messages.append({
        "role": "user",
        "content": (
            f"Here are the relevant notes for this question:\n\n{context}\n\n"
            f"---\n\nQuestion: {question}"
        ),
    })

    message = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=messages,
    )

    return {
        "answer": message.content[0].text,
        "chunks_used": len(chunks),
        "sources": list({c["source"] for c in chunks}),
    }
