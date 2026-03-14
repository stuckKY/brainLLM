# Project Context: brainLLM

## What We're Building

A self-hosted RAG (Retrieval-Augmented Generation) application called **brainLLM** that allows querying a personal document corpus using natural language. The UI includes two sliders to control how the AI reasons over the documents.

Two deployment targets:
- **Local dev** — single Docker Compose file that includes pgvector as a sibling container
- **Production** — single `brainllm` app container connecting to an existing pgvector instance on an Unraid machine

Inspired by the "Ask My Brain" concept — an AI-powered version of your own knowledge base that cross-references your own work rather than generic internet content.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Frontend | Next.js (React) |
| Backend | FastAPI (Python) |
| Vector Database | pgvector (sibling container in dev, external Unraid instance in production) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | Claude API (Anthropic) |
| Containerisation | Docker Compose |

---

## Deployment Model

### Local Dev (`docker-compose.dev.yml`)

Two containers, both managed together:

```
brainllm-app    ← FastAPI + Next.js
brainllm-db     ← Postgres + pgvector (local only)
```

```bash
docker compose -f docker-compose.dev.yml up
```

Uses `.env.dev`:
```env
DATABASE_URL=postgresql://user:password@db:5432/brainllm
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### Production (`docker-compose.yml`)

Single container, connects to existing pgvector instance on Unraid:

```
brainllm    ← FastAPI + Next.js
              connects to external pgvector on Unraid
```

```bash
docker compose up
```

Uses `.env`:
```env
DATABASE_URL=postgresql://user:password@your-unraid-ip:5432/yourdb
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

**The app code does not change between environments.** Only the `DATABASE_URL` value differs.

---

## Project Structure

```
brainLLM/
  docker-compose.yml          ← production (single app container, external DB)
  docker-compose.dev.yml      ← local dev (app + pgvector together)
  Dockerfile
  CONTEXT.md
  .env                        ← production env vars, never commit
  .env.dev                    ← dev env vars, never commit
  .env.example                ← placeholder values, safe to commit
  backend/
    main.py
    ingest.py
    query.py
    db.py                     ← database connection + init_db()
    requirements.txt
  frontend/
    (Next.js app)
  documents/
    notes/
    pdfs/
    slideshows/
```

---

## Database Initialisation

On startup, the app runs an idempotent init function that creates the pgvector extension and chunks table if they don't already exist. Safe to run every time — will not overwrite existing data. Works the same in both dev and production.

```python
def init_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chunks (
                id        SERIAL PRIMARY KEY,
                content   TEXT,
                source    TEXT,
                embedding vector(1536)
            )
        """))
        conn.commit()
```

---

## Document Types

Three input formats, all ingested into the same vector store:

| Format | Library | Notes |
|---|---|---|
| Markdown `.md` | LangChain UnstructuredMarkdownLoader | Cleanest format, headings give natural split points and metadata |
| PDF `.pdf` | LangChain PyPDFLoader + pypdf | Digital PDFs only for now — OCR deferred to a later phase |
| PowerPoint `.pptx` | LangChain UnstructuredPowerPointLoader | Combine full deck into fewer chunks — individual slides are often too short |

**OCR is explicitly out of scope for v1.** Add later using `pytesseract` or Mistral's document API.

---

## Document Folder Structure

Mounted as a volume into the container at `/app/documents`:

```
/documents
  /notes/         ← Markdown university notes
  /pdfs/          ← Digital PDF documents
  /slideshows/    ← PowerPoint files
```

---

## The Two Sliders

### Inference Level (1–5)
Controls how speculative the system prompt is.
- **1–2:** "Answer strictly from the provided notes only."
- **3–4:** "You may make tactical inferences, but flag uncertainty."
- **5:** "Extrapolate novel angles. Use speculative language."

### Evidence Depth (1–5)
Controls **k** — how many document chunks are retrieved from the vector store before the LLM responds.
- Suggested mapping: `[3, 6, 10, 15, 25]` chunks for levels 1–5 respectively

---

## Core Architecture

```
Documents (.md, .pdf, .pptx)
        ↓
  Load & Chunk
        ↓
  Embed (OpenAI text-embedding-3-small)
        ↓
  Store in pgvector
        ↓ (at query time)
User Question → Embed → Vector Search (k = evidence_depth)
                                ↓
               Top K Chunks + System Prompt (shaped by inference_level)
                                ↓
                        Claude API → Answer
```

---

## API Endpoints

- `POST /ingest` — loads, chunks, embeds and stores all documents in pgvector
- `POST /ask` — accepts `question`, `inference_level`, `evidence_depth`, returns answer

---

## Build Order

1. Project scaffold — folder structure, `requirements.txt`, both Dockerfiles
2. Database connection — `db.py` with `init_db()` and SQLAlchemy + pgvector setup
3. Ingestion pipeline — load, chunk, embed, store in pgvector
4. Query pipeline — vector search + Claude API call with slider params
5. FastAPI routes — `/ingest` and `/ask` endpoints
6. Frontend — Next.js UI with question textarea and two sliders
7. Dev Docker — test end-to-end with `docker-compose.dev.yml`
8. Production Docker — strip out DB container, test against external Unraid pgvector

---

## Deferred / Future Work

- Create a way to have saved chat history
- OCR support for scanned PDFs (`pytesseract` or Mistral document API)
- Authentication / password protection for the UI
- Re-ingestion workflow for when new documents are added
- Metadata filtering (e.g. search only within `/notes` or a specific date range)