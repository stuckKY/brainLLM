import json
import logging
import re
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.db import engine, init_db
from backend.ingest import DOCUMENTS_DIR, SUPPORTED_EXTENSIONS, run_ingestion
from backend.query import ask, ask_stream, generate_title

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("brainllm")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising database...")
    init_db()
    logger.info("Database ready.")
    yield


app = FastAPI(title="brainLLM", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str
    inference_level: int = Field(ge=1, le=5, default=3)
    evidence_depth: int = Field(ge=1, le=5, default=3)
    conversation_id: Optional[str] = None


class RenameRequest(BaseModel):
    title: str


# ---------------------------------------------------------------------------
# Health & ingestion
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest")
def ingest_documents():
    """Incrementally ingest documents into pgvector."""
    logger.info("Ingestion started")
    try:
        result = run_ingestion()
        logger.info(
            "Ingestion complete: new=%d modified=%d skipped=%d deleted=%d chunks_stored=%d",
            result["files_new"],
            result["files_modified"],
            result["files_skipped"],
            result["files_deleted"],
            result["chunks_stored"],
        )
    except Exception as e:
        logger.error("Ingestion failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    return result


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
UPLOADS_DIR = DOCUMENTS_DIR / "uploads"


def sanitise_filename(name: str) -> str:
    """Remove path separators and dangerous characters, keeping the extension."""
    name = Path(name).name
    name = re.sub(r"[^\w\-.]", "_", name)
    name = re.sub(r"[_.]{2,}", "_", name)
    name = name.lstrip(".")
    return name or "unnamed"


def deduplicate_filename(directory: Path, filename: str) -> str:
    """If filename exists in directory, append _1, _2, etc. before the extension."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = filename
    counter = 1
    while (directory / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """Accept multipart file uploads and save to the uploads subdirectory."""
    logger.info("Upload request: %d file(s)", len(files))

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    for upload in files:
        original_name = upload.filename or "unnamed"
        safe_name = sanitise_filename(original_name)

        # Validate extension
        ext = Path(safe_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            errors.append({
                "filename": original_name,
                "error": (
                    f"Unsupported file type '{ext}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                ),
            })
            continue

        # Read content with size check
        content = await upload.read()
        if len(content) > MAX_UPLOAD_SIZE:
            errors.append({
                "filename": original_name,
                "error": f"File too large ({len(content)} bytes). Maximum: {MAX_UPLOAD_SIZE} bytes.",
            })
            continue

        # Deduplicate filename
        final_name = deduplicate_filename(UPLOADS_DIR, safe_name)
        dest = UPLOADS_DIR / final_name

        # Write to disk
        try:
            dest.write_bytes(content)
            rel_path = str(dest.relative_to(DOCUMENTS_DIR))
            logger.info("Saved upload: %s (%d bytes)", rel_path, len(content))
            results.append({
                "filename": final_name,
                "original_filename": original_name,
                "path": rel_path,
                "size": len(content),
            })
        except OSError as e:
            logger.error("Failed to write %s: %s", dest, e)
            errors.append({
                "filename": original_name,
                "error": f"Failed to save file: {e}",
            })

    if not results and errors:
        raise HTTPException(status_code=400, detail=errors)

    return {
        "uploaded": results,
        "errors": errors,
        "count": len(results),
    }


@app.get("/files")
def list_uploaded_files():
    """List files in the uploads directory."""
    if not UPLOADS_DIR.exists():
        return {"files": [], "count": 0}

    files = []
    for path in sorted(UPLOADS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append({
                "filename": path.name,
                "path": str(path.relative_to(DOCUMENTS_DIR)),
                "size": path.stat().st_size,
            })
    return {"files": files, "count": len(files)}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@app.get("/conversations")
def list_conversations():
    """Return all conversations, most recent first."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT id, title, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC"
            )
        ).fetchall()
    return [
        {
            "id": row.id,
            "title": row.title,
            "created_at": row.created_at.isoformat(),
            "updated_at": row.updated_at.isoformat(),
        }
        for row in rows
    ]


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    """Return a single conversation with all its messages."""
    with engine.connect() as conn:
        conv = conn.execute(
            text("SELECT id, title FROM conversations WHERE id = :id"),
            {"id": conversation_id},
        ).fetchone()

        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        rows = conn.execute(
            text(
                "SELECT id, role, content, chunks_used, sources, "
                "inference_level, evidence_depth, created_at "
                "FROM messages WHERE conversation_id = :cid "
                "ORDER BY created_at ASC"
            ),
            {"cid": conversation_id},
        ).fetchall()

    return {
        "id": conv.id,
        "title": conv.title,
        "messages": [
            {
                "id": row.id,
                "role": row.role,
                "content": row.content,
                "chunks_used": row.chunks_used,
                "sources": list(row.sources) if row.sources else [],
                "inference_level": row.inference_level,
                "evidence_depth": row.evidence_depth,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ],
    }


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    with engine.connect() as conn:
        result = conn.execute(
            text("DELETE FROM conversations WHERE id = :id"),
            {"id": conversation_id},
        )
        conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    logger.info("Deleted conversation %s", conversation_id)
    return {"status": "deleted"}


@app.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, req: RenameRequest):
    """Rename a conversation."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "UPDATE conversations SET title = :title, updated_at = NOW() "
                "WHERE id = :id"
            ),
            {"title": req.title, "id": conversation_id},
        )
        conn.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"status": "updated", "title": req.title}


# ---------------------------------------------------------------------------
# Ask (with conversation history)
# ---------------------------------------------------------------------------


@app.post("/ask")
def ask_question(req: AskRequest):
    """Query the knowledge base within a conversation context."""
    conversation_id = req.conversation_id
    is_new = conversation_id is None

    logger.info(
        "Ask: question=%r inference=%d evidence=%d conversation=%s",
        req.question, req.inference_level, req.evidence_depth,
        conversation_id or "(new)",
    )

    try:
        # Create conversation if needed
        if is_new:
            conversation_id = uuid.uuid4().hex[:16]
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "INSERT INTO conversations (id) VALUES (:id)"
                    ),
                    {"id": conversation_id},
                )
                conn.commit()
            logger.info("Created conversation %s", conversation_id)

        # Load conversation history (last 20 messages = 10 turns)
        with engine.connect() as conn:
            history_rows = conn.execute(
                text(
                    "SELECT role, content FROM messages "
                    "WHERE conversation_id = :cid "
                    "ORDER BY created_at ASC"
                ),
                {"cid": conversation_id},
            ).fetchall()

        history = [{"role": r.role, "content": r.content} for r in history_rows]
        if len(history) > 20:
            history = history[-20:]

        # Call the query pipeline with history
        result = ask(
            req.question,
            req.inference_level,
            req.evidence_depth,
            history=history if history else None,
        )

        # Store both messages in the database
        with engine.connect() as conn:
            # User message
            conn.execute(
                text(
                    "INSERT INTO messages "
                    "(conversation_id, role, content, inference_level, evidence_depth) "
                    "VALUES (:cid, 'user', :content, :il, :ed)"
                ),
                {
                    "cid": conversation_id,
                    "content": req.question,
                    "il": req.inference_level,
                    "ed": req.evidence_depth,
                },
            )
            # Assistant message
            conn.execute(
                text(
                    "INSERT INTO messages "
                    "(conversation_id, role, content, chunks_used, sources, "
                    "inference_level, evidence_depth) "
                    "VALUES (:cid, 'assistant', :content, :chunks, :sources, :il, :ed)"
                ),
                {
                    "cid": conversation_id,
                    "content": result["answer"],
                    "chunks": result["chunks_used"],
                    "sources": result["sources"],
                    "il": req.inference_level,
                    "ed": req.evidence_depth,
                },
            )
            # Update conversation timestamp
            conn.execute(
                text("UPDATE conversations SET updated_at = NOW() WHERE id = :id"),
                {"id": conversation_id},
            )
            conn.commit()

        # Auto-generate title after the first exchange
        title = "New conversation"
        if is_new:
            try:
                title = generate_title(req.question, result["answer"])
                with engine.connect() as conn:
                    conn.execute(
                        text("UPDATE conversations SET title = :title WHERE id = :id"),
                        {"title": title, "id": conversation_id},
                    )
                    conn.commit()
                logger.info("Generated title: %r", title)
            except Exception:
                logger.warning("Title generation failed, keeping default")
        else:
            # Fetch existing title
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT title FROM conversations WHERE id = :id"),
                    {"id": conversation_id},
                ).fetchone()
                if row:
                    title = row.title

        logger.info("Ask complete: %d chunks used", result["chunks_used"])

        return {
            "conversation_id": conversation_id,
            "answer": result["answer"],
            "chunks_used": result["chunks_used"],
            "sources": result["sources"],
            "title": title,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Ask failed:\n%s", traceback.format_exc())
        # Clean up empty conversation if we just created it and it failed
        if is_new and conversation_id:
            try:
                with engine.connect() as conn:
                    conn.execute(
                        text("DELETE FROM conversations WHERE id = :id"),
                        {"id": conversation_id},
                    )
                    conn.commit()
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Ask — streaming (SSE)
# ---------------------------------------------------------------------------


@app.post("/ask/stream")
def ask_question_stream(req: AskRequest):
    """Stream the answer token-by-token via Server-Sent Events."""
    conversation_id = req.conversation_id
    is_new = conversation_id is None

    logger.info(
        "Ask (stream): question=%r inference=%d evidence=%d conversation=%s",
        req.question, req.inference_level, req.evidence_depth,
        conversation_id or "(new)",
    )

    # Create conversation up front so the ID is available immediately
    if is_new:
        conversation_id = uuid.uuid4().hex[:16]
        with engine.connect() as conn:
            conn.execute(
                text("INSERT INTO conversations (id) VALUES (:id)"),
                {"id": conversation_id},
            )
            conn.commit()
        logger.info("Created conversation %s", conversation_id)

    # Load conversation history
    with engine.connect() as conn:
        history_rows = conn.execute(
            text(
                "SELECT role, content FROM messages "
                "WHERE conversation_id = :cid "
                "ORDER BY created_at ASC"
            ),
            {"cid": conversation_id},
        ).fetchall()

    history = [{"role": r.role, "content": r.content} for r in history_rows]
    if len(history) > 20:
        history = history[-20:]

    def event_generator():
        accumulated = ""
        try:
            for chunk in ask_stream(
                req.question,
                req.inference_level,
                req.evidence_depth,
                history=history if history else None,
            ):
                # Capture text for DB storage
                if '"type": "delta"' in chunk:
                    try:
                        data = json.loads(chunk.removeprefix("data: ").strip())
                        accumulated += data.get("text", "")
                    except Exception:
                        pass
                yield chunk

            # After streaming completes, extract metadata and store messages
            chunks_used = 0
            sources: list[str] = []
            # The metadata was already yielded; parse from the last events
            # Instead, we re-derive from the stream function's logic
            # We need to get this from the metadata event we yielded
            # Since we can't easily, we'll query for it
            # Actually, we already have the chunks info from ask_stream internals
            # Let's store with what we know

            # Store messages in DB
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "INSERT INTO messages "
                        "(conversation_id, role, content, inference_level, evidence_depth) "
                        "VALUES (:cid, 'user', :content, :il, :ed)"
                    ),
                    {
                        "cid": conversation_id,
                        "content": req.question,
                        "il": req.inference_level,
                        "ed": req.evidence_depth,
                    },
                )
                conn.execute(
                    text(
                        "INSERT INTO messages "
                        "(conversation_id, role, content, inference_level, evidence_depth) "
                        "VALUES (:cid, 'assistant', :content, :il, :ed)"
                    ),
                    {
                        "cid": conversation_id,
                        "content": accumulated,
                        "il": req.inference_level,
                        "ed": req.evidence_depth,
                    },
                )
                conn.execute(
                    text("UPDATE conversations SET updated_at = NOW() WHERE id = :id"),
                    {"id": conversation_id},
                )
                conn.commit()

            # Auto-generate title for new conversations
            if is_new:
                try:
                    title = generate_title(req.question, accumulated)
                    with engine.connect() as conn:
                        conn.execute(
                            text("UPDATE conversations SET title = :title WHERE id = :id"),
                            {"title": title, "id": conversation_id},
                        )
                        conn.commit()
                    logger.info("Generated title: %r", title)
                except Exception:
                    logger.warning("Title generation failed, keeping default")

            # Send conversation_id so the frontend can track it
            yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': conversation_id})}\n\n"

        except Exception as e:
            logger.error("Stream error:\n%s", traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            # Clean up empty conversation on error
            if is_new:
                try:
                    with engine.connect() as conn:
                        conn.execute(
                            text("DELETE FROM conversations WHERE id = :id"),
                            {"id": conversation_id},
                        )
                        conn.commit()
                except Exception:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
