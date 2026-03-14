import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.db import init_db
from backend.ingest import run_ingestion
from backend.query import ask

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


class AskRequest(BaseModel):
    question: str
    inference_level: int = Field(ge=1, le=5, default=3)
    evidence_depth: int = Field(ge=1, le=5, default=3)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest")
def ingest_documents():
    """Load, chunk, embed, and store all documents in pgvector."""
    logger.info("Ingestion started")
    try:
        result = run_ingestion()
        logger.info("Ingestion complete: %s", result)
    except Exception as e:
        logger.error("Ingestion failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/ask")
def ask_question(req: AskRequest):
    """Query the knowledge base with slider-controlled retrieval and inference."""
    logger.info("Ask: question=%r inference=%d evidence=%d", req.question, req.inference_level, req.evidence_depth)
    try:
        result = ask(req.question, req.inference_level, req.evidence_depth)
        logger.info("Ask complete: %d chunks used", result["chunks_used"])
    except Exception as e:
        logger.error("Ask failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
    return result
