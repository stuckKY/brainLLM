from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.db import init_db
from backend.ingest import run_ingestion
from backend.query import ask


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
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


@app.post("/ingest")
def ingest_documents():
    """Load, chunk, embed, and store all documents in pgvector."""
    try:
        result = run_ingestion()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/ask")
def ask_question(req: AskRequest):
    """Query the knowledge base with slider-controlled retrieval and inference."""
    try:
        result = ask(req.question, req.inference_level, req.evidence_depth)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result
