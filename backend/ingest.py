import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pgvector.sqlalchemy import Vector  # noqa: F401
from sqlalchemy import text

from backend.db import engine

DOCUMENTS_DIR = Path(os.environ.get("DOCUMENTS_DIR", "documents"))
EMBEDDING_MODEL = "text-embedding-3-small"

openai_client = OpenAI()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
)


def load_documents() -> list:
    """Walk the documents directory and load all supported file types."""
    docs = []

    for md_path in DOCUMENTS_DIR.rglob("*.md"):
        loader = UnstructuredMarkdownLoader(str(md_path))
        docs.extend(loader.load())

    for pdf_path in DOCUMENTS_DIR.rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    for pptx_path in DOCUMENTS_DIR.rglob("*.pptx"):
        loader = UnstructuredPowerPointLoader(str(pptx_path))
        docs.extend(loader.load())

    return docs


def chunk_documents(docs: list) -> list:
    """Split loaded documents into smaller chunks for embedding."""
    return splitter.split_documents(docs)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of text strings."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def store_chunks(chunks: list) -> int:
    """Embed chunks and insert them into the pgvector chunks table.

    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    batch_size = 100
    total_stored = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk.page_content for chunk in batch]
        sources = [chunk.metadata.get("source", "unknown") for chunk in batch]
        embeddings = embed_texts(texts)

        with engine.connect() as conn:
            for content, source, embedding in zip(texts, sources, embeddings):
                conn.execute(
                    text(
                        "INSERT INTO chunks (content, source, embedding) "
                        "VALUES (:content, :source, :embedding)"
                    ),
                    {
                        "content": content,
                        "source": source,
                        "embedding": str(embedding),
                    },
                )
            conn.commit()

        total_stored += len(batch)

    return total_stored


def run_ingestion() -> dict:
    """Full ingestion pipeline: load → chunk → embed → store."""
    docs = load_documents()
    chunks = chunk_documents(docs)

    # Clear existing chunks before re-ingesting
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM chunks"))
        conn.commit()

    stored = store_chunks(chunks)

    return {
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunks_stored": stored,
    }
