import hashlib
import logging
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
from backend.ocr import IMAGE_EXTENSIONS, is_scanned_pdf, ocr_pdf, ocr_image

logger = logging.getLogger("brainllm")

DOCUMENTS_DIR = Path(os.environ.get("DOCUMENTS_DIR", "documents"))
EMBEDDING_MODEL = "text-embedding-3-small"
SUPPORTED_EXTENSIONS = {".md", ".pdf", ".pptx"} | IMAGE_EXTENSIONS

openai_client = OpenAI()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def hash_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def scan_files() -> dict[str, Path]:
    """Walk DOCUMENTS_DIR and return {relative_path: absolute_path} for supported files."""
    result: dict[str, Path] = {}
    for ext in SUPPORTED_EXTENSIONS:
        for abs_path in DOCUMENTS_DIR.rglob(f"*{ext}"):
            rel = str(abs_path.relative_to(DOCUMENTS_DIR))
            result[rel] = abs_path
    return result


def get_tracked_files() -> dict[str, str]:
    """Return {path: sha256} for all files in the tracking table."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT path, sha256 FROM files")).fetchall()
    return {row.path: row.sha256 for row in rows}


def load_single_file(abs_path: Path) -> list:
    """Load a single file using the appropriate LangChain loader.

    For PDFs, falls back to OCR if the file appears to be scanned.
    For image files, uses OCR directly.
    """
    ext = abs_path.suffix.lower()

    if ext == ".md":
        loader = UnstructuredMarkdownLoader(str(abs_path))
        docs = loader.load()
    elif ext == ".pdf":
        loader = PyPDFLoader(str(abs_path))
        docs = loader.load()
        # Fall back to OCR if the PDF appears to be scanned
        if is_scanned_pdf(docs, len(docs)):
            logger.info("Falling back to OCR for scanned PDF: %s", abs_path)
            docs = ocr_pdf(str(abs_path))
    elif ext == ".pptx":
        loader = UnstructuredPowerPointLoader(str(abs_path))
        docs = loader.load()
    elif ext in IMAGE_EXTENSIONS:
        docs = ocr_image(str(abs_path))
    else:
        return []

    # Strip NUL bytes — PDFs and PPTX files often contain them
    # and PostgreSQL TEXT columns reject them.
    for doc in docs:
        doc.page_content = doc.page_content.replace("\x00", "")
    return docs


def delete_chunks_for_files(paths: list[str]):
    """Delete chunks and file tracking rows for the given relative paths."""
    if not paths:
        return
    with engine.connect() as conn:
        for path in paths:
            conn.execute(
                text("DELETE FROM chunks WHERE source = :source"),
                {"source": path},
            )
            conn.execute(
                text("DELETE FROM files WHERE path = :path"),
                {"path": path},
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Chunking & embedding (unchanged)
# ---------------------------------------------------------------------------


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
        texts = [chunk.page_content.replace("\x00", "") for chunk in batch]
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_ingestion() -> dict:
    """Incremental ingestion: only process new/modified files, remove deleted."""

    # 1. Scan filesystem
    on_disk = scan_files()  # {rel_path: abs_path}
    logger.info("Found %d files on disk", len(on_disk))

    # 2. Compute hashes for every file on disk
    disk_hashes: dict[str, str] = {}
    for rel_path, abs_path in on_disk.items():
        disk_hashes[rel_path] = hash_file(abs_path)

    # 3. Get previously tracked files from DB
    tracked = get_tracked_files()  # {rel_path: sha256}

    # 4. Classify files
    disk_paths = set(disk_hashes.keys())
    tracked_paths = set(tracked.keys())

    new_paths = disk_paths - tracked_paths
    deleted_paths = tracked_paths - disk_paths
    common_paths = disk_paths & tracked_paths

    modified_paths = {p for p in common_paths if disk_hashes[p] != tracked[p]}
    skipped_paths = common_paths - modified_paths

    logger.info(
        "Classification: new=%d modified=%d unchanged=%d deleted=%d",
        len(new_paths),
        len(modified_paths),
        len(skipped_paths),
        len(deleted_paths),
    )

    # 5. First-run-after-upgrade: if files table was empty but chunks table
    #    has data from the old wipe-and-reload system, clear stale chunks
    #    (they store absolute paths which won't match relative source keys).
    if not tracked:
        with engine.connect() as conn:
            existing = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar()
            if existing and existing > 0:
                logger.info(
                    "Upgrade detected — clearing %d legacy chunks", existing
                )
                conn.execute(text("DELETE FROM chunks"))
                conn.commit()

    # 6. Delete chunks for deleted and modified files
    to_delete = sorted(deleted_paths | modified_paths)
    if to_delete:
        logger.info("Removing chunks for %d files: %s", len(to_delete), to_delete)
        delete_chunks_for_files(to_delete)

    # 7. Process new and modified files
    to_process = sorted(new_paths | modified_paths)
    total_chunks_stored = 0

    for rel_path in to_process:
        abs_path = on_disk[rel_path]
        logger.info("Processing: %s", rel_path)

        # Load
        docs = load_single_file(abs_path)

        # Normalise source metadata to the relative path
        for doc in docs:
            doc.metadata["source"] = rel_path

        # Chunk
        chunks = chunk_documents(docs)

        # Embed and store
        stored = store_chunks(chunks)
        total_chunks_stored += stored

        # Track in files table (upsert)
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO files (path, sha256, chunk_count) "
                    "VALUES (:path, :sha256, :chunk_count) "
                    "ON CONFLICT (path) DO UPDATE SET "
                    "sha256 = :sha256, chunk_count = :chunk_count, ingested_at = NOW()"
                ),
                {
                    "path": rel_path,
                    "sha256": disk_hashes[rel_path],
                    "chunk_count": stored,
                },
            )
            conn.commit()

        logger.info("  → %d chunks stored for %s", stored, rel_path)

    # 8. Return detailed summary
    return {
        "files_new": len(new_paths),
        "files_modified": len(modified_paths),
        "files_skipped": len(skipped_paths),
        "files_deleted": len(deleted_paths),
        "chunks_stored": total_chunks_stored,
        "details": {
            "new": sorted(new_paths),
            "modified": sorted(modified_paths),
            "skipped": sorted(skipped_paths),
            "deleted": sorted(deleted_paths),
        },
    }
