import os

from sqlalchemy import create_engine, text


def _build_database_url() -> str:
    """Build the database URL from env vars.

    Accepts either a full DATABASE_URL or individual components:
      DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    name = os.environ.get("DB_NAME", "brainllm")
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


engine = create_engine(_build_database_url())


def init_db():
    """Create pgvector extension and all tables if they don't exist."""
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
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS files (
                id          SERIAL PRIMARY KEY,
                path        TEXT UNIQUE NOT NULL,
                sha256      TEXT NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                ingested_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         TEXT PRIMARY KEY,
                title      TEXT NOT NULL DEFAULT 'New conversation',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                id              SERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content         TEXT NOT NULL,
                chunks_used     INTEGER DEFAULT 0,
                sources         TEXT[] DEFAULT '{}',
                inference_level INTEGER DEFAULT 3,
                evidence_depth  INTEGER DEFAULT 3,
                created_at      TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages(conversation_id, created_at)
        """))
        conn.commit()
