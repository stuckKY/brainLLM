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
    """Create pgvector extension and chunks table if they don't exist."""
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
