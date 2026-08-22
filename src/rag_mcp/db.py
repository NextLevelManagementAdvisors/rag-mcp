"""PostgreSQL + pgvector storage and cosine-similarity search."""
from __future__ import annotations

import json
import re

from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from . import config

_pool: AsyncConnectionPool | None = None


async def _configure(conn) -> None:
    # pgvector must exist before the vector type adapter can be registered.
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await conn.commit()
    await register_vector_async(conn)


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        _pool = AsyncConnectionPool(
            config.DATABASE_URL, min_size=1, max_size=5, open=False, configure=_configure
        )
        await _pool.open()
        await _init_schema()
    return _pool


async def _init_schema() -> None:
    async with _pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          BIGSERIAL PRIMARY KEY,
                collection  TEXT NOT NULL,
                content     TEXT NOT NULL,
                metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding   vector({config.EMBED_DIM}) NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS chunks_collection_idx ON chunks (collection)")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_idx "
            "ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )
        await _check_embedding_dim(conn)


async def _check_embedding_dim(conn) -> None:
    # `CREATE TABLE IF NOT EXISTS` above is a no-op when the table already exists,
    # so a changed EMBED_DIM would otherwise go unnoticed until the first insert
    # fails with an opaque pgvector dimension-mismatch error.
    cur = await conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
    )
    row = await cur.fetchone()
    match = re.fullmatch(r"vector\((\d+)\)", row[0]) if row else None
    if not match:
        raise RuntimeError(f"could not determine dimension of chunks.embedding column (got {row!r})")
    actual_dim = int(match.group(1))
    if actual_dim != config.EMBED_DIM:
        raise RuntimeError(
            f"chunks.embedding is vector({actual_dim}) but EMBED_DIM is {config.EMBED_DIM}. "
            "CREATE TABLE IF NOT EXISTS does not migrate an existing column's dimension. "
            "To switch embedding dimensions, dump the collection's `content`/`metadata`, drop "
            "the chunks table (or rename it), let the app recreate it, and re-ingest via "
            "db.insert_chunks (not ingest_text, which would re-chunk the content)."
        )


async def insert_chunks(collection: str, rows: list[tuple[str, dict, list[float]]]) -> int:
    pool = await get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        for content, meta, emb in rows:
            await cur.execute(
                "INSERT INTO chunks (collection, content, metadata, embedding) "
                "VALUES (%s, %s, %s, %s)",
                (collection, content, json.dumps(meta or {}), emb),
            )
    return len(rows)


async def search(collection: str, query_embedding: list[float], k: int) -> list[dict]:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            """
            SELECT content, metadata, 1 - (embedding <=> %s::vector) AS score
            FROM chunks
            WHERE collection = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, collection, query_embedding, k),
        )
        rows = await cur.fetchall()
    return [{"content": r[0], "metadata": r[1], "score": round(float(r[2]), 4)} for r in rows]


async def list_collections() -> list[dict]:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT collection, count(*) FROM chunks GROUP BY collection ORDER BY collection"
        )
        rows = await cur.fetchall()
    return [{"collection": r[0], "chunks": r[1]} for r in rows]


async def delete_collection(collection: str) -> int:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute("DELETE FROM chunks WHERE collection = %s", (collection,))
        return cur.rowcount
