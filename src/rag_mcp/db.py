"""PostgreSQL + pgvector storage and cosine-similarity search."""
from __future__ import annotations

import json
import logging
import re

from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool

from . import config

logger = logging.getLogger("rag-mcp")

_pool: AsyncConnectionPool | None = None

# Name of the staging column used by reindex_with_new_dimension() while a
# dimension migration is in progress. Never exposed outside this module.
_PENDING_COLUMN = "embedding_pending"

# Set by _check_embedding_dim() when chunks.embedding's actual width doesn't
# match config.EMBED_DIM. insert_chunks()/search() refuse to run while this is
# set (their vectors would be the wrong width for the live column); reads and
# reindex_collection() keep working so the mismatch is recoverable.
_dim_mismatch: tuple[int, int] | None = None


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


async def _column_dim(conn, column: str) -> int | None:
    cur = await conn.execute(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'chunks'::regclass AND attname = %s",
        (column,),
    )
    row = await cur.fetchone()
    if row is None:
        return None
    match = re.fullmatch(r"vector\((\d+)\)", row[0])
    if not match:
        raise RuntimeError(f"could not determine dimension of chunks.{column} column (got {row[0]!r})")
    return int(match.group(1))


async def _check_embedding_dim(conn) -> None:
    # `CREATE TABLE IF NOT EXISTS` above is a no-op when the table already exists,
    # so a changed EMBED_DIM would otherwise go unnoticed until the first insert
    # fails with an opaque pgvector dimension-mismatch error.
    global _dim_mismatch
    actual_dim = await _column_dim(conn, "embedding")
    if actual_dim is None:
        raise RuntimeError("could not determine dimension of chunks.embedding column")
    if actual_dim != config.EMBED_DIM:
        _dim_mismatch = (actual_dim, config.EMBED_DIM)
        logger.error(
            "chunks.embedding is vector(%d) but EMBED_DIM is %d. search/ingest_text will "
            "refuse to run until every collection is migrated — call the reindex_collection "
            "MCP tool for each collection to re-embed it and migrate the shared column.",
            actual_dim,
            config.EMBED_DIM,
        )
    else:
        _dim_mismatch = None


def _raise_if_dim_mismatch() -> None:
    if _dim_mismatch is not None:
        actual, expected = _dim_mismatch
        raise RuntimeError(
            f"chunks.embedding is vector({actual}) but EMBED_DIM is {expected}. "
            "Call the reindex_collection MCP tool for every collection to migrate to the "
            "new dimension before searching or ingesting."
        )


async def insert_chunks(collection: str, rows: list[tuple[str, dict, list[float]]]) -> int:
    pool = await get_pool()
    _raise_if_dim_mismatch()
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
    _raise_if_dim_mismatch()
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


async def get_embedding_dim() -> int:
    pool = await get_pool()
    async with pool.connection() as conn:
        dim = await _column_dim(conn, "embedding")
    if dim is None:
        raise RuntimeError("could not determine dimension of chunks.embedding column")
    return dim


async def fetch_collection_rows(collection: str) -> list[dict]:
    """Read a collection's stored content/metadata directly, bypassing chunking.

    Used by reindex_collection so re-embedding doesn't re-run chunking.chunk_text
    and shift chunk boundaries.
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT id, content, metadata FROM chunks WHERE collection = %s ORDER BY id",
            (collection,),
        )
        rows = await cur.fetchall()
    return [{"id": r[0], "content": r[1], "metadata": r[2]} for r in rows]


async def update_embeddings(collection: str, ids: list[int], vectors: list[list[float]]) -> None:
    """Overwrite embeddings in place for rows already at the live column's dimension.

    All updates happen on one pooled connection, which commits as a single
    transaction on successful exit (or rolls back together on error) — the
    collection's embeddings never end up half-updated.
    """
    pool = await get_pool()
    async with pool.connection() as conn, conn.cursor() as cur:
        for row_id, vec in zip(ids, vectors):
            await cur.execute(
                "UPDATE chunks SET embedding = %s WHERE id = %s AND collection = %s",
                (vec, row_id, collection),
            )


async def _ensure_pending_column(conn, new_dim: int) -> None:
    existing_dim = await _column_dim(conn, _PENDING_COLUMN)
    if existing_dim == new_dim:
        return
    if existing_dim is not None:
        # Stale staging column from an abandoned reindex at a different width.
        await conn.execute(f"ALTER TABLE chunks DROP COLUMN {_PENDING_COLUMN}")
    await conn.execute(f"ALTER TABLE chunks ADD COLUMN {_PENDING_COLUMN} vector({new_dim})")


async def reindex_with_new_dimension(
    collection: str, ids: list[int], vectors: list[list[float]], new_dim: int
) -> bool:
    """Stage `ids`' re-embedded vectors at the new width and swap in the column
    once every collection has been staged.

    The live `embedding` column (and its ivfflat index) is left untouched until
    the very last collection is staged, so search()/insert_chunks() keep working
    for every not-yet-reindexed collection throughout. Returns True if this call
    performed the final swap (i.e. this was the last collection pending).
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        await _ensure_pending_column(conn, new_dim)
        await conn.commit()

        async with conn.cursor() as cur:
            for row_id, vec in zip(ids, vectors):
                await cur.execute(
                    f"UPDATE chunks SET {_PENDING_COLUMN} = %s WHERE id = %s AND collection = %s",
                    (vec, row_id, collection),
                )
        await conn.commit()

        cur = await conn.execute(f"SELECT count(*) FROM chunks WHERE {_PENDING_COLUMN} IS NULL")
        (remaining,) = await cur.fetchone()
        if remaining > 0:
            return False

        await conn.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
        await conn.execute("ALTER TABLE chunks DROP COLUMN embedding")
        await conn.execute(f"ALTER TABLE chunks RENAME COLUMN {_PENDING_COLUMN} TO embedding")
        await conn.execute("ALTER TABLE chunks ALTER COLUMN embedding SET NOT NULL")
        await conn.execute(
            "CREATE INDEX chunks_embedding_idx ON chunks "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )
        await conn.commit()

    global _dim_mismatch
    _dim_mismatch = None
    return True
