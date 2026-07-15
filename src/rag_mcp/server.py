"""rag-mcp — MCP tools for ingesting documents and semantically searching them.

RAG is split the MCP-native way: this server owns *retrieval* (chunk, embed,
store, similarity search); the calling LLM agent owns *generation*. Point any
MCP client (Claude, ChatGPT, Cursor, ...) at it and the model can ground its
answers in your documents.
"""
from __future__ import annotations

from fastmcp import FastMCP

from . import chunking, config, db, embeddings

mcp = FastMCP("rag-mcp")


@mcp.tool
async def ingest_text(collection: str, text: str, metadata: dict | None = None) -> dict:
    """Chunk, embed, and store text in a collection for later semantic search.

    collection: logical namespace (e.g. a project or document set).
    metadata:   optional JSON attached to every chunk (source, title, url, ...).
    """
    chunks = chunking.chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        return {"collection": collection, "chunks_ingested": 0}
    vectors = await embeddings.embed(chunks)
    rows = [(c, metadata or {}, v) for c, v in zip(chunks, vectors)]
    n = await db.insert_chunks(collection, rows)
    return {"collection": collection, "chunks_ingested": n}


@mcp.tool
async def search(collection: str, query: str, k: int = 5) -> list[dict]:
    """Semantic search a collection. Returns the top-k chunks with a 0-1 cosine score."""
    query_vec = (await embeddings.embed([query]))[0]
    return await db.search(collection, query_vec, max(1, min(k, 50)))


@mcp.tool
async def list_collections() -> list[dict]:
    """List all collections and their chunk counts."""
    return await db.list_collections()


@mcp.tool
async def delete_collection(collection: str) -> dict:
    """Delete a collection and all of its chunks. Irreversible."""
    deleted = await db.delete_collection(collection)
    return {"collection": collection, "deleted_chunks": deleted}


def main() -> None:
    if config.MCP_TRANSPORT == "http":
        mcp.run(transport="http", host=config.MCP_HOST, port=config.MCP_PORT)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
