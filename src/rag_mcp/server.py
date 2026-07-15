"""rag-mcp — MCP tools for ingesting documents and semantically searching them.

RAG is split the MCP-native way: this server owns *retrieval* (chunk, embed,
store, similarity search); the calling LLM agent owns *generation*. Point any
MCP client (Claude, ChatGPT, Cursor, ...) at it and the model can ground its
answers in your documents.
"""
from __future__ import annotations

import logging

from fastmcp import FastMCP

from . import chunking, config, db, embeddings

logger = logging.getLogger("rag-mcp")


def _enforce_email_allowlist(provider, allowed: set[str]) -> None:
    """Wrap the provider's Google token verifier to reject non-allowlisted emails.

    Fail-closed: if we can't find the verifier to wrap, raise rather than run
    an open gate.
    """
    from fastmcp.server.auth.providers.google import GoogleTokenVerifier

    verifier = next(
        (v for v in vars(provider).values() if isinstance(v, GoogleTokenVerifier)),
        None,
    )
    if verifier is None:
        raise RuntimeError("Could not locate Google token verifier to enforce allowlist")

    original_verify = verifier.verify_token

    async def verify(token: str):
        access = await original_verify(token)
        if access is not None:
            claims = getattr(access, "claims", None) or {}
            email = str(claims.get("email") or "").lower()
            if email not in allowed:
                logger.warning("Rejected sign-in for non-allowlisted email")
                return None
        return access

    verifier.verify_token = verify  # type: ignore[method-assign]


def _build_auth():
    if not config.GOOGLE_WEB_CLIENT_ID:
        return None  # stdio / ungated: rely on the network boundary
    from fastmcp.server.auth.providers.google import GoogleProvider

    provider = GoogleProvider(
        client_id=config.GOOGLE_WEB_CLIENT_ID,
        client_secret=config.GOOGLE_WEB_CLIENT_SECRET,
        base_url=config.MCP_BASE_URL,
        redirect_path="/auth/callback",
        required_scopes=["openid", "email"],
    )
    allowed = set(config.GOOGLE_ALLOWED_EMAILS)
    if not allowed:
        raise RuntimeError(
            "GOOGLE_WEB_CLIENT_ID is set but GOOGLE_ALLOWED_EMAILS is empty — refusing "
            "to run an open gate. Set GOOGLE_ALLOWED_EMAILS."
        )
    _enforce_email_allowlist(provider, allowed)
    return provider


mcp = FastMCP("rag-mcp", auth=_build_auth())


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
