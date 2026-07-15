"""Async embeddings via an OpenAI-compatible /embeddings endpoint."""
from __future__ import annotations

import httpx

from . import config


async def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    headers = {"Content-Type": "application/json"}
    if config.EMBEDDINGS_API_KEY:
        headers["Authorization"] = f"Bearer {config.EMBEDDINGS_API_KEY}"
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{config.EMBEDDINGS_API_BASE.rstrip('/')}/embeddings",
            headers=headers,
            json={"model": config.EMBEDDINGS_MODEL, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
    # preserve input order regardless of provider response ordering
    data.sort(key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]
