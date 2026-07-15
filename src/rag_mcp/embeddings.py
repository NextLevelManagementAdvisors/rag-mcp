"""Embeddings — either a local in-process model (fastembed/ONNX) or any
OpenAI-compatible HTTP endpoint, selected by EMBEDDINGS_PROVIDER."""
from __future__ import annotations

import asyncio

import httpx

from . import config

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from fastembed import TextEmbedding  # lazy: only needed for provider=local

        _local_model = TextEmbedding(model_name=config.EMBEDDINGS_MODEL)
    return _local_model


async def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if config.EMBEDDINGS_PROVIDER == "local":
        def run() -> list[list[float]]:
            return [vec.tolist() for vec in _get_local_model().embed(texts)]

        return await asyncio.to_thread(run)

    # OpenAI-compatible HTTP endpoint
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
    data.sort(key=lambda d: d.get("index", 0))  # preserve input order
    return [d["embedding"] for d in data]
