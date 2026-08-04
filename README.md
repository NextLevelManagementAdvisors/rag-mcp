# rag-mcp

A **RAG-over-docs MCP server**: ingest text into collections, then let any LLM
agent semantically search it. Backed by **PostgreSQL + pgvector**, embeddings via
any **OpenAI-compatible** endpoint (OpenAI, Azure, or a local Ollama / llama.cpp
server).

RAG is split the MCP-native way — this server owns **retrieval** (chunk → embed →
store → cosine search); the calling model owns **generation**. Point Claude,
ChatGPT, Cursor, or any MCP client at it and answers get grounded in your docs.

## Tools

| Tool | Purpose |
| --- | --- |
| `ingest_text` | Chunk, embed, and store text in a collection (with optional metadata) |
| `search` | Top-k semantic search over a collection; returns chunks + 0–1 cosine score |
| `list_collections` | List collections and their chunk counts |
| `delete_collection` | Delete a collection and all its chunks |

## Quickstart

```bash
# 1. Start Postgres + pgvector and the server
cp .env.example .env         # set EMBEDDINGS_API_KEY
docker compose up -d --build

# — or run locally over stdio (e.g. for Claude Desktop) —
uv pip install --system .
docker compose up -d db      # just the database
EMBEDDINGS_API_KEY=sk-... rag-mcp
```

Add to an MCP client (stdio):

```json
{ "mcpServers": { "rag": { "command": "rag-mcp",
  "env": { "EMBEDDINGS_API_KEY": "sk-...",
           "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/rag" } } } }
```

## Deployment overrides

For host-specific tweaks (e.g. dropping public port bindings on a server
where only the internal Docker network should reach Postgres), copy
[`docker-compose.override.yml.example`](docker-compose.override.yml.example)
to `docker-compose.override.yml` — it's gitignored and loaded automatically by
`docker compose`. Don't hand-edit the tracked `docker-compose.yml` or hide
local changes with `git update-index --skip-worktree`; either desyncs
`git status`/`git diff` from what's actually deployed and can silently block
fast-forward merges of upstream fixes.

## Configuration

All via environment (see [.env.example](.env.example)): `DATABASE_URL`,
`EMBEDDINGS_API_BASE` / `EMBEDDINGS_API_KEY` / `EMBEDDINGS_MODEL`, `EMBED_DIM`
(**must match your model's dimension** — 1536 for `text-embedding-3-small`),
`CHUNK_SIZE`, `CHUNK_OVERLAP`, and `MCP_TRANSPORT` (`stdio` | `http`).

## How it works

`ingest_text` splits text on natural boundaries with overlap, embeds each chunk,
and stores it in a `chunks` table with a `vector(EMBED_DIM)` column and an
`ivfflat` cosine index. `search` embeds the query and returns the nearest chunks
by cosine distance (`<=>`). Everything is namespaced by `collection`.

## License

Copyright © 2026 Next Level Management Advisors, LLC.

Licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0) — see
[LICENSE](LICENSE). If you run a modified version over a network, the AGPL
requires you to make your modified source available to its users.

**Commercial licensing:** to use this in a closed-source or commercial product,
or to host a modified version without publishing your source, a commercial
license is available — contact **forrest@nlma.io**.
