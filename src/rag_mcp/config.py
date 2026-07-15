"""Environment-driven configuration. No secrets are hardcoded."""
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag")

# Provider: "openai" (any OpenAI-compatible /embeddings endpoint) or "local"
# (in-process ONNX via fastembed — no API key, no external service).
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai")

# OpenAI-compatible embeddings endpoint. Works with OpenAI, Azure OpenAI, or a
# local server (Ollama, llama.cpp, LM Studio) that speaks the /embeddings API.
EMBEDDINGS_API_BASE = os.getenv("EMBEDDINGS_API_BASE", "https://api.openai.com/v1")
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY", "")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))  # must match the model's dimension

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")  # "stdio" or "http"
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "3050"))

# --- OAuth 2.1 via Google sign-in (HTTP transport; optional) ---
# When GOOGLE_WEB_CLIENT_ID/SECRET are set, the HTTP server fronts an OAuth 2.1
# provider (DCR + PKCE) gated by "Sign in with Google". Only emails in
# GOOGLE_ALLOWED_EMAILS may authenticate (fail-closed: empty list blocks all).
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:3050")
GOOGLE_WEB_CLIENT_ID = os.getenv("GOOGLE_WEB_CLIENT_ID", "")
GOOGLE_WEB_CLIENT_SECRET = os.getenv("GOOGLE_WEB_CLIENT_SECRET", "")
GOOGLE_ALLOWED_EMAILS = [
    e.strip().lower() for e in os.getenv("GOOGLE_ALLOWED_EMAILS", "").split(",") if e.strip()
]
