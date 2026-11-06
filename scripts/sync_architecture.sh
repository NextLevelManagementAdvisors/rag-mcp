#!/usr/bin/env bash
# Host-side wrapper: refresh upstream sources, then re-ingest into rag-mcp.
# Run on the VPS: scripts/sync_architecture.sh  (intended for a weekly cron)
set -euo pipefail
SRC=${ARCH_SOURCES_DIR:-/opt/rag-mcp-sources}
C=${RAG_CONTAINER:-rag-mcp-rag-mcp-1}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$SRC" && cd "$SRC"
[ -d cosmicpython ] || git clone -q --depth 1 https://github.com/cosmicpython/book cosmicpython
[ -d system-design-primer ] || git clone -q --depth 1 --filter=blob:limit=200k https://github.com/donnemartin/system-design-primer
[ -d cloud-design-patterns ] || git clone -q --depth 1 https://github.com/mspnp/cloud-design-patterns
if [ ! -d architecture-center ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/MicrosoftDocs/architecture-center
  git -C architecture-center sparse-checkout set docs/patterns
fi
for r in cosmicpython system-design-primer cloud-design-patterns architecture-center; do
  git -C "$r" pull -q --ff-only || echo "WARN: pull failed for $r" >&2
  git -C "$r" rev-parse HEAD > "$r/.commit"
done
cp "$HERE/ingest_architecture.py" "$SRC/"
# Ollama must be up (see issue #11); fail fast if not.
docker exec "$C" python -c "import httpx,os;httpx.post(os.environ['EMBEDDINGS_API_BASE'].rstrip('/')+'/embeddings',json={'model':os.environ['EMBEDDINGS_MODEL'],'input':'ping'},timeout=120).raise_for_status()"
docker exec "$C" mkdir -p /sources
tar --exclude=.git --exclude='*.png' --exclude='*.jpg' --exclude='*.svg' --exclude='*.gif' -cf - . | docker exec -i "$C" tar -xf - -C /sources
docker exec -w /sources "$C" python ingest_architecture.py /sources
