"""Re-runnable ingest of openly licensed architecture sources into rag-mcp's
'architecture' collection. Run inside the rag-mcp container (uses its config).
Every chunk carries source/license/url/section metadata for citation.

Usage (via scripts/sync_architecture.sh): python ingest_architecture.py /sources
Each source dir may contain a `.commit` file (written on the host, since the
container has no git) so chunk URLs pin to the exact upstream revision.

Note: the collection is cleared and rebuilt, so searches against
'architecture' return partial results while a re-sync is running.
"""
import asyncio, re, sys
from pathlib import Path
from rag_mcp import chunking, config, db, embeddings

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/sources")
COLL = "architecture"


def sha(d):
    p = d / ".commit"
    return p.read_text().strip() if p.exists() else "main"


SOURCES = [
  dict(dir="cosmicpython", name="Architecture Patterns with Python (Cosmic Python)",
       license="CC-BY-ND (per repo license.txt)", repo="https://github.com/cosmicpython/book",
       files=lambda d: sorted([*d.glob("chapter_*.asciidoc"), *d.glob("appendix_*.asciidoc"),
                               d/"introduction.asciidoc", d/"part1.asciidoc", d/"part2.asciidoc",
                               d/"preface.asciidoc", *d.glob("epilogue_*.asciidoc")]),
       head=r"^(={2,4})\s+(.+)$"),
  dict(dir="system-design-primer", name="The System Design Primer",
       license="CC-BY-4.0", repo="https://github.com/donnemartin/system-design-primer",
       files=lambda d: [d/"README.md", *sorted(d.glob("solutions/system_design/*/README.md"))],
       head=r"^(#{2,4})\s+(.+)$"),
  dict(dir="architecture-center", name="Azure Architecture Center - Cloud Design Patterns",
       license="CC-BY-4.0", repo="https://github.com/MicrosoftDocs/architecture-center",
       files=lambda d: sorted((d/"docs/patterns").glob("*.md")),
       head=r"^(#{1,3})\s+(.+)$"),
  dict(dir="cloud-design-patterns", name="mspnp Cloud Design Patterns samples",
       license="MIT", repo="https://github.com/mspnp/cloud-design-patterns",
       files=lambda d: [d/"README.md", *sorted(d.glob("*/README.md"))],
       head=r"^(#{1,3})\s+(.+)$"),
]


def sections(text, pat):
    rx = re.compile(pat, re.M); out = []; last = 0; title = "(intro)"
    for m in rx.finditer(text):
        if text[last:m.start()].strip():
            out.append((title, text[last:m.start()]))
        title = m.group(2).strip(); last = m.start()
    if text[last:].strip():
        out.append((title, text[last:]))
    return out


def clean(t):
    t = re.sub(r"^---\n.*?\n---\n", "", t, flags=re.S)   # yaml front matter
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)            # md images
    t = re.sub(r"^image::.*$", "", t, flags=re.M)        # asciidoc images
    return re.sub(r"\n{3,}", "\n\n", t)


async def main():
    await db.get_pool()
    deleted = await db.delete_collection(COLL)
    print(f"cleared {deleted} old chunks", flush=True)
    total = 0
    for s in SOURCES:
        d = ROOT / s["dir"]; commit = sha(d); n_src = 0
        for f in s["files"](d):
            if not f.exists():
                continue
            rel = f.relative_to(d).as_posix()
            for title, body in sections(clean(f.read_text(errors="ignore")), s["head"]):
                chunks = [c for c in chunking.chunk_text(body, config.CHUNK_SIZE, config.CHUNK_OVERLAP) if len(c) > 80]
                if not chunks:
                    continue
                vecs = []
                for i in range(0, len(chunks), 32):
                    vecs += await embeddings.embed(chunks[i:i + 32])
                meta = dict(source=s["name"], license=s["license"], section=title, file=rel,
                            url=f'{s["repo"]}/blob/{commit}/{rel}', commit=commit)
                n_src += await db.insert_chunks(COLL, [(c, meta, v) for c, v in zip(chunks, vecs)])
        print(f'{s["name"]}: {n_src} chunks', flush=True); total += n_src
    print(f"TOTAL {total}", flush=True)


asyncio.run(main())
