"""Build (or verify) the persistent index from data/pdfs and print an inventory.

    python scripts/ingest_documents.py            # build if stale, else reuse
    python scripts/ingest_documents.py --force    # always rebuild
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import Settings  # noqa: E402
from backend.embeddings import load_embedder  # noqa: E402
from backend.ingestion import build_index, build_or_load_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if the index is up to date")
    args = ap.parse_args()
    settings = Settings.from_env()
    embedder = load_embedder(settings.embedding_backend, settings.embedding_model)
    store, manifest = (build_index if args.force else build_or_load_index)(settings, embedder)
    print(f"Index: {settings.index_dir}  embeddings: {manifest['embedding_model'] or 'none (BM25 only)'}")
    for d in manifest["documents"]:
        print(f"  {d['name']:34} {d['pages']} pages  sha256={d['sha256'][:12]}  ({d['title']})")
    print(f"{manifest['chunk_count']} chunks:")
    for c in store.chunks:
        print(f"  {c.chunk_id:40} p{c.page_number}  {c.section_title}")
    if manifest.get("empty_pages"):
        print("WARNING: pages without extractable text:", manifest["empty_pages"])


if __name__ == "__main__":
    main()
