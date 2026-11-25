"""Build (or rebuild) the FAISS index from company documents.

    python ingest.py                     # data/docs + data/notion (+ Confluence if configured)
    python ingest.py --docs-dir D:/hr    # a different folder
    python ingest.py --no-confluence
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

from config import settings
from rag.index import build_index, split_documents
from rag.providers import embedding_signature, get_embeddings
from rag.loaders import load_confluence, load_directory, load_notion_export


def collect_documents(docs_dir: Path, notion_dir: Path, use_confluence: bool):
    docs = load_directory(docs_dir)
    docs += load_notion_export(notion_dir)
    if use_confluence and settings.confluence_enabled:
        logging.info("Fetching Confluence space %s ...", settings.confluence_space_key)
        docs += load_confluence(settings.confluence_url, settings.confluence_username,
                                settings.confluence_api_key, settings.confluence_space_key)
    return docs


def run(docs_dir: Path = settings.docs_dir, notion_dir: Path = settings.notion_dir,
        use_confluence: bool = True, embeddings=None) -> dict:
    docs = collect_documents(docs_dir, notion_dir, use_confluence)
    if not docs:
        raise SystemExit(f"No documents found in {docs_dir} or {notion_dir}.")
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    build_index(chunks, embeddings or get_embeddings(settings), settings.index_dir,
                embedding_signature(settings))
    by_type = Counter(d.metadata.get("source_type", "?") for d in docs)
    return {"documents": len({d.metadata['source'] for d in docs}), "chunks": len(chunks),
            "by_type": dict(by_type)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--docs-dir", type=Path, default=settings.docs_dir)
    p.add_argument("--notion-dir", type=Path, default=settings.notion_dir)
    p.add_argument("--no-confluence", action="store_true")
    args = p.parse_args()

    stats = run(args.docs_dir, args.notion_dir, not args.no_confluence)
    print(f"\n✅ Indexed {stats['documents']} documents → {stats['chunks']} chunks "
          f"({', '.join(f'{k}: {v}' for k, v in stats['by_type'].items())})")
    print(f"   Saved to {settings.index_dir}")


if __name__ == "__main__":
    sys.exit(main())
