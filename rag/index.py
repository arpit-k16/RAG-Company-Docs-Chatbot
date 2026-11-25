"""Chunking and the FAISS vector index."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"
# Normalising vectors makes FAISS distances map cleanly onto a 0..1 relevance score,
# whichever embedding model is used.
FAISS_KWARGS = {"normalize_L2": True}


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Prefer to break on markdown headings, then paragraphs, then sentences
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = hashlib.sha1(
            f"{c.metadata.get('path', c.metadata.get('source'))}:{c.metadata.get('page', 0)}:"
            f"{c.metadata.get('start_index', i)}".encode()
        ).hexdigest()[:12]
    return chunks


def build_index(chunks: list[Document], embeddings: Embeddings, index_dir: Path,
                embedding_signature: str = "unknown", batch_size: int = 200) -> FAISS:
    """Embed chunks in batches and save the index + a manifest to disk."""
    if not chunks:
        raise ValueError("No document chunks to index. Add files to data/docs/ first.")

    store: FAISS | None = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        if store is None:
            store = FAISS.from_documents(batch, embeddings, **FAISS_KWARGS)
        else:
            store.add_documents(batch)
        log.info("Embedded %d/%d chunks", min(start + batch_size, len(chunks)), len(chunks))

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))

    sources = sorted({c.metadata.get("source", "?") for c in chunks})
    (index_dir / MANIFEST).write_text(json.dumps({
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding": embedding_signature,
        "chunks": len(chunks),
        "sources": sources,
    }, indent=2), encoding="utf-8")
    return store


def index_exists(index_dir: Path) -> bool:
    return (Path(index_dir) / "index.faiss").exists()


def load_index(index_dir: Path, embeddings: Embeddings, embedding_signature: str | None = None) -> FAISS:
    index_dir = Path(index_dir)
    if not index_exists(index_dir):
        raise FileNotFoundError(f"No index found at {index_dir}. Run `python ingest.py` first.")

    manifest = read_manifest(index_dir) or {}
    built_with = manifest.get("embedding")
    if embedding_signature and built_with and built_with != embedding_signature:
        raise RuntimeError(
            f"The index was built with '{built_with}' but you're now using '{embedding_signature}'. "
            "Run `python ingest.py` to rebuild it."
        )
    # The pickle in this folder was written by our own ingest step, so it is trusted.
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True,
                            **FAISS_KWARGS)


def read_manifest(index_dir: Path) -> dict | None:
    path = Path(index_dir) / MANIFEST
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
