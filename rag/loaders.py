"""Load company documents from local files, a Notion export and (optionally) Confluence.

Every Document gets consistent metadata so the UI can show where an answer came from:
    source       -> display name, e.g. "HR_Policy.pdf"
    source_type  -> "pdf" | "markdown" | "text" | "docx" | "notion" | "confluence"
    page         -> 1-based page number for PDFs (absent otherwise)
    url          -> link back to the original (Confluence only)
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".docx"}


def _load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        docs = PyPDFLoader(str(path)).load()
        for d in docs:
            # PyPDFLoader pages are 0-based; humans read 1-based
            d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
            d.metadata["source_type"] = "pdf"
        return docs

    if ext == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        docs = Docx2txtLoader(str(path)).load()
        for d in docs:
            d.metadata["source_type"] = "docx"
        return docs

    # Markdown / plain text
    text = path.read_text(encoding="utf-8", errors="ignore")
    source_type = "markdown" if ext in {".md", ".markdown"} else "text"
    return [Document(page_content=text, metadata={"source_type": source_type})]


def load_directory(directory: Path, source_type_override: str | None = None) -> list[Document]:
    """Load every supported file under `directory` (recursively)."""
    directory = Path(directory)
    if not directory.exists():
        log.info("Skipping %s (does not exist)", directory)
        return []

    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name.startswith(("~$", ".")):  # Office lock files / hidden files
            continue
        try:
            docs = _load_file(path)
        except Exception as exc:  # one bad file should not kill ingestion
            log.warning("Could not load %s: %s", path, exc)
            continue
        for d in docs:
            d.metadata["source"] = path.name
            d.metadata["path"] = str(path.relative_to(directory))
            if source_type_override:
                d.metadata["source_type"] = source_type_override
        documents.extend(d for d in docs if d.page_content.strip())
        log.info("Loaded %s (%d part%s)", path.name, len(docs), "" if len(docs) == 1 else "s")
    return documents


def load_notion_export(directory: Path) -> list[Document]:
    """Notion: Settings -> Export -> 'Markdown & CSV', unzip into data/notion/."""
    docs = load_directory(directory, source_type_override="notion")
    for d in docs:
        # Notion appends a 32-char id to file names: "Onboarding 1a2b...md" -> "Onboarding"
        stem = Path(d.metadata["source"]).stem
        parts = stem.rsplit(" ", 1)
        if len(parts) == 2 and len(parts[1]) == 32:
            d.metadata["source"] = parts[0]
    return docs


def load_confluence(url: str, username: str, api_key: str, space_key: str, limit: int = 50) -> list[Document]:
    """Pull every page of a Confluence space (needs `atlassian-python-api`)."""
    from langchain_community.document_loaders import ConfluenceLoader

    loader = ConfluenceLoader(url=url, username=username, api_key=api_key,
                              space_key=space_key, limit=limit)
    docs = loader.load()
    for d in docs:
        original = str(d.metadata.get("source", ""))
        if original.startswith("http"):
            d.metadata["url"] = original  # ConfluenceLoader stores the page URL in "source"
        d.metadata["source"] = d.metadata.get("title", "Confluence page")
        d.metadata["source_type"] = "confluence"
    return docs
