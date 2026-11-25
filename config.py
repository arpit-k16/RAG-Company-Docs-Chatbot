"""Central configuration, read from environment variables / .env (or Streamlit secrets)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)  # the project .env wins over system-wide variables

# The embedding model is public, so never send a saved Hugging Face login with downloads.
# (An expired `hf auth login` token on the machine otherwise makes every download fail with 401.)
# Must be set before huggingface_hub is imported, which is why it lives here in config.
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# Sensible default models per provider (override with OPENAI_CHAT_MODEL / CHAT_MODEL etc.)
DEFAULT_CHAT_MODEL = {"groq": "openai/gpt-oss-120b", "openai": "gpt-4o-mini"}
DEFAULT_EMBEDDING_MODEL = {"fastembed": "BAAI/bge-small-en-v1.5", "openai": "text-embedding-3-small"}


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    company_name: str = field(default_factory=lambda: _env("COMPANY_NAME", "Acme Corp"))

    # Where documents live and where the FAISS index / downloaded models are stored
    docs_dir: Path = field(default_factory=lambda: Path(_env("DOCS_DIR", str(ROOT / "data" / "docs"))))
    notion_dir: Path = field(default_factory=lambda: Path(_env("NOTION_EXPORT_DIR", str(ROOT / "data" / "notion"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", str(ROOT / "storage" / "faiss_index"))))
    model_cache_dir: Path = field(default_factory=lambda: Path(_env("MODEL_CACHE_DIR", str(ROOT / "storage" / "models"))))

    # Providers: groq (free) / openai  ·  fastembed (free, local) / openai
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "groq").lower())
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "fastembed").lower())
    chat_model: str = field(default_factory=lambda: _env(
        "CHAT_MODEL", DEFAULT_CHAT_MODEL.get(_env("LLM_PROVIDER", "groq").lower(), "")))
    embedding_model: str = field(default_factory=lambda: _env(
        "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL.get(_env("EMBEDDING_PROVIDER", "fastembed").lower(), "")))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0")))

    # Chunking + retrieval
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "120")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    min_relevance: float = field(default_factory=lambda: float(_env("MIN_RELEVANCE", "0.3")))

    # Optional Confluence source
    confluence_url: str | None = field(default_factory=lambda: _env("CONFLUENCE_URL"))
    confluence_username: str | None = field(default_factory=lambda: _env("CONFLUENCE_USERNAME"))
    confluence_api_key: str | None = field(default_factory=lambda: _env("CONFLUENCE_API_KEY"))
    confluence_space_key: str | None = field(default_factory=lambda: _env("CONFLUENCE_SPACE_KEY"))

    @property
    def confluence_enabled(self) -> bool:
        return all([self.confluence_url, self.confluence_username,
                    self.confluence_api_key, self.confluence_space_key])


settings = Settings()
