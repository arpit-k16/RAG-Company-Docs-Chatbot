"""Pick the chat model and the embedding model from settings.

    LLM_PROVIDER=groq        (free tier)  -> ChatGroq,  needs GROQ_API_KEY
    LLM_PROVIDER=openai                   -> ChatOpenAI, needs OPENAI_API_KEY

    EMBEDDING_PROVIDER=fastembed (free, runs locally on CPU, no key, ~70 MB model)
    EMBEDDING_PROVIDER=openai             -> OpenAIEmbeddings, needs OPENAI_API_KEY
"""
from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

REQUIRED_KEYS = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY"}


def missing_api_key(settings) -> str | None:
    """Return the name of the env var the chosen LLM needs but doesn't have, else None."""
    key = REQUIRED_KEYS.get(settings.llm_provider)
    return key if key and not os.getenv(key) else None


def get_llm(settings, streaming: bool = True) -> BaseChatModel:
    provider = settings.llm_provider
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=settings.chat_model, temperature=settings.temperature,
                        streaming=streaming, max_retries=3)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.chat_model, temperature=settings.temperature,
                          streaming=streaming, max_retries=3)
    raise ValueError(f"Unknown LLM_PROVIDER {provider!r}. Use 'groq' or 'openai'.")


def get_embeddings(settings) -> Embeddings:
    provider = settings.embedding_provider
    if provider == "fastembed":
        from langchain_community.embeddings import FastEmbedEmbeddings

        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        return FastEmbedEmbeddings(model_name=settings.embedding_model,
                                   cache_dir=str(settings.model_cache_dir))
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER {provider!r}. Use 'fastembed' or 'openai'.")


def embedding_signature(settings) -> str:
    """Identifies the embedding model an index was built with (indexes aren't interchangeable)."""
    return f"{settings.embedding_provider}:{settings.embedding_model}"
