"""Offline tests — no API key needed (fake embeddings + fake LLM).

    pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chatbot import NOT_FOUND, RAGChatbot, to_messages  # noqa: E402
from rag.index import build_index, load_index, read_manifest, split_documents  # noqa: E402
from rag.loaders import load_directory, load_notion_export  # noqa: E402

DOCS = ROOT / "data" / "docs"
NOTION = ROOT / "data" / "notion"


@pytest.fixture(scope="module")
def docs():
    return load_directory(DOCS) + load_notion_export(NOTION)


def test_loaders_attach_metadata(docs):
    names = {d.metadata["source"] for d in docs}
    assert {"HR_Policy.pdf", "Company_Handbook.md", "IT_Security_Policy.md"} <= names
    pdf_pages = [d for d in docs if d.metadata["source"] == "HR_Policy.pdf"]
    assert [d.metadata["page"] for d in pdf_pages] == [1, 2]  # 1-based pages
    assert "24 paid leaves" in pdf_pages[0].page_content


def test_notion_id_suffix_is_stripped(docs):
    notion = [d for d in docs if d.metadata["source_type"] == "notion"]
    assert notion and notion[0].metadata["source"] == "Engineering Onboarding"


def test_chunking_keeps_metadata_and_ids(docs):
    chunks = split_documents(docs, chunk_size=400, chunk_overlap=50)
    assert len(chunks) > len(docs)
    assert all(len(c.page_content) <= 400 for c in chunks)
    assert all("source" in c.metadata and "chunk_id" in c.metadata for c in chunks)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_build_save_load_and_answer(docs, tmp_path):
    emb = DeterministicFakeEmbedding(size=64)
    chunks = split_documents(docs, 400, 50)
    build_index(chunks, emb, tmp_path)
    assert read_manifest(tmp_path)["chunks"] == len(chunks)

    store = load_index(tmp_path, emb)
    llm = FakeListChatModel(responses=["You get 24 paid leaves per year [1]."])
    bot = RAGChatbot(store, llm, company="Acme", top_k=3, min_relevance=float("-inf"))

    ans = bot.ask("What is our leave policy?")
    assert "24" in ans.text
    assert len(ans.sources) == 3
    assert [s.number for s in ans.sources] == [1, 2, 3]


def test_no_relevant_context_skips_llm(docs, tmp_path):
    emb = DeterministicFakeEmbedding(size=64)
    store = build_index(split_documents(docs, 400, 50), emb, tmp_path)
    llm = FakeListChatModel(responses=["SHOULD NOT BE CALLED"])
    bot = RAGChatbot(store, llm, min_relevance=999)  # nothing can pass this threshold
    ans = bot.ask("anything")
    assert ans.text == NOT_FOUND and ans.sources == []


def test_follow_up_question_is_condensed(docs, tmp_path):
    emb = DeterministicFakeEmbedding(size=64)
    store = build_index(split_documents(docs, 400, 50), emb, tmp_path)
    llm = FakeListChatModel(responses=["How many sick leaves do employees get?", "6 sick leaves [1]."])
    bot = RAGChatbot(store, llm, min_relevance=float("-inf"))
    history = to_messages([{"role": "user", "content": "What is our leave policy?"},
                           {"role": "assistant", "content": "24 paid leaves [1]."}])
    ans = bot.ask("what about sick leave?", history)
    assert ans.standalone_question == "How many sick leaves do employees get?"
    assert ans.text.startswith("6 sick leaves")


def test_embedding_model_mismatch_is_detected(docs, tmp_path):
    emb = DeterministicFakeEmbedding(size=64)
    build_index(split_documents(docs, 400, 50), emb, tmp_path, embedding_signature="fastembed:model-a")
    load_index(tmp_path, emb, "fastembed:model-a")  # same model: fine
    with pytest.raises(RuntimeError, match="rebuild"):
        load_index(tmp_path, emb, "openai:model-b")


def test_provider_key_check(monkeypatch):
    from types import SimpleNamespace

    from rag.providers import missing_api_key

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert missing_api_key(SimpleNamespace(llm_provider="groq")) == "GROQ_API_KEY"
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert missing_api_key(SimpleNamespace(llm_provider="groq")) is None


def test_missing_index_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="ingest.py"):
        load_index(tmp_path / "nope", DeterministicFakeEmbedding(size=8))
