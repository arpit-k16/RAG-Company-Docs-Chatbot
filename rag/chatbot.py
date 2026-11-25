"""The question-answering chain: retrieve -> ground -> answer with citations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore

SYSTEM_PROMPT = """You are the internal knowledge assistant for {company}.
Answer employee questions using ONLY the numbered context passages provided.

Rules:
- Cite the passages you used inline with plain square brackets, like [1] or [2][3]. Every factual sentence needs a citation.
- If the context does not contain the answer, say you could not find it in the company documents
  and suggest which team (HR, IT, Finance, ...) to ask. Never invent policies, numbers or dates.
- If passages conflict, prefer the most recent version and mention the conflict.
- Be concise: a direct answer first, then short supporting detail or bullet points if useful."""

CONDENSE_PROMPT = """Given the conversation so far and a follow-up question, rewrite the follow-up
as a standalone question that can be understood without the conversation. Return only the question.

Conversation:
{history}

Follow-up question: {question}
Standalone question:"""

def normalize_citations(text: str) -> str:
    """Some models write 【1】 instead of [1]; make citations consistent."""
    return text.replace("\u3010", "[").replace("\u3011", "]")


NOT_FOUND = ("I couldn't find anything about that in the company documents. "
             "Try rephrasing, or check with the relevant team (HR, IT, Finance).")


@dataclass
class Source:
    number: int
    source: str
    source_type: str
    snippet: str
    score: float
    page: int | None = None
    url: str | None = None

    @property
    def label(self) -> str:
        return f"{self.source} (p. {self.page})" if self.page else self.source


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    standalone_question: str | None = None


class RAGChatbot:
    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel, company: str = "the company",
                 top_k: int = 4, min_relevance: float = 0.25, history_turns: int = 6):
        self.vectorstore = vectorstore
        self.llm = llm
        self.company = company
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.history_turns = history_turns

    # ---------- construction ----------
    @classmethod
    def from_settings(cls, settings) -> "RAGChatbot":
        from .index import load_index
        from .providers import embedding_signature, get_embeddings, get_llm

        store = load_index(settings.index_dir, get_embeddings(settings), embedding_signature(settings))
        llm = get_llm(settings)
        return cls(store, llm, company=settings.company_name, top_k=settings.top_k,
                   min_relevance=settings.min_relevance)

    # ---------- retrieval ----------
    def retrieve(self, question: str) -> list[tuple[Document, float]]:
        results = self.vectorstore.similarity_search_with_relevance_scores(question, k=self.top_k)
        return [(doc, score) for doc, score in results if score >= self.min_relevance]

    @staticmethod
    def build_sources(results: Sequence[tuple[Document, float]]) -> list[Source]:
        sources = []
        for i, (doc, score) in enumerate(results, start=1):
            m = doc.metadata
            snippet = " ".join(doc.page_content.split())
            sources.append(Source(
                number=i,
                source=m.get("source", "unknown"),
                source_type=m.get("source_type", "text"),
                snippet=snippet[:300] + ("…" if len(snippet) > 300 else ""),
                score=round(float(score), 3),
                page=m.get("page"),
                url=m.get("url"),
            ))
        return sources

    @staticmethod
    def format_context(results: Sequence[tuple[Document, float]]) -> str:
        blocks = []
        for i, (doc, _) in enumerate(results, start=1):
            m = doc.metadata
            where = m.get("source", "unknown") + (f", page {m['page']}" if m.get("page") else "")
            blocks.append(f"[{i}] Source: {where}\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(blocks)

    # ---------- conversation ----------
    def _trim_history(self, history: Sequence[BaseMessage] | None) -> list[BaseMessage]:
        return list(history or [])[-self.history_turns * 2:]

    def condense_question(self, question: str, history: Sequence[BaseMessage] | None) -> str:
        """Turn 'what about sick leave?' into 'How many sick leaves do employees get?'."""
        history = self._trim_history(history)
        if not history:
            return question
        transcript = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in history
        )
        reply = self.llm.invoke(CONDENSE_PROMPT.format(history=transcript, question=question))
        return str(reply.content).strip() or question

    def _messages(self, question: str, context: str,
                  history: Sequence[BaseMessage] | None) -> list[BaseMessage]:
        return [
            SystemMessage(SYSTEM_PROMPT.format(company=self.company)),
            *self._trim_history(history),
            HumanMessage(f"Context passages:\n\n{context}\n\nQuestion: {question}"),
        ]

    # ---------- public API ----------
    def ask(self, question: str, history: Sequence[BaseMessage] | None = None) -> Answer:
        standalone = self.condense_question(question, history)
        results = self.retrieve(standalone)
        if not results:
            return Answer(NOT_FOUND, [], standalone)
        reply = self.llm.invoke(self._messages(question, self.format_context(results), history))
        return Answer(normalize_citations(str(reply.content)), self.build_sources(results), standalone)

    def stream(self, question: str, history: Sequence[BaseMessage] | None = None
               ) -> tuple[list[Source], Iterator[str]]:
        """Return sources immediately plus a token iterator for the answer."""
        standalone = self.condense_question(question, history)
        results = self.retrieve(standalone)
        if not results:
            return [], iter([NOT_FOUND])

        messages = self._messages(question, self.format_context(results), history)

        def tokens() -> Iterator[str]:
            for chunk in self.llm.stream(messages):
                if chunk.content:
                    yield normalize_citations(str(chunk.content))

        return self.build_sources(results), tokens()


def to_messages(turns: Sequence[dict]) -> list[BaseMessage]:
    """Convert [{'role': 'user'|'assistant', 'content': str}, ...] to LangChain messages."""
    out: list[BaseMessage] = []
    for t in turns:
        out.append(HumanMessage(t["content"]) if t["role"] == "user" else AIMessage(t["content"]))
    return out
