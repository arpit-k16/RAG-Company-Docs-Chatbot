"""Streamlit chat UI.   Run:  streamlit run app.py"""
from __future__ import annotations

import os
import re

import streamlit as st

# On Streamlit Community Cloud, keys live in "Secrets" — copy them into env vars
# *before* config is imported so local (.env) and cloud runs behave the same.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float)):
            os.environ.setdefault(_k, str(_v))
except Exception:  # no secrets.toml locally — that's fine, .env is used instead
    pass

from config import settings  # noqa: E402
from rag import RAGChatbot  # noqa: E402
from rag.chatbot import to_messages  # noqa: E402
from rag.index import index_exists, read_manifest  # noqa: E402
from rag.loaders import SUPPORTED_EXTENSIONS  # noqa: E402
from rag.providers import embedding_signature, missing_api_key  # noqa: E402

ICONS = {"pdf": "📕", "notion": "📓", "confluence": "🌐", "markdown": "📄", "text": "📄", "docx": "📘"}

st.set_page_config(page_title=f"{settings.company_name} AI Assistant", page_icon="🤖", layout="wide")
st.markdown("""
<style>
.src-chip {display:inline-block;padding:2px 10px;margin:2px 4px 2px 0;border-radius:8px;
           background:rgba(163,230,53,.12);border:1px solid rgba(163,230,53,.45);font-size:.82rem}
</style>""", unsafe_allow_html=True)


def build_index_now() -> dict:
    import ingest
    return ingest.run(use_confluence=settings.confluence_enabled)


@st.cache_resource(show_spinner="Loading knowledge base…")
def get_bot() -> RAGChatbot:
    return RAGChatbot.from_settings(settings)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    chips = "".join(
        f'<span class="src-chip">{ICONS.get(s["source_type"], "📄")} [{s["number"]}] {s["label"]}</span>'
        for s in sources
    )
    st.markdown(chips, unsafe_allow_html=True)
    with st.expander("View source passages"):
        for s in sources:
            link = f" — [open]({s['url']})" if s.get("url") else ""
            st.markdown(f"**[{s['number']}] {s['label']}** · relevance {s['score']:.2f}{link}")
            st.caption(s["snippet"])


# First launch (or a fresh cloud container): build the index from the bundled docs.
# Also rebuild automatically if the embedding model was changed in settings.
manifest = read_manifest(settings.index_dir)
if not index_exists(settings.index_dir) or (manifest and manifest.get("embedding") != embedding_signature(settings)):
    with st.spinner("First run: downloading the embedding model and indexing documents (about a minute)…"):
        build_index_now()
    manifest = read_manifest(settings.index_dir)

# ---------------- Sidebar: knowledge base management ----------------
with st.sidebar:
    st.header("📚 Knowledge base")
    if manifest:
        st.success(f"{len(manifest['sources'])} documents · {manifest['chunks']} chunks")
        st.caption(f"Last built {manifest['built_at']}")
        with st.expander("Indexed documents"):
            for name in manifest["sources"]:
                st.write("•", name)

    uploads = st.file_uploader("Add documents", accept_multiple_files=True,
                               type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS])
    if st.button("🔄 Rebuild index", use_container_width=True):
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        for f in uploads or []:
            (settings.docs_dir / os.path.basename(f.name)).write_bytes(f.getbuffer())
        with st.spinner("Reading, chunking and embedding documents…"):
            stats = build_index_now()
        get_bot.clear()
        st.success(f"Indexed {stats['documents']} docs → {stats['chunks']} chunks")
        st.rerun()

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"LLM: `{settings.llm_provider} / {settings.chat_model}`  \n"
               f"Embeddings: `{settings.embedding_provider} / {settings.embedding_model}`  \n"
               f"top-k {settings.top_k} · min relevance {settings.min_relevance}")

# ---------------- Main chat ----------------
st.title(f"🤖 {settings.company_name} AI Assistant")
st.caption("Ask anything about company policies, handbooks and internal docs. Answers cite their sources.")

if key := missing_api_key(settings):
    st.error(f"Add `{key}` to your `.env` file (local) or the app's Secrets (Streamlit Cloud) to start chatting.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("**Try:** _What is our leave policy?_ · _How do I report a phishing email?_ · "
                "_What's the hotel limit in metro cities?_")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        render_sources(m.get("sources", []))

if question := st.chat_input("Ask a question about company docs…"):
    history = to_messages(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            sources, tokens = get_bot().stream(question, history)
            answer = st.write_stream(tokens)
            # Only show passages the answer actually cites ([1], [2]…). A "not found" reply cites none.
            cited = {int(n) for n in re.findall(r"\[(\d+)\]", str(answer))}
            source_dicts = [{**s.__dict__, "label": s.label} for s in sources if s.number in cited]
            render_sources(source_dicts)
        except Exception as exc:  # surface API / rate-limit errors in the UI
            msg = str(exc)
            if "429" in msg or "rate limit" in msg.lower():
                msg = "The free API rate limit was hit — wait a minute and try again."
            answer, source_dicts = f"⚠️ {msg}", []
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": source_dicts})
