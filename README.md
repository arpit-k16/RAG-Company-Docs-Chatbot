# Project 01 — RAG Chatbot with Company Docs

A private AI assistant that answers employee questions from internal documents (PDFs, Word files,
Markdown, a Notion export and optionally Confluence) using **Retrieval-Augmented Generation**.
Every answer cites the exact document and page it came from.

**Stack:** Python · LangChain · FAISS · Groq (gpt-oss-120b) · FastEmbed (BGE) · Streamlit

**Runs 100% free:** Groq's free API tier for the chat model + a free open-source embedding model that runs
locally on CPU. Switch either one to OpenAI with a single line in `.env`.

```
 You: What is our leave policy?
 Bot: As per the HR policy (v2.1), you get 24 paid leaves per year, including casual and
      sick leaves [1]. Up to 10 unused earned leaves carry forward [2].
      📕 [1] HR_Policy.pdf (p. 1)   📕 [2] HR_Policy.pdf (p. 1)
```

## How it works

```
             INGEST (python ingest.py)                          QUERY (app.py / cli.py)
 ┌──────────┐   ┌─────────┐   ┌──────────────┐   ┌───────┐     ┌──────────────────────┐
 │ PDF/DOCX │   │ load +  │   │ chunk 800    │   │ FAISS │◀────│ 1. condense follow-up│
 │ Markdown │──▶│ metadata│──▶│ overlap 120  │──▶│ index │     │ 2. top-k retrieval   │
 │ Notion   │   │ (source,│   │ + embeddings │   │ (disk)│────▶│ 3. relevance filter  │
 │Confluence│   │  page)  │   │ bge-small    │   └───────┘     │ 4. Groq gpt-oss-120b │
 └──────────┘   └─────────┘   └──────────────┘                 │ 5. answer + [n] cites│
                                                               └──────────────────────┘
```

Design decisions worth talking about in an interview:

- **Grounding + citations.** The system prompt forces the model to answer only from numbered
  passages and cite them as `[n]`; the UI maps those numbers back to file + page.
- **Relevance threshold.** If no chunk clears `MIN_RELEVANCE`, the bot says it couldn't find the
  answer *without calling the LLM* — this removes the most common source of hallucinations and saves cost.
- **Conversational memory.** Follow-ups like *"what about sick leave?"* are rewritten into a standalone
  question before retrieval, so search still works mid-conversation.
- **Metadata-aware chunking.** Splits prefer Markdown headings and paragraphs; each chunk keeps
  its source, 1-based page, and a stable `chunk_id`.
- **Private by design.** Embeddings run locally, so documents and the index never leave the server;
  only the few retrieved passages are sent to the chat model.
- **Provider-agnostic.** `rag/providers.py` picks Groq or OpenAI (chat) and FastEmbed or OpenAI (embeddings)
  from config. The index records which embedding model built it and refuses to load with a different one.
- **Measured, not guessed.** `evaluate.py` reports retrieval hit-rate and answer accuracy on a golden set,
  including an out-of-scope question the bot must refuse.

## Project structure

```
RAG-Company-Docs-Chatbot/
├── app.py               Streamlit chat UI (upload docs, rebuild index, chat with sources)
├── cli.py               Terminal chat
├── ingest.py            Load → chunk → embed → save FAISS index
├── evaluate.py          Retrieval + answer evaluation on data/eval/golden_set.json
├── config.py            All settings (from .env or Streamlit secrets)
├── rag/
│   ├── providers.py     Groq / OpenAI chat models, FastEmbed / OpenAI embeddings
│   ├── loaders.py       PDF, DOCX, Markdown/TXT, Notion export, Confluence
│   ├── index.py         Chunking, embeddings, FAISS build/load
│   └── chatbot.py       Retrieval, prompt, citations, streaming, memory
├── data/
│   ├── docs/            Sample company docs (HR_Policy.pdf, Company_Handbook.md, IT_Security_Policy.md)
│   ├── notion/          Sample Notion export
│   └── eval/            Golden Q&A set
└── tests/test_rag.py    Offline tests (fake embeddings + fake LLM, no API key)
```

## Run it locally (Windows PowerShell)

```powershell
git clone https://github.com/arpit-k16/RAG-Company-Docs-Chatbot.git
cd RAG-Company-Docs-Chatbot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env        # then put your free GROQ_API_KEY in .env (console.groq.com/keys)

pytest -q                     # offline tests, no key needed
python ingest.py              # build the index (first run downloads the ~70 MB embedding model)
streamlit run app.py          # open http://localhost:8501
```

Other ways to use it:

```powershell
python cli.py                                   # terminal chat
python cli.py -q "How do I report a phishing email?"
python evaluate.py                              # accuracy report
```

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a GitHub repo (`.env` and `storage/` are git-ignored, so your key stays private).
2. Go to **share.streamlit.io → Create app**, pick the repo, branch and `app.py`.
   Under **Advanced settings** choose Python 3.11.
3. In **Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_..."
   ```
4. **Deploy.** On first start the app downloads the embedding model and indexes `data/docs` automatically
   (about a minute), then you get a public `https://<name>.streamlit.app` link for your resume.

Why it fits on the free tier: FastEmbed runs the embedding model with ONNX Runtime instead of PyTorch,
so the whole app stays well under the free tier's memory limit.

## Use your own documents

- **Files:** drop PDFs, DOCX, MD or TXT into `data/docs/` (or upload them in the app sidebar) and click
  **Rebuild index**. Or set `DOCS_DIR` in `.env` to an existing folder.
- **Notion:** *Settings → Export → Markdown & CSV*, unzip into `data/notion/`.
- **Confluence:** `pip install atlassian-python-api beautifulsoup4`, fill the `CONFLUENCE_*` values in `.env`,
  run `python ingest.py`.

## Ideas to extend

- Hybrid search (BM25 + vectors) and a cross-encoder re-ranker for better recall on exact terms.
- Per-user access control: store a `department` field in metadata and filter at retrieval time.
- Incremental ingestion keyed on file hash instead of full rebuilds.
- Serve via FastAPI and plug into Slack/Teams.
