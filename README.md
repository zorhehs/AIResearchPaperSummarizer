# AI Research Paper Summarizer

Upload a research paper PDF or paste a DOI and get a structured, research-grade summary — Summary, Methodology, Research Gaps, Findings, and Future Work — plus an Obsidian-style knowledge graph of the paper's sections, authors, and key concepts, and a chat to ask questions about the paper.

## Architecture

```
PDF / DOI ──► pipeline.py ──► summarize.py ──► 5 section strategies (parallel)
              │ extract.py      (Groq gpt-oss-120b,  │ each: echo/meta guard + retry
              │ clean.py         Ollama fallback)    └─► cached in users.db (SQLite)
              │ metadata.py
              └─ fetch_doi.py (Unpaywall) + Crossref (authors, year, citations)
                     │
FastAPI (src/api.py) ─┴─► static/index.html  (vanilla JS, canvas knowledge graph)
```

- Papers **≤ 60k chars**: summarized in one pass with full context.
- Papers **> 60k chars**: condensed first via `map_reduce.py` (chunk → summarize → combine).
- Identical papers are served from the **summary cache** — no quota burned.
- Summaries run through **Groq** (`openai/gpt-oss-120b`, fallback `gpt-oss-20b`), falling back to a local **Ollama** model if Groq is unavailable.
- Rate limit: 5 summaries / session / day (cookie-based, `users` + `usage` tables in SQLite).

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Create a `.env` in the project root:

```
GROQ_API_KEY=gsk_...
# optional
UNPAYWALL_EMAIL=you@example.com
```

## Run

```bash
cd src
../venv/bin/uvicorn api:app --port 8000
```

Open http://127.0.0.1:8000 (the UI is served by the same server).

## Tests

```bash
./venv/bin/python -m pytest tests/ -v
```

## Project layout

| Path | Purpose |
|---|---|
| `src/api.py` | FastAPI server: `/summarize`, `/chat`, `/health`, serves the UI |
| `src/summarize.py` | Per-section LLM strategies, Groq/Ollama clients, summary cache |
| `src/map_reduce.py` | Chunked condensation for very long papers |
| `src/pipeline.py` | Input routing: PDF or DOI → text + metadata |
| `src/extract.py` / `clean.py` / `metadata.py` | PDF text extraction & cleanup |
| `src/fetch_doi.py` | Open-access PDF lookup via Unpaywall |
| `src/user_session.py` | Session cookies + daily usage limits (SQLite) |
| `static/index.html` | Single-file frontend (UI, loader, knowledge graph, chat) |
| `tests/` | pytest suite (API + metadata extraction) |
| `scripts/` | Manual/scratch experiment scripts |
