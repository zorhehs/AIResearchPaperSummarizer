# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

AI Research Paper Summarizer: upload a research paper PDF or paste a DOI and get a structured summary (Summary, Methodology, Research Gaps, Findings, Future Work), an Obsidian-style knowledge graph, and a chat to ask questions about the paper.

## Commands

```bash
# Install (Python 3.11, local venv at ./venv)
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# Run the dev server (must run from src/ so sibling imports resolve)
cd src && ../venv/bin/uvicorn api:app --port 8000
# UI is served at http://127.0.0.1:8000 by the same server

# Run tests (must be from project root)
./venv/bin/python -m pytest tests/ -v

# Docker
docker compose up --build
```

Environment: `.env` in project root with `GROQ_API_KEY=gsk_...` (required) and optionally `UNPAYWALL_EMAIL`. Loaded via `python-dotenv` at import time in `pipeline.py` and `summarize.py`.

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

Key behaviors to preserve when changing code:

- **Input routing** (`src/pipeline.py::process_input`): PDF path → extract/clean/metadata; DOI → Unpaywall PDF URL → download → same PDF path, else abstract-only from Crossref. Returns a dict with `source`, `title`, `abstract`, `full_text`, `error`.
- **Summarization** (`src/summarize.py`): papers ≤ 60k chars (`SINGLE_PASS_CHAR_LIMIT`) summarized in one pass; longer ones condensed via `src/map_reduce.py` (chunk → summarize → combine). Each section strategy guards against LLM echoing the prompt (TEMPLATE_PHRASES) and meta preambles (META_PHRASES), with retries. Groq `openai/gpt-oss-120b` primary, `gpt-oss-20b` fallback, local Ollama (`llama3.2:1b` at `localhost:11434`) as last resort.
- **Summary cache**: SHA-256 of full text → `summary_cache` table in `users.db` (SQLite). Identical papers never re-hit the LLM.
- **Rate limiting** (`src/user_session.py`): cookie-based `session_id`, 5 summaries per session per day (`DAILY_LIMIT`), `users` + `usage` tables in SQLite.
- **API** (`src/api.py`): `/summarize` (multipart PDF or DOI form), `/summarize/stream` (same input; SSE with `meta`, `section_done`, `done`, `error` events via `summarize.stream_summarize_paper`), `/chat`, `/health`, `/` serves `static/index.html`; plus `/save-email`, `/get-email`, `/usage-status` from the user router. Rate-limit/quota errors map to 503.

## Conventions

- All backend code lives in `src/`; sibling modules are imported flat (no package-relative imports) — `api.py` and tests both `sys.path.insert` the `src/` directory. Run uvicorn from inside `src/`.
- Tests insert `src/` onto `sys.path` and monkeypatch `user_session.DB_PATH` to a temp DB per test; follow that pattern for new DB-touching tests.
- The frontend is a single self-contained file: `static/index.html` (vanilla JS, canvas knowledge graph) — no build step, no npm.
- **Dashboard layout**: card header (`.card-head`) with title + usage chip on the right; `.input-panel` contains a segmented `.mode-switch` (PDF file / DOI tabs); PDF tab has a centered `.dropzone` → `.file-card` (name, size, remove); results/recent/footer sections below. All dashboard elements get `body.dark` variants.
- SQLite (`users.db`) is the only persistence; tables are created lazily with `CREATE TABLE IF NOT EXISTS`.
- `scripts/` contains manual/scratch experiment scripts, not part of the app or test suite.
