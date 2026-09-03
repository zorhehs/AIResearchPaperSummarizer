# AI Research Paper Summarizer

Upload a research paper (a PDF file, or just a DOI) and get a structured summary back: what the paper is about, the problem it tackles, the approach, the key findings with page-grounded quotes, the headline numbers, the limitations, and the future work. The same page also draws an interactive knowledge graph of the paper and lets you ask follow-up questions in chat.

The backend is FastAPI; the frontend is a single-file vanilla-JS page. No build step, no npm, no separate client to deploy.

## Features

- **Structured summaries** — every section comes back from one consolidated LLM call as a single JSON object, validated against a Pydantic schema before it is served.
- **Page-grounded findings** — each key finding carries a short verbatim supporting quote. The server searches the paper text for every quote and reports which page it appears on. Quotes that can't be found are marked *unverified* instead of being silently trusted.
- **Knowledge graph** — sections, authors, and key concepts drawn on a canvas.
- **Paper chat** — ask the paper questions and get answers grounded in its text.
- **DOI support** — Unpaywall resolves open-access PDFs; Crossref supplies authors, year, journal, and citation counts where available.
- **Caching** — the same paper text always hashes to the same key, so re-summarizing an identical paper is served from SQLite and costs nothing.
- **Daily limit** — 5 summaries per session per day, tracked by cookie.

## How a summary is generated

`src/pipeline.py` turns the input into clean text (plus per-page spans used for grounding). `src/summarize.py` then produces the summary on Groq — `openai/gpt-oss-20b` primary, with rotation to `gpt-oss-120b` and `qwen/qwen3.8-27b` when a model's free-tier daily quota is exhausted — and a local Ollama model as the last resort.

Two design choices exist mostly because of the Groq free tier:

- the whole summary is **one** request, not six parallel section calls, and
- the input is capped: papers that fit are passed through whole; longer ones are sampled from the head, middle, and tail so the request stays inside the per-minute token budget. The model is told it received a truncated excerpt and can flag that in `confidence_notes`.

## Setup

Python 3.11 recommended (the Docker image is `python:3.11-slim`).

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

Create a `.env` in the project root:

```bash
GROQ_API_KEY=gsk_...
# optional — used for Unpaywall lookups
UNPAYWALL_EMAIL=you@example.com
```

`GROQ_API_KEY` is what actually does the work; without it the app can only fall back to a local Ollama instance.

## Run

```bash
cd src
../venv/bin/uvicorn api:app --port 8000
```

Open http://127.0.0.1:8000 — the UI is served by the same process.

Or with Docker:

```bash
docker compose up --build
```

## Tests

```bash
./venv/bin/python -m pytest tests/ -v
```

37 tests, all offline — the DB is swapped for a temp file per test and the model API is never called.

## API

| Endpoint | Description |
|---|---|
| `GET /` | the frontend |
| `POST /summarize` | multipart form with `file` (PDF) or `doi`; returns the full summary JSON |
| `POST /summarize/stream` | same input as SSE events (`meta`, `section_done`, `done`, `error`) |
| `POST /chat` | `{paper_text, question, chat_history}` → answer |
| `GET /health` | liveness check |
| `GET /usage-status` | summaries used today for this session |
| `POST /save-email` / `GET /get-email` | optional email capture |

Input errors map to 400/404/422; provider quota problems map to 503.

## Project layout

| Path | Purpose |
|---|---|
| `src/api.py` | routes, session cookies, citation grounding |
| `src/pipeline.py` | PDF/DOI input → text + metadata |
| `src/summarize.py` | LLM clients, summary schema, cache |
| `src/extract.py`, `src/clean.py`, `src/metadata.py` | PDF text extraction, cleaning, title detection |
| `src/fetch_doi.py` | Unpaywall/Crossref lookups |
| `src/user_session.py` | sessions, daily limits, email capture |
| `static/index.html` | the entire frontend |
| `tests/` | pytest suite |

## Notes

- **Quotes are only as reliable as the model.** Grounding is literal text search: a perfectly true finding whose quote the model slightly reworded will show as *unverified*. That's deliberate — the server won't pretend a quote checks out when it doesn't.
- **The free tier shaped the code.** Per-minute and per-day token budgets are the reason for the single consolidated call, the input cap, and the model rotation. If you hit quota errors, that's the tier, not a bug.
- **Email capture stores the address and nothing else** — it isn't used for anything yet.
- `src/map_reduce.py` is older experiment code and is not imported by the server.
- No license has been chosen for this repo yet.
