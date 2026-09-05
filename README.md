# AI Research Paper Summarizer

[![tests](https://github.com/zorhehs/AIResearchPaperSummarizer/actions/workflows/tests.yml/badge.svg)](https://github.com/zorhehs/AIResearchPaperSummarizer/actions/workflows/tests.yml)

Upload a research paper (a PDF file, or just a DOI) and get a structured summary back: what the paper is about, the problem it tackles, the approach, the key findings with page-grounded quotes, the headline numbers, the limitations, and the future work. The same page also draws an interactive knowledge graph of the paper and lets you ask follow-up questions in chat.

The backend is FastAPI; the frontend is a single-file vanilla-JS page. No build step, no npm, no separate client to deploy.

## Features

- **Structured summaries** — every section comes back from one consolidated LLM call as a single JSON object, validated against a Pydantic schema before it is served.
- **Page-grounded findings** — each key finding carries a short verbatim supporting quote. The server searches the paper text for every quote and reports which page it appears on. Quotes that can't be found are marked *unverified* instead of being silently trusted.
- **Knowledge graph** — sections, authors, and key concepts drawn on a canvas.
- **Paper chat** — ask the paper questions and get answers grounded in its text.
- **DOI support** — Unpaywall resolves open-access PDFs; Crossref supplies authors, year, journal, and citation counts where available.
- **Scanned papers** — a PDF with no text layer is OCR'd automatically (Tesseract, via PyMuPDF). Page numbers survive OCR, so citation grounding still works.
- **Caching** — the same paper text always hashes to the same key, so re-summarizing an identical paper is served from SQLite and costs nothing. Crossref and Unpaywall lookups are cached too, so a repeated DOI does not re-hit either API.
- **Daily limit** — 5 summaries per session per day, tracked by cookie. Requests
  that fail before producing a summary are refunded. An optional per-address
  backstop (`IP_DAILY_LIMIT`) bounds clients that discard cookies.

## How a summary is generated

`src/pipeline.py` turns the input into clean text (plus per-page spans used for grounding). `src/summarize.py` then produces the summary on Groq — `openai/gpt-oss-20b` primary, with rotation to `gpt-oss-120b` and `qwen/qwen3-32b` when a model's free-tier daily quota is exhausted — and a local Ollama model as the last resort.

Two design choices exist mostly because of the Groq free tier:

- the whole summary is **one** request, not six parallel section calls, and
- the input is capped: papers that fit are passed through whole; longer ones are sampled from the head, middle, and tail so the request stays inside the per-minute token budget. The model is told it received a truncated excerpt and can flag that in `confidence_notes`.

## Setup

Python 3.11 recommended (the Docker image is `python:3.11-slim`).

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements-dev.txt
```

`requirements.txt` holds the runtime dependencies alone — that is what the
Docker image installs. `requirements-dev.txt` adds the test tooling on top.

Create a `.env` in the project root (see [.env.example](.env.example)):

```bash
GROQ_API_KEY=gsk_...
# optional — your contact address, required by Unpaywall on every request
UNPAYWALL_EMAIL=you@example.com
# optional — where the SQLite file lives (default: ./users.db)
DB_PATH=/absolute/path/to/users.db
# optional — per-address daily cap, off by default (see Notes)
IP_DAILY_LIMIT=0
TRUST_PROXY_HEADERS=0
# optional — OCR for scanned PDFs; on when Tesseract is installed
ENABLE_OCR=1
OCR_MAX_PAGES=20
# optional — application log level (default INFO)
LOG_LEVEL=INFO
```

For OCR of scanned PDFs, install Tesseract too (`brew install tesseract` on
macOS, `apt install tesseract-ocr` on Debian/Ubuntu). It is optional — without
it, scanned uploads return an error saying OCR is unavailable, and everything
else works unchanged. The Docker image installs it for you.

`GROQ_API_KEY` is what actually does the work; without it the app can only fall
back to a local Ollama instance. `UNPAYWALL_EMAIL` has no default — leave it
unset and DOI lookups skip Unpaywall and use Crossref metadata alone, so
open-access PDFs won't be fetched.

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

```bash
node --test "tests/frontend/*.test.mjs"
```

129 Python tests and 24 frontend tests, all offline — the DB is swapped for a
temp file per test and the model API is never called. The OCR tests build a
scanned PDF on the fly and skip themselves where Tesseract is missing.

The frontend tests need no npm install and no build step: they read the inline
`<script>` out of `static/index.html` and evaluate it in a sandboxed stub DOM
(`tests/frontend/harness.mjs`), then exercise the pure helpers directly. One
test runs the same DOI corpus through `normalizeDoi()` and `normalize_doi()`
and fails if the two ever disagree.

Both suites run in CI on every push and pull request
(`.github/workflows/tests.yml`).

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

Input errors map to 400 (nothing supplied) and 422 (unreadable PDF, unresolvable DOI, no extractable text); hitting the daily limit maps to 429; provider quota problems map to 503. A request that fails before a summary is delivered does not consume one of the day's five summaries.

## Project layout

| Path | Purpose |
|---|---|
| `src/api.py` | routes, session cookies, citation grounding |
| `src/pipeline.py` | PDF/DOI input → text + metadata |
| `src/summarize.py` | LLM clients, summary schema, cache |
| `src/extract.py`, `src/clean.py`, `src/metadata.py` | PDF text extraction, cleaning, title detection |
| `src/fetch_doi.py` | Unpaywall/Crossref lookups |
| `src/user_session.py` | sessions, daily limits, email capture |
| `src/cache.py` | TTL cache for Crossref/Unpaywall lookups |
| `static/index.html` | the entire frontend |
| `tests/` | pytest suite |
| `tests/frontend/` | frontend tests + the sandbox harness |

## Notes

- **Quotes are only as reliable as the model.** Grounding is literal text search: a perfectly true finding whose quote the model slightly reworded will show as *unverified*. That's deliberate — the server won't pretend a quote checks out when it doesn't.
- **External lookups are cached for 30 days** (misses for 1 day, since a
  paywalled paper may open up later). Transient failures — a timeout, a 5xx —
  are deliberately not cached, so an outage is never remembered as a fact
  about the paper.
- **The free tier shaped the code.** Per-minute and per-day token budgets are the reason for the single consolidated call, the input cap, and the model rotation. If you hit quota errors, that's the tier, not a bug.
- **The daily limit is advisory by default.** Identity is a cookie, so a client
  that discards cookies gets a fresh session and an unlimited allowance. Setting
  `IP_DAILY_LIMIT` adds a per-address cap that closes that path. It is off by
  default deliberately: universities, offices and mobile carriers put thousands
  of people behind one address, and capping those punishes ordinary users to
  inconvenience an attacker who can rent another address cheaply. Enable
  `TRUST_PROXY_HEADERS` only behind a proxy you control — otherwise callers can
  forge `X-Forwarded-For` and the cap is worse than useless.
- **OCR needs Tesseract on the host.** The Docker image installs it. Without it,
  scanned PDFs return an honest error saying so rather than failing vaguely.
  OCR runs only when a PDF has no text layer, is capped at `OCR_MAX_PAGES`, and
  costs a few seconds per page.
- **Email capture stores the address and nothing else** — it isn't used for anything yet.
- Licensed under the MIT License — see [LICENSE](LICENSE).
