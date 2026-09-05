import sys
import os
import json
import re
import uuid

# Ensure this directory is on the path so sibling imports (pipeline, summarize, etc.) work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
from fastapi import FastAPI, UploadFile, File, Form, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from pipeline import process_input
from summarize import summarize_paper, stream_summarize_paper, answer_question
from user_session import (
    check_and_increment_usage,
    check_and_increment_ip_usage,
    refund_usage,
    refund_ip_usage,
    client_ip,
    init_db,
    router as user_router,
)

SESSION_COOKIE = "session_id"


def _read_or_create_session(request: Request) -> str:
    """Return the caller's session id, minting a new one if the request has no
    cookie yet.

    The session middleware resolves this once per request and stashes it on
    request.state, so every handler in a request sees the same id.
    """
    existing = getattr(request.state, "session_id", None)
    if existing:
        return existing
    return request.cookies.get(SESSION_COOKIE) or str(uuid.uuid4())


def _set_session_cookie(resp: Response, session_id: str):
    resp.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        max_age=60 * 60 * 24 * 365,
        httponly=True,
    )

app = FastAPI(title="AI Research Paper Summarizer")

from fastapi.middleware.cors import CORSMiddleware
# Sessions ride on a cookie, so credentialed cross-origin requests must stay
# off: a wildcard origin combined with allow_credentials=True would let any
# site drive a visitor's session. Same-origin requests from the bundled UI
# send the cookie regardless of this policy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def session_middleware(request: Request, call_next):
    """Resolve the caller's session id once and set the cookie on every response.

    Previously the cookie was only attached to successful /summarize responses,
    so a client whose requests kept failing (or one that never loaded the UI)
    was handed a brand-new identity each time and the daily limit never bound.
    Setting it here covers error responses and the SSE stream alike.
    """
    session_id = request.cookies.get(SESSION_COOKIE) or str(uuid.uuid4())
    request.state.session_id = session_id
    response = await call_next(request)
    if SESSION_COOKIE not in request.cookies:
        _set_session_cookie(response, session_id)
    return response


app.include_router(user_router)

init_db()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _save_upload(file: UploadFile) -> str:
    """Persist an uploaded file under a generated name and return its path.

    The client-supplied filename is never used to build the path: a name like
    "../../etc/passwd" would otherwise escape UPLOAD_DIR, and the caller's
    cleanup would then delete whatever it landed on. Only the extension is
    carried over, and only when it is a plain alphanumeric suffix."""
    ext = os.path.splitext(file.filename or "")[1]
    if not re.fullmatch(r"\.[A-Za-z0-9]{1,8}", ext or ""):
        ext = ".pdf"
    saved_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return saved_path


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _ground_citations(payload: dict, full_text: str, page_spans) -> dict:
    """Verify each key finding's supporting quote against the paper text and
    map it to a page number.

    Findings whose quote cannot be found stay visibly unverified — the tool
    never silently trusts the model. Pure text search, so grounding is honest
    even for cached summaries.
    """
    spans = page_spans or [[0, len(full_text or "")]]
    grounded = 0
    for f in payload.get("key_findings") or []:
        f["citation"] = {"verified": False, "page": None}
        quote = (f.get("quote") or "").strip()
        nq = _normalize_ws(quote)
        if len(nq) < 12:
            continue  # too short to verify meaningfully
        for i, span in enumerate(spans):
            start, end = int(span[0]), int(span[1])
            if nq in _normalize_ws(full_text[start:end]):
                f["citation"] = {"verified": True, "page": i + 1}
                grounded += 1
                break
    payload["grounded_citations"] = grounded
    return payload


@app.post("/summarize")
async def summarize(
    request: Request,
    file: UploadFile = File(None),
    doi: str = Form(None),
):
    if not file and not doi:
        raise HTTPException(status_code=400, detail="Provide either a PDF file or a DOI.")

    session_id = _read_or_create_session(request)
    ip = client_ip(request)

    # The address backstop is checked first so a client that cycles cookies to
    # dodge the session limit still runs into it (no-op unless configured).
    check_and_increment_ip_usage(ip)
    try:
        check_and_increment_usage(session_id)
    except HTTPException:
        refund_ip_usage(ip)
        raise

    saved_path = None
    delivered = False
    try:
        if file:
            saved_path = _save_upload(file)
            result = process_input(pdf_path=saved_path)
        else:
            result = process_input(doi=doi)

        if result["source"] == "error":
            raise HTTPException(status_code=422, detail=result["error"])

        if not result["full_text"]:
            raise HTTPException(
                status_code=422,
                detail="Could not extract usable text from this input. Please upload the PDF directly or use a DOI that exposes the abstract/text."
            )

        try:
            summary_result = summarize_paper(
                result["full_text"],
                title=result.get("title", ""),
                abstract=result.get("abstract", ""),
                source=result["source"],
            )
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise HTTPException(
                    status_code=503,
                    detail="Our AI provider's daily quota is temporarily exhausted. Please try again later."
                )
            raise HTTPException(status_code=500, detail=f"Summarization failed: {error_msg}")

        # Structured model summary + pipeline metadata; pipeline values win
        # where they were verified (Crossref) or extracted directly from the PDF.
        payload = dict(summary_result)
        payload["source"] = result["source"]
        payload["abstract"] = result.get("abstract", "")
        payload["year"] = result.get("year", "")
        payload["journal"] = result.get("journal", "")
        payload["cited_by"] = result.get("cited_by")
        if result.get("title"):
            payload["title"] = result["title"]
        if result.get("authors"):
            payload["authors"] = result["authors"]
        page_spans = result.get("page_spans") or [[0, len(result.get("full_text", ""))]]
        payload["page_spans"] = page_spans
        payload["page_count"] = len(page_spans)
        _ground_citations(payload, result["full_text"], page_spans)
        delivered = True
        return JSONResponse(payload)

    finally:
        # The credit was reserved before any work began; hand it back unless the
        # caller actually got a summary out of it.
        if not delivered:
            refund_usage(session_id)
            refund_ip_usage(ip)
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)


@app.post("/summarize/stream")
async def summarize_stream(
    request: Request,
    file: UploadFile = File(None),
    doi: str = Form(None),
):
    """Server-sent-events variant of /summarize: emits a `meta` event, one
    `section_done` event per finished section, and a final `done` event carrying
    the same payload as /summarize."""
    if not file and not doi:
        raise HTTPException(status_code=400, detail="Provide either a PDF file or a DOI.")

    session_id = _read_or_create_session(request)
    ip = client_ip(request)

    # The address backstop is checked first so a client that cycles cookies to
    # dodge the session limit still runs into it (no-op unless configured).
    check_and_increment_ip_usage(ip)
    try:
        check_and_increment_usage(session_id)
    except HTTPException:
        refund_ip_usage(ip)
        raise

    saved_path = None
    streaming = False
    try:
        if file:
            saved_path = _save_upload(file)
            result = process_input(pdf_path=saved_path)
        else:
            result = process_input(doi=doi)

        if result["source"] == "error":
            raise HTTPException(status_code=422, detail=result["error"])

        if not result["full_text"]:
            raise HTTPException(
                status_code=422,
                detail="Could not extract usable text from this input. Please upload the PDF directly or use a DOI that exposes the abstract/text."
            )

        meta = {
            "title": result["title"],
            "source": result["source"],
            "abstract": result.get("abstract", ""),
            "authors": result.get("authors", []),
            "year": result.get("year", ""),
            "journal": result.get("journal", ""),
            "cited_by": result.get("cited_by"),
        }

        def event_source():
            yield f"data: {json.dumps({'type': 'meta', 'meta': meta})}\n\n"
            try:
                for event in stream_summarize_paper(
                    result["full_text"],
                    title=result.get("title", ""),
                    abstract=result.get("abstract", ""),
                    source=result["source"],
                ):
                    # Merge the paper metadata into the final payload so the
                    # streamed result carries title/authors like /summarize
                    # does. Pipeline-verified metadata wins where present.
                    if event.get("type") == "done" and isinstance(event.get("result"), dict):
                        meta_clean = {k: v for k, v in meta.items() if v not in ("", None, [])}
                        merged = {**event["result"], **meta_clean}
                        page_spans = result.get("page_spans") or [[0, len(result.get("full_text", ""))]]
                        merged["page_spans"] = page_spans
                        merged["page_count"] = len(page_spans)
                        _ground_citations(merged, result["full_text"], page_spans)
                        event = {**event, "result": merged}
                    if event.get("type") == "error":
                        refund_usage(session_id)
                        refund_ip_usage(ip)
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                refund_usage(session_id)
                refund_ip_usage(ip)
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

        streaming = True
        return StreamingResponse(event_source(), media_type="text/event-stream")
    finally:
        # Refund setup failures here; failures raised once the stream is already
        # running are refunded inside event_source() instead.
        if not streaming:
            refund_usage(session_id)
            refund_ip_usage(ip)
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)


@app.post("/chat")
async def chat(payload: dict):
    paper_text = payload.get("paper_text", "")
    question = payload.get("question", "").strip()
    chat_history = payload.get("chat_history", [])

    if not paper_text:
        raise HTTPException(status_code=400, detail="No paper loaded. Summarize a paper first.")
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = answer_question(paper_text, question, chat_history)
    except Exception as e:
        error_msg = str(e)
        if "Ollama is not running" in error_msg:
            raise HTTPException(status_code=503, detail=error_msg)
        raise HTTPException(status_code=500, detail=f"Chat failed: {error_msg}")

    return JSONResponse({"answer": answer})


@app.get("/health")
def health():
    return {"status": "ok"}


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")

@app.get("/")
def serve_ui():
    # The session cookie is attached by session_middleware, so the usage counter
    # and daily limit track the same session_id from the first page load on.
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
