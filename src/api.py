import sys
import os
import json

# Ensure this directory is on the path so sibling imports (pipeline, summarize, etc.) work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
from fastapi import FastAPI, UploadFile, File, Form, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from pipeline import process_input
from summarize import summarize_paper, stream_summarize_paper, answer_question
from user_session import get_or_create_session_id, check_and_increment_usage, init_db, router as user_router

app = FastAPI(title="AI Research Paper Summarizer")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(user_router)

init_db()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/summarize")
async def summarize(
    request: Request,
    response: Response,
    file: UploadFile = File(None),
    doi: str = Form(None),
):
    if not file and not doi:
        raise HTTPException(status_code=400, detail="Provide either a PDF file or a DOI.")

    session_id = get_or_create_session_id(request, response)

    try:
        check_and_increment_usage(session_id)
    except HTTPException:
        raise

    saved_path = None
    try:
        if file:
            saved_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(saved_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            result = process_input(pdf_path=saved_path)
        else:
            result = process_input(doi=doi)

        if result["source"] == "error":
            raise HTTPException(status_code=422, detail=result["error"])

        if not result["full_text"]:
            raise HTTPException(
                status_code=422,
                detail="Could not extract usable text from this input."
            )

        try:
            summary_result = summarize_paper(result["full_text"])
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise HTTPException(
                    status_code=503,
                    detail="Our AI provider's daily quota is temporarily exhausted. Please try again later."
                )
            raise HTTPException(status_code=500, detail=f"Summarization failed: {error_msg}")

        return JSONResponse({
            "title": result["title"],
            "source": result["source"],
            "abstract": result.get("abstract", ""),
            "authors": result.get("authors", []),
            "year": result.get("year", ""),
            "journal": result.get("journal", ""),
            "cited_by": result.get("cited_by"),
            **summary_result,
        })

    finally:
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)


@app.post("/summarize/stream")
async def summarize_stream(
    request: Request,
    response: Response,
    file: UploadFile = File(None),
    doi: str = Form(None),
):
    """Server-sent-events variant of /summarize: emits a `meta` event, one
    `section_done` event per finished section, and a final `done` event carrying
    the same payload as /summarize."""
    if not file and not doi:
        raise HTTPException(status_code=400, detail="Provide either a PDF file or a DOI.")

    session_id = get_or_create_session_id(request, response)

    try:
        check_and_increment_usage(session_id)
    except HTTPException:
        raise

    saved_path = None
    try:
        if file:
            saved_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(saved_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            result = process_input(pdf_path=saved_path)
        else:
            result = process_input(doi=doi)

        if result["source"] == "error":
            raise HTTPException(status_code=422, detail=result["error"])

        if not result["full_text"]:
            raise HTTPException(
                status_code=422,
                detail="Could not extract usable text from this input."
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
                for event in stream_summarize_paper(result["full_text"]):
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")
    finally:
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
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
