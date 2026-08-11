import sys
import os

# Ensure this directory is on the path so sibling imports (pipeline, summarize, etc.) work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
from fastapi import FastAPI, UploadFile, File, Form, Request, Response, HTTPException
from fastapi.responses import JSONResponse

from pipeline import process_input
from summarize import summarize_paper
from user_session import get_or_create_session_id, check_and_increment_usage, init_db

app = FastAPI(title="AI Research Paper Summarizer")

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

        summary_result = summarize_paper(result["full_text"])

        return JSONResponse({
            "title": result["title"],
            "source": result["source"],
            **summary_result,
        })

    finally:
        if saved_path and os.path.exists(saved_path):
            os.remove(saved_path)


@app.get("/health")
def health():
    return {"status": "ok"}
