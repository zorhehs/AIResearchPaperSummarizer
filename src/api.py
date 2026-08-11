import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from pipeline import process_input
from summarize import summarize_paper

app = FastAPI(title="AI Research Paper Summarizer")


@app.post("/summarize")
async def summarize(
    file: UploadFile = File(None),
    doi: str = Form(None),
):
    if not file and not doi:
        raise HTTPException(
            status_code=400,
            detail="Provide either a PDF file or a DOI."
        )

    temp_path = None
    try:
        # --- Step 1: get raw paper content (PDF or DOI) ---
        if file:
            if not file.filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are supported."
                )
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            try:
                result = process_input(pdf_path=temp_path)
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Could not read this PDF. It may be corrupted or not a valid PDF."
                )
        else:
            try:
                result = process_input(doi=doi)
            except Exception:
                raise HTTPException(
                    status_code=502,
                    detail="Error while trying to resolve the DOI. Please try again."
                )

            if not result.get("full_text") or len(result["full_text"].strip()) < 50:
                raise HTTPException(
                    status_code=404,
                    detail=f"Could not find a paper or usable metadata for DOI: {doi}"
                )

        # --- Step 2: run summarization ---
        try:
            summary_fields = summarize_paper(result["full_text"])
        except Exception as e:
            raise HTTPException(
                status_code=504,
                detail=f"Summarization failed or timed out: {str(e)}"
            )

        response = {
            "source": result["source"],
            "title": result["title"],
            **summary_fields,
        }
        return JSONResponse(content=response)

    except HTTPException:
        # re-raise HTTPExceptions as-is, don't let them get caught by the generic handler below
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
async def root():
    return {"status": "AI Research Paper Summarizer API is running"}
