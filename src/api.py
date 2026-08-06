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
        raise HTTPException(status_code=400, detail="Provide either a PDF file or a DOI.")

    temp_path = None
    try:
        if file:
            temp_path = f"temp_{file.filename}"
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            result = process_input(pdf_path=temp_path)
        else:
            result = process_input(doi=doi)

        summary_fields = summarize_paper(result["full_text"])

        response = {
            "source": result["source"],
            "title": result["title"],
            **summary_fields,
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/")
async def root():
    return {"status": "AI Research Paper Summarizer API is running"}
