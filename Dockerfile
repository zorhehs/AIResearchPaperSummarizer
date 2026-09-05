FROM python:3.11-slim

WORKDIR /app

# Tesseract powers the OCR fallback for scanned PDFs (see ENABLE_OCR).
# Without it the app still runs — scanned uploads just return an honest
# error instead of being read.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/

# Mount point for the SQLite database (see DB_PATH in docker-compose.yml).
RUN mkdir -p /app/data

WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
