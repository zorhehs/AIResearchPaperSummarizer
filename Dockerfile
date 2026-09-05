FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/

# Mount point for the SQLite database (see DB_PATH in docker-compose.yml).
RUN mkdir -p /app/data

WORKDIR /app/src
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
