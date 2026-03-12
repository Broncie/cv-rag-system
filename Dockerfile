# Dockerfile - defines how to build the Docker image for the RAG API service
FROM python:3.11-slim

WORKDIR /app

# System deps (faiss-cpu wheels usually work without extra libs; keep slim)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Production-ish single process; for multi-worker later use gunicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]