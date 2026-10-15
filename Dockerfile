FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/hf FASTEMBED_CACHE_PATH=/app/.cache/fastembed
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Pre-build the index and download the embedding model at build time so the
# container starts fast and needs no network except the LLM API.
RUN EMBEDDING_BACKEND=${EMBEDDING_BACKEND:-fastembed} python scripts/ingest_documents.py \
 && useradd -m -u 1000 app && chown -R app /app
USER app

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
