FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

RUN addgroup --system app \
    && adduser --system --ingroup app app

WORKDIR /app

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app app.py ./
COPY --chown=app:app app_pages ./app_pages
COPY --chown=app:app local_rag ./local_rag
COPY --chown=app:app watcher ./watcher
COPY --chown=app:app config/settings.toml ./config/settings.toml

RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

RUN mkdir -p /app/data /app/logs \
    && chown -R app:app /app/data /app/logs

USER app

EXPOSE 8501

VOLUME ["/app/data", "/app/logs"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.getenv('PORT', '8501')}/_stcore/health\")"]

CMD ["sh", "-c", "exec streamlit run app.py --server.address=0.0.0.0 --server.headless=true --server.port=${PORT:-8501}"]
