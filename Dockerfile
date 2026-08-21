FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GALIT_MAX_BATCH_SIZE=25

WORKDIR /app
RUN addgroup --system galit && adduser --system --ingroup galit galit
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY api.py ./
COPY galit ./galit
RUN chown -R galit:galit /app
USER galit
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2)"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
