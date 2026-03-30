FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
# CROPEYE_APP must be set in each Railway service's Variables tab.
# 1 replica: use 4 workers for parallel request handling within single container.
ENV GUNICORN_WORKERS=4
CMD ["sh", "-c", "gunicorn run:app -w ${GUNICORN_WORKERS:-4} -k uvicorn.workers.UvicornWorker -b 0.0.0.0:${PORT:-8000}"]
