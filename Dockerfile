FROM python:3.11-slim

# Install system build dependencies needed by shapely, numpy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgeos-dev \
    libgdal-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first to avoid legacy resolver issues
RUN pip install --upgrade pip

# Install dependencies with relaxed version pinning to avoid conflicts
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
