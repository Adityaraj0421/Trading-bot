# Crypto Trading Agent v6.0 — Production Container
# Runs: FastAPI server (main) + Trading Agent (background thread)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for state persistence
RUN mkdir -p /app/data

# Environment defaults
ENV DATA_DIR=/app/data
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV TRADING_MODE=paper
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "api.server"]
