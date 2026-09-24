# Experiment 6 — Containerization & API Deployment
# Cloud-Edge Tradeoff: Adaptive Workload Placement

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer) so `docker build` only
# re-installs packages when requirements.txt actually changes, not on
# every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code and saved models
COPY main.py .
COPY models/ ./models/

EXPOSE 8000

# Basic container-level healthcheck hitting our own /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
