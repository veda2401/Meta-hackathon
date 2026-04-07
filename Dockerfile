# ─── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="power-grid-team"
LABEL description="Power Grid Optimization Environment"

# ─── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ───────────────────────────────────────────────────────
WORKDIR /app

# ─── Install Python dependencies ─────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy project ────────────────────────────────────────────────────────────
COPY . .

# ─── Expose port for HF Spaces / Docker ──────────────────────────────────────
EXPOSE 7860

# ─── Default: start the application ──────────────────────────────────────────
CMD ["python", "-m", "server.app", "--host", "0.0.0.0", "--port", "7860"]
