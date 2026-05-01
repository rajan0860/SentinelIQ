# ── SentinelIQ Dockerfile ────────────────────────────────────────────────────
#
# Multi-stage build:
#   Stage 1 (builder) — install Python dependencies into a venv
#   Stage 2 (runtime) — copy only the venv + source, keeping the image lean
#
# The image exposes both the FastAPI backend (8000) and the Streamlit
# dashboard (8501). In docker-compose these are run as separate services
# sharing this same image.
#
# Build:
#   docker build -t sentineliq .
#
# Run API only:
#   docker run -p 8000:8000 --env-file .env sentineliq \
#       uvicorn src.api.main:app --host 0.0.0.0 --port 8000
#
# Run dashboard only:
#   docker run -p 8501:8501 --env-file .env sentineliq \
#       streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed to compile some Python packages (e.g. chromadb, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated venv so we can copy it cleanly to the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY src/       ./src/
COPY scripts/   ./scripts/
COPY lib/       ./lib/
COPY .env.example .env.example

# Create data directories so volume mounts work correctly on first run
RUN mkdir -p \
        data/raw \
        data/processed \
        data/synthetic \
        data/graphs \
        data/models \
        data/chroma

# Expose both service ports
EXPOSE 8000 8501

# Default command — override in docker-compose per service
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
