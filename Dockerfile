# ─── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for layer caching)
COPY pyproject.toml requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ─── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Copy only model checkpoints from artifacts (not raw data or large images)
COPY artifacts/EXP-LONGITUDINAL-001/temporal_gru_cpu.pt     artifacts/EXP-LONGITUDINAL-001/temporal_gru_cpu.pt
COPY artifacts/EXP-FUSION-BIMODAL-001/clinical_plus_mri.pt  artifacts/EXP-FUSION-BIMODAL-001/clinical_plus_mri.pt
COPY artifacts/EXP-EEG-STANDALONE-001/                       artifacts/EXP-EEG-STANDALONE-001/
COPY artifacts/EXP-BASELINE-001/metrics.json                 artifacts/EXP-BASELINE-001/metrics.json
COPY artifacts/EXP-LONGITUDINAL-001/metrics.json             artifacts/EXP-LONGITUDINAL-001/metrics.json
COPY artifacts/EXP-FUSION-BIMODAL-001/metrics.json           artifacts/EXP-FUSION-BIMODAL-001/metrics.json
COPY artifacts/EXP-EXPLAIN-001/metrics.json                  artifacts/EXP-EXPLAIN-001/metrics.json

# Create writable directory for SQLite DB
RUN mkdir -p /data

# Set environment
ENV PYTHONPATH=/app/src
ENV DATABASE_URL=sqlite:////data/cerebro_x.db
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch API
CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0", "--port", "8000"]
