# ══════════════════════════════════════════════════
# DiabéticaAI — Production Docker Image
# Multi-stage build for minimal attack surface
# ══════════════════════════════════════════════════

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r diabetica && useradd -r -g diabetica diabetica

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=diabetica:diabetica . .

# Create necessary directories
RUN mkdir -p logs models monitoring/reports && \
    chown -R diabetica:diabetica /app

USER diabetica

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use gunicorn with uvicorn workers for production
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--log-level", "info", \
     "--access-log"]
