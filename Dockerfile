# ===========================================
# Multi-stage Dockerfile for Titanic Prediction API
# ===========================================

# Base stage for shared dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies stage
FROM base as deps

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements/prod.txt requirements/dev.txt requirements/test.txt /tmp/requirements/

# Install production dependencies
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements/prod.txt && \
    pip install gunicorn==20.1.0

# Development stage
FROM deps as development

# Install development dependencies
RUN pip install -r /tmp/requirements/dev.txt

# Copy project files
COPY . .

# Testing stage
FROM development as testing

# Install test dependencies
RUN pip install -r /tmp/requirements/test.txt

# Run tests
RUN pytest

# Production stage
FROM base as production

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy only necessary files from deps stage
COPY --from=deps /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=deps /usr/local/bin/gunicorn /usr/local/bin/gunicorn
COPY ./app ./app
COPY ./models ./models

# Create necessary directories with correct permissions
RUN mkdir -p /app/data/raw /app/data/processed /app/logs \
    && chown -R appuser:appuser /app \
    && find /app -type d -exec chmod 755 {} \; \
    && find /app -type f -exec chmod 644 {} \;

# Set environment variables for production
ENV ENV=production \
    MODEL_VERSION=latest \
    ENABLE_PREDICTION_CACHE=true \
    CACHE_TTL_SECONDS=3600 \
    BATCH_SIZE_LIMIT=100 \
    ENABLE_RESPONSE_COMPRESSION=true \
    COMPRESSION_MINIMUM_SIZE=1000 \
    PATH="/usr/local/bin:$PATH"

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Run the application with Gunicorn
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app.main:app"]

