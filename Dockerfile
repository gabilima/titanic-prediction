# ===========================================
# Multi-stage Dockerfile for Titanic Prediction API
# ===========================================

# Base image
FROM python:3.11-slim as base
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        gcc \
        python3-dev \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies stage
FROM base as deps
WORKDIR /app
COPY requirements/prod.txt requirements/dev.txt requirements/test.txt /tmp/requirements/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements/prod.txt && \
    rm -rf /tmp/requirements

# Production stage
FROM base as production
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy only necessary files from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# Copy application code
COPY ./app ./app

# Create necessary directories
RUN mkdir -p /app/models /app/logs && \
    chown -R appuser:appuser /app

# Copy model files
COPY ./models/* /app/models/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Start the application
CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

