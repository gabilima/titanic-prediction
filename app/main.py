"""
FastAPI Application for Titanic Survival Prediction

This module provides the main FastAPI application with endpoints for prediction
and A/B testing functionality.
"""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app, generate_latest, CONTENT_TYPE_LATEST
import time
import uvicorn
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel
import mlflow
import psutil
import os
import json

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.api.endpoints import predict, health
from app.monitoring.metrics import (
    get_prediction_count,
    increment_prediction_counter,
    record_prediction_latency,
    get_system_metrics
)

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Initialize metrics exporter without predictor (we'll set it during startup)
metrics_exporter = None

async def collect_metrics_periodically():
    """Collect metrics periodically."""
    while True:
        try:
            # Collect system metrics
            process = psutil.Process()
            memory_info = process.memory_info()
            
            await asyncio.sleep(300)  # Collect every 5 minutes
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            await asyncio.sleep(60)  # Retry after 1 minute on error

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for the FastAPI application."""
    # Startup: load model and resources
    logger.info("Starting application, initializing resources")
    try:
        # Try loading from local file
        try:
            from app.ml.model import TitanicModel
            model = TitanicModel()
            model.load()
            app.state.model = model
            logger.info("Model loaded successfully from local file")
        except Exception as e:
            logger.error(f"Failed to load model from local file: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
        
        # Start metrics collection
        asyncio.create_task(collect_metrics_periodically())
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

    yield
    # Shutdown: clean up resources
    logger.info("Shutting down application")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request timing and logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(time.time()))
    request_logger = get_logger("api.request")
    
    start_time = time.time()
    request_logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={"trace_id": request_id}
    )
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    process_time_ms = round(process_time * 1000, 2)
    
    request_logger.info(
        f"Request completed in {process_time_ms}ms",
        extra={
            "trace_id": request_id,
            "status_code": response.status_code,
            "latency_ms": process_time_ms
        }
    )
    
    response.headers["X-Process-Time"] = str(process_time_ms)
    response.headers["X-Request-ID"] = request_id
    return response

# Include routers
app.include_router(predict.router, prefix=settings.API_V1_STR)
app.include_router(health.router)  # Remove prefix for health endpoint

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint returning basic API info."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "description": settings.DESCRIPTION,
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )
