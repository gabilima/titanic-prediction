import time
import psutil
import os
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.core.config import settings
from app.core.logging import get_logger
from app.api.schemas.prediction import HealthResponse, SystemMetrics
from app.ml.model import get_model, TitanicModel
from app.ml.pipeline import load_pipeline
from app.core.health import health_config
from app.monitoring.metrics import get_prediction_count

router = APIRouter()
logger = get_logger(__name__)

async def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics."""
    process = psutil.Process(os.getpid())
    return {
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "thread_count": process.num_threads(),
        "open_files": len(process.open_files()),
        "predictions_per_minute": get_prediction_count()
    }

@router.get("/health", response_model=HealthResponse)
async def health_check(model: TitanicModel = Depends(get_model)):
    """Health check endpoint."""
    # Get system metrics
    system_metrics = await get_system_metrics()
    
    # Check if model is loaded
    model_loaded = model.model is not None
    
    # Determine status based on model and system metrics
    status = "healthy" if model_loaded else "degraded"
    error = None
    
    # Check system metrics
    if system_metrics["memory_usage_mb"] > health_config.MEMORY_THRESHOLD_MB:
        status = "degraded"
        error = "Memory usage above threshold"
    
    if system_metrics["cpu_percent"] > health_config.CPU_THRESHOLD_PERCENT:
        status = "degraded"
        error = "CPU usage above threshold"
    
    if system_metrics["predictions_per_minute"] < health_config.MIN_PREDICTIONS_PER_MINUTE:
        status = "degraded"
        error = "Prediction rate below minimum expected"
    
    return HealthResponse(
        status=status,
        version=settings.API_VERSION,
        model_loaded=model_loaded,
        model_version=model.get_version(),
        processing_time_ms=0,  # Not making a prediction
        error=error,
        metrics=SystemMetrics(**system_metrics)
    )
