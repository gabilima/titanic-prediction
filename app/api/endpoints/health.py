import time
import psutil
import os
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.core.config import settings
from app.core.logging import get_logger
from app.api.schemas.prediction import HealthResponse, SystemMetrics
from app.ml.model import get_model, TitanicModel
from app.ml.pipeline import load_pipeline, preprocess_input
from app.core.health import health_config
from app.monitoring.metrics import get_prediction_count
from app.core.model import model

router = APIRouter()
logger = get_logger(__name__)

async def get_system_metrics() -> Dict[str, Any]:
    """Obtém métricas do sistema."""
    process = psutil.Process(os.getpid())
    return {
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "thread_count": process.num_threads(),
        "open_files": len(process.open_files()),
        "predictions_per_minute": model.get_predictions_per_minute()
    }

async def check_model_health() -> Dict[str, Any]:
    """Verifica a saúde do modelo."""
    try:
        start_time = time.time()
        prediction = await model.predict(health_config.TEST_INPUT)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy" if processing_time < health_config.MODEL_HEALTH_THRESHOLD_MS else "degraded",
            "processing_time_ms": processing_time,
            "error": None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "processing_time_ms": 0,
            "error": str(e)
        }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint para verificação de saúde do sistema."""
    # Obtém métricas do sistema
    system_metrics = await get_system_metrics()
    
    # Verifica saúde do modelo
    model_health = await check_model_health()
    
    # Verifica métricas do sistema
    if system_metrics["memory_usage_mb"] > health_config.MEMORY_THRESHOLD_MB:
        model_health["status"] = "degraded"
        model_health["error"] = "Uso de memória acima do limite"
    
    if system_metrics["cpu_percent"] > health_config.CPU_THRESHOLD_PERCENT:
        model_health["status"] = "degraded"
        model_health["error"] = "Uso de CPU acima do limite"
    
    if system_metrics["predictions_per_minute"] < health_config.MIN_PREDICTIONS_PER_MINUTE:
        model_health["status"] = "degraded"
        model_health["error"] = "Taxa de predições abaixo do mínimo esperado"
    
    return HealthResponse(
        status=model_health["status"],
        version=settings.API_VERSION,
        model_loaded=model.model is not None,
        model_version=model.get_version(),
        processing_time_ms=model_health["processing_time_ms"],
        error=model_health["error"],
        metrics=SystemMetrics(**system_metrics)
    )
