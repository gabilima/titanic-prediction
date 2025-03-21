from prometheus_client import Counter, Histogram, Gauge
import psutil
from typing import Dict, Any

# Initialize Prometheus metrics
PREDICTION_COUNTER = Counter(
    'model_predictions_total',
    'Total number of predictions made',
    ['model_version', 'result']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time taken for prediction',
    ['model_version']
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_bytes',
    'Current system memory usage'
)

def get_prediction_count(model_version: str = "latest") -> int:
    """Get the total number of predictions made."""
    return PREDICTION_COUNTER.labels(model_version=model_version, result="success")._value.get()

def increment_prediction_counter(model_version: str = "latest", success: bool = True) -> None:
    """Increment the prediction counter."""
    result = "success" if success else "failure"
    PREDICTION_COUNTER.labels(model_version=model_version, result=result).inc()

def record_prediction_latency(latency: float, model_version: str = "latest") -> None:
    """Record prediction latency."""
    PREDICTION_LATENCY.labels(model_version=model_version).observe(latency)

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics."""
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    
    SYSTEM_MEMORY_USAGE.set(memory.used)
    
    return {
        "memory_usage": memory.percent,
        "cpu_usage": cpu_percent,
        "prediction_rate": get_prediction_count()
    } 