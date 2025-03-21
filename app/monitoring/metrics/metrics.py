from prometheus_client import Counter, Histogram, Gauge, Summary
from app.core.config import settings

# Request metrics
REQUEST_COUNT = Counter(
    'titanic_api_requests_total', 
    'Total number of requests received',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'titanic_api_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 25.0, 50.0, 75.0, 100.0, float("inf"))
)

# Model metrics
MODEL_PREDICTION_COUNT = Counter(
    'titanic_model_predictions_total',
    'Total number of model predictions',
    ['model_version', 'prediction']
)

PREDICTION_LATENCY = Histogram(
    'titanic_model_prediction_latency_seconds',
    'Prediction latency in seconds',
    ['model_version'],
    buckets=(0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

# System metrics
SYSTEM_MEMORY_USAGE = Gauge(
    'titanic_system_memory_usage_bytes',
    'Memory usage in bytes'
)

MODEL_LOAD_TIME = Summary(
    'titanic_model_load_time_seconds',
    'Time to load model in seconds',
    ['model_version']
)

FEATURE_PIPELINE_LATENCY = Histogram(
    'titanic_feature_pipeline_latency_seconds',
    'Feature pipeline processing latency in seconds',
    buckets=(0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5)
)

def record_prediction(prediction: bool, latency_seconds: float):
    """Record a prediction event with its latency."""
    if not settings.METRICS_ENABLED:
        return
        
    MODEL_PREDICTION_COUNT.labels(
        model_version=settings.MODEL_VERSION,
        prediction="survived" if prediction else "died"
    ).inc()
    
    PREDICTION_LATENCY.labels(
        model_version=settings.MODEL_VERSION
    ).observe(latency_seconds)

def record_request(method: str, endpoint: str, status_code: int, latency_seconds: float):
    """Record an API request with its latency."""
    if not settings.METRICS_ENABLED:
        return
        
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint
    ).observe(latency_seconds)

def record_model_load_time(load_time_seconds: float):
    """Record time taken to load the model."""
    if not settings.METRICS_ENABLED:
        return
        
    MODEL_LOAD_TIME.labels(
        model_version=settings.MODEL_VERSION
    ).observe(load_time_seconds)

def update_memory_usage(memory_bytes: int):
    """Update the memory usage gauge."""
    if not settings.METRICS_ENABLED:
        return
        
    SYSTEM_MEMORY_USAGE.set(memory_bytes)

def get_prediction_count() -> int:
    """Retorna o número total de predições feitas pelo modelo."""
    if not settings.METRICS_ENABLED:
        return 0
        
    # Soma todas as predições (sobreviventes e não sobreviventes)
    total = 0
    for label in ["survived", "died"]:
        total += MODEL_PREDICTION_COUNT.labels(
            model_version=settings.MODEL_VERSION,
            prediction=label
        )._value.get()
    return total
