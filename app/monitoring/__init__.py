"""
Monitoring and metrics collection for the Titanic Prediction Service.

This package contains:
- Prometheus metrics exporters
- Performance monitoring utilities
- Health check implementations
- Logging and tracing utilities
"""

from .metrics import (
    get_prediction_count,
    increment_prediction_counter,
    record_prediction_latency,
    get_system_metrics
)

__all__ = [
    'get_prediction_count',
    'increment_prediction_counter',
    'record_prediction_latency',
    'get_system_metrics'
]
