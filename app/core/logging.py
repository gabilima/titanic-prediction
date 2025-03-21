import json
import logging
import logging.handlers
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps

from pythonjsonlogger import jsonlogger

from app.core.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for structured logging in production environments
    Adds standardized fields for filtering and distributed tracing
    """
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)

        # Add timestamp in ISO format for consistent parsing
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add service metadata
        log_record["service"] = settings.PROJECT_NAME
        log_record["environment"] = settings.ENV
        log_record["level"] = record.levelname

        # Add tracing and model info
        for field in ["trace_id", "model_version", "latency_ms"]:
            if hasattr(record, field):
                log_record[field] = getattr(record, field)


class RequestIdFilter(logging.Filter):
    """
    Filter that adds request ID to log for distributed tracing
    """
    def __init__(self, request_id: str = None):
        super().__init__()
        self.request_id = request_id
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = getattr(record, "trace_id", self.request_id or "-")
        return True


class ModelVersionFilter(logging.Filter):
    """
    Filter that adds model version to log
    """
    def __init__(self, model_version: str = None):
        super().__init__()
        self.model_version = model_version or settings.VERSION

    def filter(self, record: logging.LogRecord) -> bool:
        record.model_version = getattr(record, "model_version", self.model_version)
        return True


class LatencyLogFilter(logging.Filter):
    """
    Filter that adds operation latency to log when start time is provided
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, "start_time"):
            record.latency_ms = round((time.time() - record.start_time)*1000, 2)
            return True


class LoggingContextTimer:
    """
    Context Manager for timing and logging operations

    Example:
        with LoggingContextTimer(logger, "model_inference"):
            result = model.predict(features)
    """
    def __init__(self, logger: logging.Logger, operation_name: str, log_level: int = logging.INFO):
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        elapsed_ms = round(elapsed_time * 1000, 2)
        
        if exc_type is not None:
            # Log error if an exception occurred
            self.logger.error(
                f"Operation '{self.operation_name}' failed after {elapsed_ms}ms",
                extra={
                    "operation": self.operation_name, 
                    "latency_ms": elapsed_ms, 
                    "success": False,
                    "error": str(exc_val)
                }
            )
        else:
            # Log success with timing information
            self.logger.log(
                self.log_level,
                f"Operation '{self.operation_name}' completed in {elapsed_ms}ms",
                extra={
                    "operation": self.operation_name, 
                    "latency_ms": elapsed_ms, 
                    "success": True
                }
            )
        
        return False  # Don't suppress exceptions


def get_console_handler() -> logging.StreamHandler:
    """
    Create a console handler with appropriate formatting based on environment
    """
    console_handler = logging.StreamHandler(sys.stdout)

    if settings.ENV != "production" and settings.LOG_FORMAT == "console":
        # Simple format for development without trace_id
        log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_handler.setFormatter(logging.Formatter(log_format))
    else:
        # JSON format for production and structured logging
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s %(pathname)s %(lineno)d %(funcName)s %(process)d %(thread)d",
            json_ensure_ascii=False
        )
        console_handler.setFormatter(formatter)
    
    return console_handler


def get_file_handler(log_file: Union[str, Path]) -> logging.Handler:
    """
    Create a rotating file handler for persistent logs
    """
    # Create parent directory if needed
    Path(log_file).parent.mkdir(exist_ok=True)

    # Set up rotating handler to prevent huge log files
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024, # 10MB
        backupCount=5
    )

    formatter = CustomJsonFormatter(
        "%(timestamp)s %(level)s %(name)s %(message)s %(pathname)s %(lineno)d %(funcName)s",
        json_ensure_ascii=False
    )
    file_handler.setFormatter(formatter)
    return file_handler


def configure_logging() -> None:
    """
    Configure the logging system for the application
    This should be called once during app startup
    """
    # Create log directory
    log_dir = settings.BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Reset existing handlers if reconfiguring
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Add console handler
    root_logger.addHandler(get_console_handler())

    # Add file handler in production
    if settings.ENV == "production":
        file_path = log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.json"
        root_logger.addHandler(get_file_handler(file_path))
    
    # Apply global filters
    root_logger.addFilter(ModelVersionFilter())
    root_logger.addFilter(LatencyLogFilter())

    # Configure specific module loggers
    for logger_name, level in [
        ("api", settings.LOG_LEVEL),
        ("ml", settings.LOG_LEVEL),
        ("uvicorn.access", "WARNING"),
        ("urllib3.connectionpool", "WARNING")
    ]:
        module_logger = logging.getLogger(logger_name)
        module_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        module_logger.propagate = True
    
    # Log configuration successful
    logging.getLogger(__name__).info(
        f"Logging configured. Environment: {settings.ENV}, Level: {settings.LOG_LEVEL}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger
    """
    return logging.getLogger(name)


def get_request_logger(base_logger: logging.Logger, request_id: str) -> logging.Logger:
    """
    Get a logger that includes the request ID
    """
    logger_adapter = logging.LoggerAdapter(
        base_logger,
        {"trace_id": request_id}
    )
    return logger_adapter


def log_prediction(
    logger: logging.Logger,
    request_id: str,
    input_features: Dict[str, Any],
    prediction: Any,
    latency_ms: float
) -> None:
    """
    Log info about a model prediction with standardized fields

    Args:
        logger: Logger instance
        request_id: ID for request tracing
        input_features: input data
        prediction: prediction result
        latency_ms: total processing time (in ms)
    """
    logger.info(
        f"Prediction completed in {latency_ms}ms",
        extra={
            "trace_id": request_id,
            "latency_ms": latency_ms,
            "feature_keys": list(input_features.keys()),
            "prediction_type": type(prediction).__name__
        }
    )


def timing_decorator(operation_name:str):
    """
    Decorator to time and log function execution
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            with LoggingContextTimer(logger, operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
