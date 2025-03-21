"""
Prometheus metrics exporter for Titanic prediction service.

This module implements a Prometheus metrics exporter that collects and exposes
metrics related to the async predictor, including queue sizes, prediction latencies,
and other performance metrics that can be used for monitoring and horizontal pod autoscaling.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Union, Set, Awaitable
from contextlib import asynccontextmanager
from enum import Enum

import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client.exposition import start_http_server
from fastapi import FastAPI, Response
from feature_store.monitoring import FeatureMonitoring

# Set up logging
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Enum representing the supported Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricsExporter:
    """
    Prometheus metrics exporter for the Titanic prediction service.
    
    This class collects metrics from the model and exposes them in Prometheus format,
    allowing for monitoring and horizontal pod autoscaling based on custom metrics.
    
    Attributes:
        _metrics (Dict): Dictionary of registered Prometheus metrics
        _model (Any): Reference to the model instance
        _collection_interval (float): Interval in seconds for collecting metrics
        _is_running (bool): Flag indicating if the background task is running
        _stop_event (asyncio.Event): Event to signal stopping the background task
        _http_server_port (int): Port for the Prometheus HTTP server
    """
    
    # Class-level registry to avoid duplicate metrics
    _registry = prometheus_client.CollectorRegistry()
    
    def __init__(self, model: Any, collection_interval: float = 5.0, http_server_port: int = 8000):
        """
        Initialize the metrics exporter.
        
        Args:
            model: Instance of the model (can be MLflow model or local model)
            collection_interval: Interval in seconds for collecting metrics
            http_server_port: Port for the Prometheus HTTP server
        """
        self._metrics: Dict[str, Any] = {}
        self._model = model
        self._collection_interval = collection_interval
        self._is_running = False
        self._stop_event = asyncio.Event()
        self._http_server_port = http_server_port
        self._task: Optional[asyncio.Task] = None
        
        # Register default metrics
        self._register_default_metrics()
        
        # Feature store monitoring metrics
        self.feature_drift_gauge = Gauge(
            'feature_drift_score',
            'Feature drift score by feature name',
            ['feature_name'],
            registry=self._registry
        )
        
        self.feature_validation_errors = Counter(
            'feature_validation_errors_total',
            'Total number of feature validation errors',
            ['feature_name', 'error_type'],
            registry=self._registry
        )
        
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Overall data quality score',
            registry=self._registry
        )
        
        # Initialize feature monitoring
        self.feature_monitoring = FeatureMonitoring()
    
    def _register_default_metrics(self) -> None:
        """Register the default set of metrics for the predictor."""
        # Queue metrics
        self.register_metric(
            name="prediction_queue_size",
            description="Current size of the prediction queue",
            metric_type=MetricType.GAUGE,
            labels=["priority"]
        )
        
        # Latency metrics
        self.register_metric(
            name="prediction_latency_seconds",
            description="Latency of predictions in seconds",
            metric_type=MetricType.HISTOGRAM,
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )
        
        # Prediction volume metrics
        self.register_metric(
            name="predictions_total",
            description="Total number of predictions",
            metric_type=MetricType.COUNTER,
            labels=["model", "status"]
        )
        
        # Batch processing metrics
        self.register_metric(
            name="batch_size",
            description="Size of prediction batches",
            metric_type=MetricType.HISTOGRAM,
            buckets=(1, 5, 10, 20, 50, 100)
        )
        
        # Resource utilization metrics
        self.register_metric(
            name="resource_utilization",
            description="Resource utilization percentage",
            metric_type=MetricType.GAUGE,
            labels=["resource_type"]  # cpu, memory, gpu
        )
        
        # Error metrics
        self.register_metric(
            name="prediction_errors_total",
            description="Total number of prediction errors",
            metric_type=MetricType.COUNTER,
            labels=["error_type"]
        )
        
        # Cache metrics
        self.register_metric(
            name="cache_hit_ratio",
            description="Ratio of cache hits to total lookups",
            metric_type=MetricType.GAUGE
        )
    
    def register_metric(
        self, 
        name: str, 
        description: str, 
        metric_type: MetricType,
        labels: List[str] = None,
        buckets: tuple = None
    ) -> Any:
        """
        Register a new Prometheus metric.
        
        Args:
            name: Name of the metric
            description: Description of the metric
            metric_type: Type of the metric (counter, gauge, histogram, summary)
            labels: List of label names for the metric
            buckets: Buckets for histogram metrics
            
        Returns:
            The created Prometheus metric object
        """
        if name in self._metrics:
            logger.warning(f"Metric {name} already registered, returning existing metric")
            return self._metrics[name]
        
        labels = labels or []
        
        if metric_type == MetricType.COUNTER:
            metric = Counter(name, description, labels, registry=self._registry)
        elif metric_type == MetricType.GAUGE:
            metric = Gauge(name, description, labels, registry=self._registry)
        elif metric_type == MetricType.HISTOGRAM:
            metric = Histogram(name, description, labels, buckets=buckets, registry=self._registry)
        elif metric_type == MetricType.SUMMARY:
            metric = Summary(name, description, labels, registry=self._registry)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        self._metrics[name] = metric
        return metric
    
    def get_metric(self, name: str) -> Any:
        """
        Get a registered Prometheus metric by name.
        
        Args:
            name: Name of the metric to retrieve
            
        Returns:
            The Prometheus metric object
            
        Raises:
            KeyError: If the metric is not registered
        """
        if name not in self._metrics:
            raise KeyError(f"Metric {name} not registered")
        return self._metrics[name]
    
    async def update_queue_metrics(self) -> None:
        """Update metrics related to prediction queue sizes."""
        # Get queue sizes from the async predictor
        try:
            queue_sizes = await self._predictor.get_queue_sizes()
            for priority, size in queue_sizes.items():
                self._metrics["prediction_queue_size"].labels(priority=priority).set(size)
        except Exception as e:
            logger.error(f"Error updating queue metrics: {e}")
    
    async def update_latency_metrics(self) -> None:
        """Update metrics related to prediction latencies."""
        try:
            latency_stats = await self._predictor.get_latency_stats()
            for model_name, latency in latency_stats.items():
                # Use the observe method for histograms
                self._metrics["prediction_latency_seconds"].observe(latency)
        except Exception as e:
            logger.error(f"Error updating latency metrics: {e}")
    
    async def update_resource_metrics(self) -> None:
        """Update metrics related to resource utilization."""
        try:
            # CPU utilization
            cpu_util = await self._predictor.get_resource_utilization("cpu")
            self._metrics["resource_utilization"].labels(resource_type="cpu").set(cpu_util)
            
            # Memory utilization
            mem_util = await self._predictor.get_resource_utilization("memory")
            self._metrics["resource_utilization"].labels(resource_type="memory").set(mem_util)
            
            # GPU utilization if available
            try:
                gpu_util = await self._predictor.get_resource_utilization("gpu")
                self._metrics["resource_utilization"].labels(resource_type="gpu").set(gpu_util)
            except Exception:
                # GPU might not be available
                pass
        except Exception as e:
            logger.error(f"Error updating resource metrics: {e}")
    
    async def update_cache_metrics(self) -> None:
        """Update metrics related to prediction cache."""
        try:
            cache_stats = await self._predictor.get_cache_stats()
            if "hit_ratio" in cache_stats:
                self._metrics["cache_hit_ratio"].set(cache_stats["hit_ratio"])
        except Exception as e:
            logger.error(f"Error updating cache metrics: {e}")
    
    async def update_all_metrics(self) -> None:
        """Update all registered metrics."""
        await asyncio.gather(
            self.update_queue_metrics(),
            self.update_latency_metrics(),
            self.update_resource_metrics(),
            self.update_cache_metrics()
        )
    
    async def _metrics_collection_task(self) -> None:
        """Background task for collecting metrics at regular intervals."""
        logger.info(f"Starting metrics collection task with interval {self._collection_interval}s")
        while not self._stop_event.is_set():
            try:
                await self.update_all_metrics()
            except Exception as e:
                logger.error(f"Error in metrics collection task: {e}")
            
            try:
                # Wait for the next collection interval or until stopped
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._collection_interval
                )
            except asyncio.TimeoutError:
                # This is expected when the timeout is reached and the stop event wasn't set
                pass
        
        logger.info("Metrics collection task stopped")
    
    def start_collection(self) -> None:
        """Start the background metrics collection task."""
        if self._is_running:
            logger.warning("Metrics collection already running")
            return
        
        self._stop_event.clear()
        self._task = asyncio.create_task(self._metrics_collection_task())
        self._is_running = True
        logger.info("Started metrics collection")
    
    async def stop_collection(self) -> None:
        """Stop the background metrics collection task."""
        if not self._is_running:
            logger.warning("Metrics collection not running")
            return
        
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None
        self._is_running = False
        logger.info("Stopped metrics collection")
    
    def start_prometheus_http_server(self) -> None:
        """Start the Prometheus HTTP server for exposing metrics."""
        try:
            start_http_server(self._http_server_port)
            logger.info(f"Started Prometheus HTTP server on port {self._http_server_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus HTTP server: {e}")
            raise
    
    async def collect_metrics_on_demand(self) -> None:
        """Collect metrics on demand (for one-off requests)."""
        await self.update_all_metrics()
    
    def get_metrics_as_text(self) -> str:
        """
        Get all metrics in the Prometheus text format.
        
        Returns:
            String containing all metrics in Prometheus format
        """
        return prometheus_client.generate_latest(self._registry).decode("utf-8")
    
    async def collect_feature_metrics(self):
        """Collect and update feature-related metrics."""
        try:
            # Get current feature data from the model
            current_data = self._model.get_feature_data() if hasattr(self._model, 'get_feature_data') else None
            
            if current_data is None:
                logger.warning("No current feature data available for monitoring")
                return
            
            # Check data quality
            quality_score = self.feature_monitoring.check_data_quality(current_data)
            self.data_quality_score.set(quality_score)
            
            # Calculate feature drift if reference data is available
            if hasattr(self.feature_monitoring, 'reference_data') and self.feature_monitoring.reference_data is not None:
                drift_scores = self.feature_monitoring.calculate_drift(current_data)
                for feature, score in drift_scores.items():
                    self.feature_drift_gauge.labels(feature_name=feature).set(score)
            else:
                logger.info("No reference data available for drift calculation")
                # Store current data as reference for future comparisons
                self.feature_monitoring.reference_data = current_data.copy()
                
        except Exception as e:
            logger.error(f"Error collecting feature metrics: {str(e)}")
            # Don't re-raise the exception to avoid breaking the metrics collection loop


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """
    Lifespan manager for the FastAPI application.
    
    This context manager handles startup and shutdown logic for the metrics exporter.
    
    Args:
        app: The FastAPI application instance
    """
    # Access predictor and create metrics exporter
    # This assumes that the async_predictor would be available in the app state
    # You may need to adjust this based on your application structure
    async_predictor = app.state.async_predictor
    metrics_exporter = MetricsExporter(async_predictor)
    app.state.metrics_exporter = metrics_exporter
    
    # Start collection on startup
    metrics_exporter.start_collection()
    metrics_exporter.start_prometheus_http_server()
    
    yield  # FastAPI operates while the context is active
    
    # Stop collection on shutdown
    await metrics_exporter.stop_collection()


def create_metrics_app() -> FastAPI:
    """
    Create a FastAPI application for metrics endpoints.
    
    Returns:
        FastAPI: The configured FastAPI application
    """
    app = FastAPI(
        title="Titanic Prediction Metrics API",
        description="API for exposing Titanic prediction service metrics",
        version="1.0.0",
        lifespan=lifespan
    )
    
    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        """
        Endpoint for exposing Prometheus metrics.
        
        Returns:
            Response containing metrics in Prometheus format
        """
        metrics_exporter = app.state.metrics_exporter
        await metrics_exporter.collect_metrics_on_demand()
        return Response(
            content=metrics_exporter.get_metrics_as_text(),
            media_type="text/plain"
        )
    
    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """
        Health check endpoint.
        
        Returns:
            Dict containing status information
        """
        return {"status": "healthy"}
    
    @app.get("/metrics/queue-size")
    async def queue_size_metric() -> Dict[str, int]:
        """
        Custom endpoint for HPA to get current queue sizes.
        
        Returns:
            Dict containing queue sizes by priority
        """
        predictor = app.state.async_predictor
        return await predictor.get_queue_sizes()
    
    @app.get("/metrics/latency")
    async def latency_metric() -> Dict[str, float]:
        """
        Custom endpoint for getting current prediction latencies.
        
        Returns:
            Dict containing latency statistics
        """
        predictor = app.state.async_predictor
        return await predictor.get_latency_stats()
    
    @app.get("/metrics/resource-utilization")
    async def resource_utilization_metric() -> Dict[str, float]:
        """
        Custom endpoint for getting current resource utilization.
        
        Returns:
            Dict containing resource utilization percentages
        """
        predictor = app.state.async_predictor
        cpu_util = await predictor.get_resource_utilization("cpu")
        mem_util = await predictor.get_resource_utilization("memory")
        return {
            "cpu": cpu_util,
            "memory": mem_util
        }
    
    return app


# Helper function to register metrics for a model
def register_model_metrics(exporter: MetricsExporter, model_name: str) -> None:
    """
    Register metrics specific to a particular model.
    
    Args:
        exporter: The metrics exporter instance
        model_name: Name of the model to register metrics for
    """
    # Model-specific prediction counter
    exporter.register_metric(
        name=f"{model_name}_predictions_total",
        description=f"Total number of predictions for {model_name}",
        metric_type=MetricType.COUNTER
    )
    
    # Model-specific latency histogram
    exporter.register_metric(
        name=f"{model_name}_prediction_latency_seconds",
        description=f"Latency of predictions in seconds for {model_name}",
        metric_type=MetricType.HISTOGRAM,
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
    )
    
    # Model-specific accuracy metric
    exporter.register_metric(
        name=f"{model_name}_accuracy",
        description=f"Prediction accuracy for {model_name}",
        metric_type=MetricType.GAUGE
    )
    
    # Model-specific feature drift metrics
    exporter.register_metric(
        name=f"{model_name}_feature_drift",
        description=f"Feature drift score for {model_name}",
        metric_type=MetricType.GAUGE,
        labels=["feature_name"]
    )
    
    # Model-specific error rate
    exporter.register_metric(
        name=f"{model_name}_error_rate",
        description=f"Error rate for {model_name}",
        metric_type=MetricType.GAUGE
    )
    
    # Model-specific prediction throughput
    exporter.register_metric(
        name=f"{model_name}_throughput",
        description=f"Predictions per second for {model_name}",
        metric_type=MetricType.GAUGE
    )
    
    # Performance decay detection metric
    exporter.register_metric(
        name=f"{model_name}_performance_decay",
        description=f"Performance decay score for {model_name} (higher means more decay)",
        metric_type=MetricType.GAUGE
    )
    
    logger.info(f"Registered metrics for model: {model_name}")
