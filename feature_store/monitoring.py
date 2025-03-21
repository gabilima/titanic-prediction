import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, Summary
from scipy.stats import ks_2samp

# Configure logging
logger = logging.getLogger(__name__)


# Prometheus metrics
FEATURE_DRIFT_GAUGE = Gauge(
    "feature_drift", "Feature drift score by feature name", ["feature_name"]
)
DATA_QUALITY_GAUGE = Gauge(
    "data_quality_score", "Data quality score by dimension", ["dimension", "feature_name"]
)
FEATURE_STORE_REQUEST_COUNTER = Counter(
    "feature_store_requests_total", "Total feature store requests", ["operation", "status"]
)
FEATURE_STORE_LATENCY = Histogram(
    "feature_store_latency_seconds", "Feature store operation latency", ["operation"]
)
MODEL_PERFORMANCE_GAUGE = Gauge(
    "model_performance", "Model performance metrics", ["metric", "model_version"]
)
HEALTH_CHECK_GAUGE = Gauge(
    "feature_store_health", "Feature store component health status", ["component"]
)
FEATURE_UPDATE_SUMMARY = Summary(
    "feature_update_duration_seconds", "Feature update duration", ["feature_name"]
)


class FeatureMonitoring:
    """
    Comprehensive monitoring for feature store with drift detection, 
    data quality checks, and performance tracking.
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.05,
        enable_alerting: bool = True,
        alert_channels: Optional[List[str]] = None,
        expectations_path: Optional[str] = None
    ):
        """
        Initialize the feature monitoring system.
        
        Args:
            reference_data: Baseline data for drift comparison
            drift_threshold: KS test p-value threshold for drift detection
            enable_alerting: Whether to enable alerting
            alert_channels: List of alert channels (email, slack, etc.)
            expectations_path: Path to Great Expectations suite
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.enable_alerting = enable_alerting
        self.alert_channels = alert_channels or ["email", "logs"]
        
        # Initialize Great Expectations context
        try:
            self.ge_context = gx.get_context()
        except:
            # If no context exists, create a basic one
            self.ge_context = None
            logger.warning("Could not initialize Great Expectations context. Some data quality checks may be limited.")
        
        self.expectations_path = expectations_path
        
        # Feature statistics cache
        self.feature_stats = {}
        
        # Initialize health status
        HEALTH_CHECK_GAUGE.labels(component="drift_detection").set(1)
        HEALTH_CHECK_GAUGE.labels(component="data_quality").set(1)
        HEALTH_CHECK_GAUGE.labels(component="alerting").set(1)
        
        logger.info("Feature monitoring initialized with threshold: %s", drift_threshold)

    def calculate_drift(
        self, current_data: pd.DataFrame, reference_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate drift between current data and reference data using KS test.
        
        Args:
            current_data: Current batch of data
            reference_data: Reference data (uses self.reference_data if None)
            
        Returns:
            Dictionary of feature names to drift scores (p-values)
        """
        start_time = time.time()
        try:
            reference = reference_data if reference_data is not None else self.reference_data
            if reference is None:
                raise ValueError("No reference data provided for drift detection")
            
            # Calculate drift for each numerical feature
            drift_scores = {}
            
            for col in current_data.select_dtypes(include=np.number).columns:
                if col in reference.columns:
                    # Perform Kolmogorov-Smirnov test
                    ks_statistic, p_value = ks_2samp(
                        current_data[col].dropna(), reference[col].dropna()
                    )
                    drift_scores[col] = p_value
                    
                    # Update Prometheus metric
                    FEATURE_DRIFT_GAUGE.labels(feature_name=col).set(p_value)
                    
                    # Alert if drift detected
                    if p_value < self.drift_threshold:
                        message = f"Drift detected for feature {col}: p-value={p_value}"
                        logger.warning(message)
                        if self.enable_alerting:
                            self._send_alert("DRIFT_DETECTED", message)
            
            # Record monitoring latency
            FEATURE_STORE_LATENCY.labels(operation="drift_detection").observe(time.time() - start_time)
            FEATURE_STORE_REQUEST_COUNTER.labels(operation="drift_detection", status="success").inc()
            return drift_scores
            
        except Exception as e:
            FEATURE_STORE_REQUEST_COUNTER.labels(operation="drift_detection", status="error").inc()
            HEALTH_CHECK_GAUGE.labels(component="drift_detection").set(0)
            logger.error("Drift detection failed: %s", str(e))
            raise

    def check_data_quality(self, data: pd.DataFrame) -> float:
        """
        Check data quality using basic validation rules.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Overall data quality score
        """
        try:
            # Define validation rules
            validations = [
                # Age validations
                (data["Age"].notna(), "Age not null"),
                ((data["Age"] >= 0) & (data["Age"] <= 120), "Age between 0 and 120"),
                
                # Sex validations
                (data["Sex"].isin(["male", "female"]), "Sex in [male, female]"),
                
                # Fare validations
                (data["Fare"] >= 0, "Fare >= 0"),
                
                # Pclass validations
                (data["Pclass"].isin([1, 2, 3]), "Pclass in [1, 2, 3]")
            ]
            
            # Run validations
            results = []
            for condition, description in validations:
                success_rate = condition.mean()
                results.append((success_rate, description))
                
                # Update individual metrics
                DATA_QUALITY_GAUGE.labels(
                    dimension="validation",
                    feature_name=description
                ).set(success_rate)
            
            # Calculate overall quality score
            quality_score = np.mean([r[0] for r in results])
            
            # Update overall metric
            DATA_QUALITY_GAUGE.labels(
                dimension="overall",
                feature_name="all"
            ).set(quality_score)
            
            # Log validation results
            for success_rate, description in results:
                logger.info(
                    "Validation '%s': %.2f%% passed",
                    description,
                    success_rate * 100
                )
            
            logger.info(
                "Data quality check completed. Overall score: %.2f",
                quality_score
            )
            
            return quality_score
            
        except Exception as e:
            logger.error("Data quality check failed: %s", str(e))
            HEALTH_CHECK_GAUGE.labels(component="data_quality").set(0)
            raise

    def _send_alert(self, alert_type: str, message: str) -> None:
        """Send alert through configured channels."""
        for channel in self.alert_channels:
            try:
                if channel == "email":
                    # Implement email alerting
                    pass
                elif channel == "slack":
                    # Implement Slack alerting
                    pass
                elif channel == "logs":
                    logger.warning("[ALERT] %s: %s", alert_type, message)
            except Exception as e:
                logger.error("Failed to send alert through %s: %s", channel, str(e))
                HEALTH_CHECK_GAUGE.labels(component="alerting").set(0)

    def get_monitoring_dashboard_url(self) -> str:
        """Get URL for monitoring dashboard."""
        # This should be implemented based on your monitoring setup
        return "http://your-monitoring-dashboard-url"

    def export_monitoring_report(self, output_path: str) -> None:
        """Export monitoring report to file."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "feature_stats": self.feature_stats,
                "health_status": self.check_health(),
                "dashboard_url": self.get_monitoring_dashboard_url()
            }
            
            pd.DataFrame([report]).to_json(output_path)
            logger.info("Monitoring report exported to %s", output_path)
            
        except Exception as e:
            logger.error("Failed to export monitoring report: %s", str(e))
            raise

    def update_feature_statistics(self, data: pd.DataFrame) -> None:
        """
        Update statistical profiles for features.
        
        Args:
            data: DataFrame with features to profile
        """
        for col in data.columns:
            self.feature_stats[col] = {}
            
            # Record basic statistics for numerical features
            if pd.api.types.is_numeric_dtype(data[col]):
                self.feature_stats[col] = {
                    "min": data[col].min(),
                    "max": data[col].max(),
                    "mean": data[col].mean(),
                    "median": data[col].median(),
                    "std": data[col].std(),
                    "last_updated": datetime.now().isoformat()
                }
            
            # Record category information for categorical features
            elif pd.api.types.is_categorical_dtype(data[col]) or pd.api.types.is_object_dtype(data[col]):
                self.feature_stats[col] = {
                    "categories": list(data[col].dropna().unique()),
                    "distribution": dict(data[col].value_counts(normalize=True)),
                    "last_updated": datetime.now().isoformat()
                }
        
        logger.info("Updated feature statistics for %d features", len(data.columns))

    def track_model_performance(
        self, 
        metrics: Dict[str, float], 
        model_version: str
    ) -> None:
        """
        Track model performance metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            model_version: Model version identifier
        """
        for metric_name, value in metrics.items():
            MODEL_PERFORMANCE_GAUGE.labels(metric=metric_name, model_version=model_version).set(value)
            
        logger.info(
            "Tracked model performance metrics for version %s: %s", 
            model_version, 
            ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        )

    def check_health(self) -> Dict[str, bool]:
        """
        Check health status of monitoring components.
        
        Returns:
            Dictionary of component names to health status (True/False)
        """
        health_status = {
            "drift_detection": HEALTH_CHECK_GAUGE.labels(component="drift_detection")._value.get() == 1,
            "data_quality": HEALTH_CHECK_GAUGE.labels(component="data_quality")._value.get() == 1,
            "alerting": HEALTH_CHECK_GAUGE.labels(component="alerting")._value.get() == 1,
        }
        
        # Overall health status
        is_healthy = all(health_status.values())
        logger.info("Health check completed: %s", "HEALTHY" if is_healthy else "UNHEALTHY")
        
        return health_status


class DashboardIntegration:
    """
    Integration with monitoring dashboards (Grafana, custom dashboards, etc.)
    """
    
    def __init__(self, dashboard_url: str = ""):
        self.dashboard_url = dashboard_url
        logger.info("Dashboard integration initialized with URL: %s", dashboard_url or "Not configured")
        
    def get_dashboard_url(self, params: Optional[Dict[str, str]] = None) -> str:
        """
        Get dashboard URL with optional parameters.
        
        Args:
            params: Query parameters to add to dashboard URL
            
        Returns:
            Complete dashboard URL
        """
        if not self.dashboard_url:
            return ""
            
        url = self.dashboard_url
        if params:
            param_strings = [f"{k}={v}" for k, v in params.items()]
            url += "?" + "&".join(param_strings)
        
        return url
        
    def push_custom_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Push custom metrics to dashboard (if not using Prometheus).
        
        Args:
            metrics: Custom metrics to push
            
        Returns:
            Success status
        """
        # Implementation would depend on dashboard system
        # This is a placeholder for custom dashboard integrations
        logger.info("Custom metrics pushed to dashboard: %s", metrics)
        return True


# Example usage in a feature pipeline context
def create_monitoring_context(reference_data_path: Optional[str] = None):
    """
    Create and initialize a monitoring context.
    
    Args:
        reference_data_path: Path to reference data file
        
    Returns:
        Initialized FeatureMonitoring instance
    """
    reference_data = None
    if reference_data_path:
        try:
            reference_data = pd.read_csv(reference_data_path)
            logger.info("Loaded reference data from %s with %d records", 
                       reference_data_path, len(reference_data))
        except Exception as e:
            logger.error("Failed to load reference data: %s", str(e))
    
    return FeatureMonitoring(
        reference_data=reference_data,
        drift_threshold=0.05,
        enable_alerting=True,
        alert_channels=["email", "logs"]
    )

