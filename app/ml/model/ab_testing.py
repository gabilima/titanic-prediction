"""
A/B Testing Module for Model Deployment

This module provides functionality for A/B testing different model versions.
"""

import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import mlflow

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.mlflow_utils import MLflowManager

logger = get_logger(__name__)

class ABTesting:
    def __init__(self, mlflow_manager: Optional[MLflowManager] = None):
        """Initialize A/B testing configuration."""
        self.mlflow_manager = mlflow_manager or MLflowManager()
        self.control_model = None
        self.treatment_model = None
        self.traffic_split = 0.5
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.total_predictions = 0
        
    def setup_test(self, control_version: str, treatment_version: str, traffic_split: float = 0.5):
        """Set up an A/B test between two model versions."""
        try:
            # Set up MLflow tracking
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
            
            # Log A/B test parameters
            logger.info(f"Setting up A/B test with control={control_version}, treatment={treatment_version}, split={traffic_split}")
            
            # Initialize models
            self.control_model = self.mlflow_manager.load_model(control_version)
            self.treatment_model = self.mlflow_manager.load_model(treatment_version)
            self.traffic_split = traffic_split
            
            # Initialize metrics
            self.model_metrics = {
                control_version: {"total_predictions": 0, "correct_predictions": 0},
                treatment_version: {"total_predictions": 0, "correct_predictions": 0}
            }
            
            # Log test setup to MLflow
            with mlflow.start_run(run_name=f"ab_test_setup_{datetime.now().isoformat()}"):
                mlflow.log_params({
                    "control_version": control_version,
                    "treatment_version": treatment_version,
                    "traffic_split": traffic_split,
                    "setup_timestamp": datetime.now().isoformat()
                })
            
            logger.info("A/B test setup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to set up A/B test: {str(e)}")
            raise
    
    def get_model_for_request(self) -> Any:
        """Get the appropriate model for a request based on the traffic split."""
        if self.control_model is None or self.treatment_model is None:
            logger.warning("A/B test not properly initialized, using control model")
            return self.control_model
        
        # Simple random assignment
        return self.control_model if np.random.random() < self.traffic_split else self.treatment_model
    
    def log_prediction(
        self,
        model_version: str,
        prediction: float,
        ground_truth: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a prediction and optionally its ground truth for A/B test analysis.
        
        Args:
            model_version: Version of the model that made the prediction
            prediction: The model's prediction
            ground_truth: Optional actual outcome for the prediction
            metadata: Optional additional metadata about the prediction
        """
        self.total_predictions += 1
        
        # Update metrics if ground truth is available
        if ground_truth is not None and model_version in self.model_metrics:
            metrics = self.model_metrics[model_version]
            metrics["total_predictions"] += 1
            if np.isclose(prediction, ground_truth):
                metrics["correct_predictions"] += 1
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"ab_test_prediction_{datetime.now().isoformat()}"):
            log_data = {
                "model_version": model_version,
                "prediction": prediction,
                "total_predictions": self.total_predictions
            }
            if ground_truth is not None:
                log_data["ground_truth"] = ground_truth
            if metadata:
                log_data.update(metadata)
            
            mlflow.log_params(log_data)
    
    def get_test_results(self) -> Dict[str, Dict[str, float]]:
        """
        Get current A/B test results.
        
        Returns:
            Dict containing metrics for each model version
        """
        results = {}
        for version, metrics in self.model_metrics.items():
            total = metrics.get("total_predictions", 0)
            correct = metrics.get("correct_predictions", 0)
            results[version] = {
                "total_predictions": total,
                "accuracy": correct / total if total > 0 else 0.0,
                "traffic_percentage": self.traffic_split if version == "control" else (1 - self.traffic_split)
            }
        return results

class ABTest:
    """Manages A/B testing between different model versions."""
    
    def __init__(self, mlflow_manager: Optional[MLflowManager] = None):
        """Initialize A/B testing configuration."""
        self.mlflow_manager = mlflow_manager or MLflowManager()
        self.model_weights: Dict[str, float] = {}
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        self.total_predictions = 0
    
    def setup_test(
        self,
        model_weights: Dict[str, float],
        description: Optional[str] = None
    ) -> None:
        """
        Set up an A/B test with specified model versions and weights.
        
        Args:
            model_weights: Dictionary mapping model versions to their traffic weights
            description: Optional description of the A/B test
        """
        # Validate weights sum to 1
        total_weight = sum(model_weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Model weights must sum to 1.0, got {total_weight}")
        
        self.model_weights = model_weights
        self.model_metrics = {version: {} for version in model_weights}
        self.total_predictions = 0
        
        # Log A/B test setup to MLflow
        with mlflow.start_run(run_name=f"ab_test_{datetime.now().isoformat()}"):
            mlflow.log_params({
                "ab_test_versions": list(model_weights.keys()),
                "ab_test_weights": list(model_weights.values()),
                "description": description or "A/B test of model versions"
            })
    
    def select_model_version(self) -> str:
        """
        Select a model version based on configured weights.
        
        Returns:
            str: Selected model version
        """
        versions, weights = zip(*self.model_weights.items())
        return random.choices(versions, weights=weights)[0]
    
    def log_prediction(
        self,
        model_version: str,
        prediction: float,
        ground_truth: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a prediction and optionally its ground truth for A/B test analysis.
        
        Args:
            model_version: Version of the model that made the prediction
            prediction: The model's prediction
            ground_truth: Optional actual outcome for the prediction
            metadata: Optional additional metadata about the prediction
        """
        self.total_predictions += 1
        
        # Update metrics if ground truth is available
        if ground_truth is not None:
            metrics = self.model_metrics.setdefault(model_version, {
                "correct_predictions": 0,
                "total_predictions": 0,
                "accuracy": 0.0
            })
            
            metrics["total_predictions"] += 1
            if np.isclose(prediction, ground_truth):
                metrics["correct_predictions"] += 1
            metrics["accuracy"] = (
                metrics["correct_predictions"] / metrics["total_predictions"]
            )
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"ab_test_prediction_{datetime.now().isoformat()}"):
            log_data = {
                "model_version": model_version,
                "prediction": prediction,
                "total_predictions": self.total_predictions
            }
            if ground_truth is not None:
                log_data["ground_truth"] = ground_truth
            if metadata:
                log_data.update(metadata)
            
            mlflow.log_params(log_data)
    
    def get_test_results(self) -> Dict[str, Dict[str, float]]:
        """
        Get current A/B test results.
        
        Returns:
            Dict containing metrics for each model version
        """
        results = {}
        for version, metrics in self.model_metrics.items():
            results[version] = {
                "total_predictions": metrics.get("total_predictions", 0),
                "accuracy": metrics.get("accuracy", 0.0),
                "traffic_weight": self.model_weights.get(version, 0.0)
            }
        return results
    
    def update_weights(
        self,
        new_weights: Dict[str, float],
        reason: Optional[str] = None
    ) -> None:
        """
        Update traffic weights for model versions.
        
        Args:
            new_weights: New weight distribution for model versions
            reason: Optional reason for the weight update
        """
        # Validate new weights
        if not np.isclose(sum(new_weights.values()), 1.0):
            raise ValueError("New weights must sum to 1.0")
        
        old_weights = self.model_weights.copy()
        self.model_weights = new_weights
        
        # Log weight update to MLflow
        with mlflow.start_run(run_name=f"ab_test_weight_update_{datetime.now().isoformat()}"):
            mlflow.log_params({
                "old_weights": old_weights,
                "new_weights": new_weights,
                "update_reason": reason or "Manual weight update",
                "update_timestamp": datetime.now().isoformat()
            }) 