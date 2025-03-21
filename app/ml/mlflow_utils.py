"""
MLflow Utilities Module

This module provides utility functions for managing MLflow experiments and runs.
"""

import os
from typing import Dict, List, Optional, Any, Union
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class MLflowManager:
    """MLflow experiment and model registry manager."""
    
    def __init__(self):
        """Initialize MLflow manager."""
        # Set MLflow tracking and registry URIs
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(settings.MLFLOW_TRACKING_URI)
        
        # Get or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    settings.MLFLOW_EXPERIMENT_NAME,
                    artifact_location=settings.MLFLOW_ARTIFACT_LOCATION
                )
            else:
                experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment '{settings.MLFLOW_EXPERIMENT_NAME}' with ID: {experiment_id}")
        except Exception as e:
            logger.error(f"Failed to get/create experiment: {str(e)}")
            raise
        
        # Set experiment
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
        
        # Initialize client
        self.client = mlflow.tracking.MlflowClient()
        
        # Store configuration
        self.tracking_uri = settings.MLFLOW_TRACKING_URI
        self.registry_uri = settings.MLFLOW_TRACKING_URI
        self.experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        self.model_name = settings.MLFLOW_MODEL_NAME
        
        logger.info(f"Initialized MLflow manager with:\n"
                   f"  - Tracking URI: {self.tracking_uri}\n"
                   f"  - Registry URI: {self.registry_uri}\n"
                   f"  - Experiment: {self.experiment_name}\n"
                   f"  - Model: {self.model_name}")
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current MLflow run.
        
        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log metrics to the current MLflow run.
        
        Args:
            metrics: Dictionary of metrics to log
        """
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, artifact_path, registered_model_name=None):
        """Log a model to MLflow and optionally register it."""
        try:
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name
            )
            
            if registered_model_name:
                # Get the latest version
                latest_version = self._get_latest_version(registered_model_name)
                
                # Transition to Production if it's the first version or if it has better metrics
                if latest_version is not None:
                    # Transition the model to Production stage
                    self.client.transition_model_version_stage(
                        name=registered_model_name,
                        version=latest_version.version,
                        stage="Production"
                    )
                    logger.info(f"Transitioned model version {latest_version.version} to Production stage")
        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")
            raise
    
    def _get_latest_version(self, model_name):
        """Get the latest version of a registered model."""
        try:
            versions = self.client.get_latest_versions(model_name)
            if versions:
                return versions[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get latest model version: {str(e)}")
            raise
    
    def load_model(self, version: str) -> Any:
        """
        Load a specific model version.
        
        Args:
            version: Model version to load
        
        Returns:
            The loaded model
        """
        model_uri = f"models:/{self.model_name}/{version}"
        return mlflow.sklearn.load_model(model_uri)
    
    def get_latest_versions(
        self,
        k: int = 5,
        stages: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the k most recent model versions.
        
        Args:
            k: Number of versions to return
            stages: Optional list of stages to filter by (e.g., ["Production", "Staging"])
        
        Returns:
            List of dictionaries containing model version information
        """
        if stages is None:
            stages = ["None", "Production", "Staging", "Archived"]
        
        try:
            versions = []
            for stage in stages:
                try:
                    latest = self.client.get_latest_versions(self.model_name, [stage])
                    versions.extend(latest)
                except Exception as e:
                    logger.warning(f"Could not get versions for stage {stage}: {str(e)}")
            
            # Sort by creation timestamp and take k most recent
            sorted_versions = sorted(
                versions,
                key=lambda x: x.creation_timestamp if hasattr(x, 'creation_timestamp') else 0,
                reverse=True
            )[:k]
            
            return [
                {
                    "version": v.version if hasattr(v, 'version') else "unknown",
                    "stage": v.current_stage if hasattr(v, 'current_stage') else "unknown",
                    "creation_time": v.creation_timestamp if hasattr(v, 'creation_timestamp') else 0,
                    "run_id": v.run_id if hasattr(v, 'run_id') else "unknown"
                }
                for v in sorted_versions
            ]
        except Exception as e:
            logger.warning(f"Could not get model versions: {str(e)}")
            return []
    
    def promote_model(
        self,
        version: str,
        stage: str,
        archive_existing: bool = True
    ) -> None:
        """
        Promote a model version to a new stage.
        
        Args:
            version: Model version to promote
            stage: Target stage (e.g., "Production", "Staging")
            archive_existing: Whether to archive existing models in the target stage
        """
        if archive_existing:
            # Archive existing models in the target stage
            existing = self.client.search_model_versions(
                f"stage = '{stage}'"
            )
            for mv in existing:
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=mv.version,
                    stage="Archived"
                )
        
        # Promote the specified version
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage=stage
        )
        
        logger.info(f"Promoted model version {version} to {stage}") 