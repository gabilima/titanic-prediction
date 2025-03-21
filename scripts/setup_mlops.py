#!/usr/bin/env python3
"""
MLOps Setup Script

This script sets up all MLOps components:
1. Builds and saves the feature pipeline
2. Registers the model in MLflow
3. Initializes feature monitoring
"""

import os
import sys
import logging
import subprocess
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.models.signature import infer_signature
from app.feature_store.monitoring import FeatureMonitoring
from app.core.config import settings
from scripts.build_pipeline import build_pipeline
from scripts.register_model import register_model

def setup_mlops():
    """Set up all MLOps components."""
    print("Setting up MLOps components...")
    
    # 1. Build and save feature pipeline
    print("\n1. Building feature pipeline...")
    pipeline = build_pipeline()
    print("Feature pipeline built successfully.")
    
    # 2. Register model in MLflow
    print("\n2. Registering model in MLflow...")
    register_model()
    print("Model registered successfully.")
    
    # 3. Initialize feature monitoring
    print("\n3. Setting up feature monitoring...")
    
    # Load reference data
    reference_data_path = settings.DATA_DIR / "raw/train.csv"
    if not reference_data_path.exists():
        raise FileNotFoundError(f"Reference data not found at {reference_data_path}")
    
    reference_data = pd.read_csv(reference_data_path)
    
    # Initialize monitoring
    monitoring = FeatureMonitoring(
        reference_data=reference_data,
        drift_threshold=0.05,
        enable_alerting=True
    )
    
    # Run initial quality check
    quality_score = monitoring.check_data_quality(reference_data)
    print(f"Initial data quality score: {quality_score:.2f}")
    
    # Export initial monitoring report
    report_path = settings.DATA_DIR / "monitoring/initial_report.json"
    os.makedirs(report_path.parent, exist_ok=True)
    monitoring.export_monitoring_report(str(report_path))
    
    print("\nMLOps setup completed successfully!")
    print(f"- Feature pipeline saved to: {settings.MODEL_PATH / settings.FEATURE_PIPELINE_FILENAME}")
    print(f"- Model registered in MLflow at: {settings.MLFLOW_TRACKING_URI}")
    print(f"- Initial monitoring report saved to: {report_path}")

if __name__ == "__main__":
    setup_mlops() 