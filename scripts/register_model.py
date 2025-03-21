import os
import mlflow
from pathlib import Path
from app.core.config import settings
from app.ml.mlflow_utils import MLflowManager
from app.ml.model import TitanicModel

def register_model():
    """Register the Titanic model with MLflow."""
    # Initialize MLflow manager
    mlflow_manager = MLflowManager()
    
    # Load the local model
    model = TitanicModel()
    model.load()
    
    # Start a new MLflow run
    with mlflow.start_run(run_name="model_registration") as run:
        # Log model parameters
        mlflow.log_params(settings.MODEL_PARAMS)
        
        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model.model,
            artifact_path="model",
            registered_model_name=settings.MLFLOW_MODEL_NAME
        )
        
        # Get the run ID
        run_id = run.info.run_id
        
        # Register the model
        model_version = mlflow.register_model(
            f"runs:/{run_id}/model",
            settings.MLFLOW_MODEL_NAME
        )
        
        # Transition to production
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=settings.MLFLOW_MODEL_NAME,
            version=model_version.version,
            stage="Production"
        )
        
        print(f"Model registered with version {model_version.version} and transitioned to Production")

if __name__ == "__main__":
    register_model() 