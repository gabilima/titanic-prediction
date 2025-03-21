import os
import multiprocessing
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from pydantic import AnyHttpUrl, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Project Metadata
    PROJECT_NAME: str = "Titanic Survival Prediction API"
    API_VERSION: str = "v1"
    API_V1_STR: str = f"/api/{API_VERSION}"
    DESCRIPTION: str = "API for predicting the probability of a Titanic's passenger to survive"

    # Base Directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Environment Settings
    ENV: str = os.getenv("ENV", "development") # development, staging or production
    DEBUG: bool = ENV != "production"

    # API Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = DEBUG
    # Auto-detect CPU cores for better concurrency
    WORKERS: int = int(os.getenv("WORKERS", str(max(multiprocessing.cpu_count() - 1, 1))))

    # Model Parameters
    MODEL_N_ESTIMATORS: int = 100
    MODEL_MAX_DEPTH: int = 10
    MODEL_RANDOM_STATE: int = 42
    
    # Resquest Handling Settings
    CONCURRENT_REQUEST_LIMIT: int = int(os.getenv("CONCURRENT_REQUEST_LIMIT", "10"))
    ENABLE_REQUEST_BATCHING: bool = True
    BATCH_TIMEOUT_MS: int = 100 # Wait time to collect batch requests

    # CORS Settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[AnyHttpUrl], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # ML Model Settings
    MODEL_PATH: Path = BASE_DIR / "models"
    DEFAULT_MODEL_FILENAME: str = "titanic_model.joblib"
    FEATURE_PIPELINE_FILENAME: str = "feature_pipeline.joblib"
    VERSION: str = os.getenv("MODEL_VERSION", "latest")
    PREDICTION_THRESHOLD: float = 0.5 # Threshold for binary classification
    MODEL_LOADING_STRATEGY: str = "eager"  # eager or lazy

    # Data Paths
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    # MLflow Settings
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        "file://" + str(Path(__file__).resolve().parent.parent.parent / "mlruns")
    )
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "titanic_survival_prediction")
    MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "titanic_model")
    MLFLOW_MODEL_STAGE: str = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    MLFLOW_REGISTRY_URI: str = os.getenv("MLFLOW_REGISTRY_URI", "")  # Same as tracking URI if empty
    MLFLOW_ARTIFACT_LOCATION: str = os.getenv(
        "MLFLOW_ARTIFACT_LOCATION",
        str(Path(__file__).resolve().parent.parent.parent / "mlartifacts")
    )

    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "json" if ENV == "production" else "console"
    REQUEST_LOG_ENABLED: bool = True

    # Prometheus Metrics Settings
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = int(os.getenv("METRICS_PORT", "8001"))
    COLLECT_MODEL_METRICS: bool = True
    COLLECT_LATENCY_METRICS: bool = True

    # Performance Settings
    BATCH_SIZE: int = 64 # For batch prediction if implemented
    REQUEST_TIMEOUT: int = 60 # In seconds

    # Features Required for Model Prediction
    REQUIRED_FEATURES: List[str] = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]

    # Feature Engineering Settings
    CATEGORICAL_FEATURES: List[str] = ["Sex", "Embarked", "Pclass"]
    NUMERICAL_FEATURES: List[str] = ["Age", "Fare", "SibSp", "Parch"]
    # Ensure these match the REQUIRED_FEATURES and what's actually used in prediction

    # Cache Settings for frequently accessed data or computations
    CACHE_TTL: int = 3600 # Time to Live (In seconds)
    PREDICTION_CACHE_ENABLED: bool = True # Enable caching predictions for repeated inputs

    # Feature Defaults for Missing Values
    FEATURES_DEFAULTS: Dict[str, Union[str, float, int]] = {
        "Age": 29.7,  # median age from training data
        "Fare": 32.2,  # median fare
        "Embarked": "S",  # most common embarked value
        "Pclass": "3",  # most common passenger class as string
        "Sex": "male",  # most common sex value
        "SibSp": 0,  # most common number of siblings/spouses aboard
        "Parch": 0,  # most common number of parents/children aboard
    }
    
    # Categorical Values
    CATEGORICAL_VALUES: Dict[str, List[str]] = {
        "Sex": ["male", "female"],
        "Embarked": ["C", "Q", "S"],
        "Pclass": ["1", "2", "3"]
    }
    MODEL_PARAMS: Dict[str, Union[str, float, int, bool]] = {
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }

    # Health Check Settings
    HEALTH_CHECK_ENABLED: bool = True
    HEALTH_CHECK_PATH: str = "/health"
    MODEL_HEALTH_THRESHOLD_MS: int = 500 # Max acceptable prediction time

    # Model Card Information
    MODEL_CARD: Dict[str, str] = {
        "model_name": "Titanic Survival Classifier",
        "version": "1.0.0",
        "description": "Random Forest Model to predict the survival probability of a Titanic passenger",
        "dataset": "Titanic dataset from Kaggle",
        "author": "Gabriela Lima",
        "created_at": "2025-03-13",
        "performance_metrics": "Accuracy: 0.82, F1-score: 0.79"
    }

    # Model Serving Settings
    MODEL_SERVING_STRATEGY: str = "direct" # direct, queue or batch

    @field_validator("MODEL_PATH", "DATA_DIR", "RAW_DATA_DIR", "PROCESSED_DATA_DIR", mode="after")
    def create_directories(cls, dir: Path) -> Path:
        """
        Function to ensure directories exist
        """
        os.makedirs(dir, exist_ok=True)
        return dir

    model_config = ConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = True,
        protected_namespaces = ()  # Desativa namespaces protegidos para resolver o warning do model_version
    )


# Create a global instance of the Settings
settings = Settings()

# Add cached accessor for dependency injection
@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings to improve performance in request handlers
    """
    return Settings()
    