import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import uuid
from datetime import datetime
import mlflow
import joblib

from app.core.config import settings
from app.core.logging import get_logger, timing_decorator
from app.ml.pipeline import train_preprocessing_pipeline, save_pipeline
from app.ml.model import TitanicModel
from app.ml.mlflow_utils import MLflowManager

# Initialize logger and MLflow manager
logger = get_logger(__name__)
mlflow_manager = MLflowManager()

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase for consistency."""
    return df.rename(columns=str.lower)

@timing_decorator("load_data")
def load_data(data_path: Path) -> pd.DataFrame:
    """Load data from CSV file with explicit dtypes."""
    logger.info(f"Loading data from {data_path}")
    
    # Define dtypes for each column
    dtypes = {
        'PassengerId': 'int64',
        'Survived': 'int64',
        'Pclass': 'int64',
        'Name': 'string',
        'Sex': 'category',
        'Age': 'float64',
        'SibSp': 'int64',
        'Parch': 'int64',
        'Ticket': 'string',
        'Fare': 'float64',
        'Cabin': 'string',
        'Embarked': 'category'
    }
    
    # Load data with specified dtypes
    df = pd.read_csv(data_path, dtype=dtypes)
    
    # Standardize column names to lowercase
    df = standardize_column_names(df)
    return df

@timing_decorator("train_model")
def train_model(
    data: pd.DataFrame = None,
    data_path: Path = None,
    experiment_name: str = settings.MLFLOW_EXPERIMENT_NAME,
    model_path: str = settings.MODEL_PATH
) -> None:
    """Train a new model and track with MLflow.
    
    Args:
        data: Optional pandas DataFrame containing the training data
        data_path: Optional path to the training data CSV file
        experiment_name: Name of the MLflow experiment
        model_path: Path where to save the trained model
    """
    logger.info("Starting model training")
    
    # Load data if not provided
    if data is None:
        if data_path is None:
            data_path = Path(settings.TRAIN_DATA_PATH)
        data = load_data(data_path)
    
    # Split features and target
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train preprocessing pipeline
    pipeline = train_preprocessing_pipeline(X_train)
    
    # Transform data
    X_train_transformed = pipeline.transform(X_train)
    X_val_transformed = pipeline.transform(X_val)
    
    # Initialize and train model
    model = TitanicModel()
    model.train(X_train_transformed, y_train)
    
    # Evaluate model
    train_metrics = model.evaluate(X_train_transformed, y_train)
    val_metrics = model.evaluate(X_val_transformed, y_val)
    
    logger.info(f"Training metrics: {train_metrics}")
    logger.info(f"Validation metrics: {val_metrics}")
    
    # Save pipeline and model
    save_pipeline(pipeline, model_path)
    model.save(model_path)
    
    # Log metrics with MLflow
    mlflow_manager.log_metrics({
        'training_accuracy': train_metrics['accuracy'],
        'validation_accuracy': val_metrics['accuracy'],
        'training_precision': train_metrics['precision'],
        'validation_precision': val_metrics['precision'],
        'training_recall': train_metrics['recall'],
        'validation_recall': val_metrics['recall'],
        'training_f1': train_metrics['f1_score'],
        'validation_f1': val_metrics['f1_score']
    })
    
    logger.info("Model training completed successfully")

if __name__ == "__main__":
    run_id = train_model()
    logger.info(f"Training completed successfully. Run ID: {run_id}")
