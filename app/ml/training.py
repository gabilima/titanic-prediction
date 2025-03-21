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

def train_model():
    """Train a new model on the Titanic dataset."""
    logger.info("Starting model training")
    
    try:
        # Load and preprocess data
        data_path = settings.DATA_DIR / "raw" / "train.csv"
        df = pd.read_csv(data_path)
        
        # Validate required features
        missing_features = [f for f in settings.REQUIRED_FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Split features and target
        X = df[settings.REQUIRED_FEATURES]
        y = df['Survived']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=settings.MODEL_RANDOM_STATE
        )
        
        # Create and fit preprocessing pipeline
        pipeline = train_preprocessing_pipeline(X_train)
        
        # Transform data
        X_train_transformed = pipeline.transform(X_train)
        X_test_transformed = pipeline.transform(X_test)
        
        # Save pipeline
        save_pipeline(pipeline)
        
        # Initialize and train model
        model = TitanicModel()
        
        # Start MLflow run
        with mlflow.start_run(run_name="titanic_training") as run:
            # Train the model
            model.train(X_train_transformed, y_train)
            
            # Evaluate model
            metrics = model.evaluate(X_test_transformed, y_test)
            logger.info(f"Model evaluation metrics: {metrics}")
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model with preprocessing pipeline
            mlflow.sklearn.log_model(
                sk_model=model.model,
                artifact_path="model",
                registered_model_name=settings.MLFLOW_MODEL_NAME
            )
            
            # Save run ID
            run_id = run.info.run_id
            logger.info(f"Model training completed. Run ID: {run_id}")
            
            # Save model locally as well
            model_path = settings.MODEL_PATH / settings.DEFAULT_MODEL_FILENAME
            joblib.dump(model.model, model_path)
            logger.info(f"Model saved locally to {model_path}")
        
        return model, metrics
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_id = train_model()
    logger.info(f"Training completed successfully. Run ID: {run_id}")
