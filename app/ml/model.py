import joblib
import numpy as np
import pandas as pd
import os
import glob
import mlflow
from pathlib import Path
from typing import Dict, Optional, Any, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.core.config import settings
from app.core.logging import get_logger, timing_decorator, LoggingContextTimer
from app.ml.mlflow_utils import MLflowManager

logger = get_logger(__name__)

class TitanicModel:
    """Titanic survival prediction model."""
    
    def __init__(self):
        """Initialize the Titanic survival prediction model."""
        self.model = RandomForestClassifier(
            n_estimators=settings.MODEL_N_ESTIMATORS,
            max_depth=settings.MODEL_MAX_DEPTH,
            random_state=settings.MODEL_RANDOM_STATE
        )
        self.pipeline = None
        
        # Calculate expected numerical features
        numerical_features = [f for f in settings.NUMERICAL_FEATURES if f in settings.REQUIRED_FEATURES]
        # Ensure Pclass is treated as categorical
        if "Pclass" in numerical_features:
            numerical_features.remove("Pclass")
        self.expected_numerical = len(numerical_features)  # Age, Fare, SibSp, Parch
        logger.info(f"Expected numerical features ({self.expected_numerical}): {numerical_features}")
        
        # Calculate expected categorical features after one-hot encoding
        categorical_features = ['Pclass', 'Sex', 'Embarked']  # Define explicit order
        self.expected_categorical = 0
        
        # Calculate categorical dimensions
        for feature in categorical_features:
            # For each categorical feature, we get n binary columns (since we don't use drop='first')
            categories = settings.CATEGORICAL_VALUES[feature]
            if feature == 'Pclass':  # Pclass has 3 categories -> 2 binary columns
                dims = 2
            elif feature == 'Sex':  # Sex has 2 categories -> 1 binary column
                dims = 1
            elif feature == 'Embarked':  # Embarked has 3 categories -> 2 binary columns
                dims = 2
            self.expected_categorical += dims
            logger.info(f"Adding {dims} dimensions for {feature} (categories: {categories})")
        
        self.expected_features = self.expected_numerical + self.expected_categorical
        logger.info(f"Model initialized with {self.expected_features} expected features "
                   f"({self.expected_numerical} numerical + {self.expected_categorical} categorical)")
        
        self.mlflow_manager = MLflowManager()
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X: Features to train on
            y: Target labels
        """
        # Validate feature dimensionality
        if X.shape[1] != self.expected_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but model is expecting {self.expected_features} features "
                f"({self.expected_numerical} numerical + {self.expected_categorical} categorical). "
                "This indicates a mismatch in the preprocessing pipeline."
            )
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Initialize a new model with settings parameters
        self.model = RandomForestClassifier(
            n_estimators=settings.MODEL_N_ESTIMATORS,
            max_depth=settings.MODEL_MAX_DEPTH,
            random_state=settings.MODEL_RANDOM_STATE
        )
        
        # Train the model
        self.model.fit(X, y)
        
        # Verify feature count was set correctly
        if not hasattr(self.model, 'n_features_in_') or self.model.n_features_in_ != X.shape[1]:
            raise ValueError(
                f"Model training failed to set the correct number of features. "
                f"Expected {X.shape[1]} but got {getattr(self.model, 'n_features_in_', None)}"
            )
        
        logger.info("Model training completed successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Binary predictions (0 or 1)
        """
        # Validate feature dimensionality
        if X.shape[1] != self.expected_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but model is expecting {self.expected_features} features "
                f"({self.expected_numerical} numerical + {self.expected_categorical} categorical). "
                "This indicates a mismatch in the preprocessing pipeline."
            )
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Check if model is trained
        if not hasattr(self.model, 'n_features_in_'):
            raise ValueError("Model has not been trained yet")
        
        # Validate against model's expected features
        if X.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"Model was trained with {self.model.n_features_in_} features but received {X.shape[1]} features. "
                "This indicates a mismatch between training and prediction data."
            )
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Probability predictions [P(y=0), P(y=1)]
        """
        # Validate feature dimensionality
        if X.shape[1] != self.expected_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but model is expecting {self.expected_features} features "
                f"({self.expected_numerical} numerical + {self.expected_categorical} categorical). "
                "This indicates a mismatch in the preprocessing pipeline."
            )
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Check if model is trained
        if not hasattr(self.model, 'n_features_in_'):
            raise ValueError("Model has not been trained yet")
        
        # Validate against model's expected features
        if X.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"Model was trained with {self.model.n_features_in_} features but received {X.shape[1]} features. "
                "This indicates a mismatch between training and prediction data."
            )
        
        # Get probabilities
        probas = self.model.predict_proba(X)
        
        # Log prediction details
        logger.info(f"Made prediction with shape {probas.shape}")
        logger.info(f"Probabilities: {probas}")
        
        return probas
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Validate feature dimensionality
        if X.shape[1] != self.expected_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but model is expecting {self.expected_features} features "
                f"({self.expected_numerical} numerical + {self.expected_categorical} categorical). "
                "This indicates a mismatch in the preprocessing pipeline."
            )
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Check if model is trained
        if not hasattr(self.model, 'n_features_in_'):
            raise ValueError("Model has not been trained yet")
        
        # Validate against model's expected features
        if X.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"Model was trained with {self.model.n_features_in_} features but received {X.shape[1]} features. "
                "This indicates a mismatch between training and prediction data."
            )
        
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
    
    @timing_decorator("save_model")
    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save model to MLflow."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Generate a unique run name
        run_name = f"model_save_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Try to save with MLflow first
            with mlflow.start_run(run_name=run_name):
                self.mlflow_manager.log_model(
                    self.model,
                    "model",
                    registered_model_name="titanic_model"
                )
                logger.info(f"Model saved to MLflow with run name: {run_name}")
            return Path(self.mlflow_manager.tracking_uri) / run_name
        except Exception as e:
            logger.warning(f"Failed to save model to MLflow: {str(e)}")
            logger.info("Falling back to joblib for model saving")
            
            # Save with joblib as fallback
            if filepath is None:
                filepath = settings.MODEL_PATH / settings.DEFAULT_MODEL_FILENAME
            filepath.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, filepath)
            logger.info(f"Model saved to {filepath} using joblib")
            return filepath
    
    @timing_decorator("load_model")
    def load(self, model_version: Optional[str] = None) -> None:
        """Load a trained model from MLflow or local file."""
        logger.info("Loading model")
        
        try:
            # Try loading from MLflow first
            if model_version is None:
                # If no version is specified, try loading from production stage
                model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/Production"
            elif isinstance(model_version, (str, Path)):
                # If a local file path is provided, load from it using joblib
                logger.info(f"Loading model from local file: {model_version}")
                self.model = joblib.load(model_version)
                
                # Calculate expected numerical features
                numerical_features = [f for f in settings.NUMERICAL_FEATURES if f in settings.REQUIRED_FEATURES]
                # Ensure Pclass is treated as categorical
                if "Pclass" in numerical_features:
                    numerical_features.remove("Pclass")
                self.expected_numerical = len(numerical_features)  # Age, Fare, SibSp, Parch
                
                # Calculate expected categorical features after one-hot encoding
                categorical_features = ['Pclass', 'Sex', 'Embarked']  # Define explicit order
                self.expected_categorical = 0
                
                # Calculate categorical dimensions
                for feature in categorical_features:
                    # For each categorical feature, we get n binary columns (since we don't use drop='first')
                    categories = settings.CATEGORICAL_VALUES[feature]
                    if feature == 'Pclass':  # Pclass has 3 categories -> 2 binary columns
                        dims = 2
                    elif feature == 'Sex':  # Sex has 2 categories -> 1 binary column
                        dims = 1
                    elif feature == 'Embarked':  # Embarked has 3 categories -> 2 binary columns
                        dims = 2
                    self.expected_categorical += dims
                
                # Update total expected features
                self.expected_features = self.expected_numerical + self.expected_categorical
                
                # Verify model's feature count matches our expectations
                if not hasattr(self.model, 'n_features_in_'):
                    raise ValueError("Loaded model has no feature count information")
                
                if self.model.n_features_in_ != self.expected_features:
                    raise ValueError(
                        f"Loaded model expects {self.model.n_features_in_} features but our preprocessing "
                        f"pipeline generates {self.expected_features} features "
                        f"({self.expected_numerical} numerical + {self.expected_categorical} categorical). "
                        "This indicates a mismatch between the model and preprocessing pipeline."
                    )
                
                logger.info(f"Model loaded successfully with {self.expected_features} features "
                          f"({self.expected_numerical} numerical + {self.expected_categorical} categorical)")
                return
            else:
                # If a version number is provided, use it
                model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/{model_version}"
            
            try:
                logger.info(f"Attempting to load model from MLflow: {model_uri}")
                mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
                self.model = mlflow.sklearn.load_model(model_uri)
                
                # Calculate expected numerical features
                numerical_features = [f for f in settings.NUMERICAL_FEATURES if f in settings.REQUIRED_FEATURES]
                # Ensure Pclass is treated as categorical
                if "Pclass" in numerical_features:
                    numerical_features.remove("Pclass")
                self.expected_numerical = len(numerical_features)  # Age, Fare, SibSp, Parch
                
                # Calculate expected categorical features after one-hot encoding
                categorical_features = ['Pclass', 'Sex', 'Embarked']  # Define explicit order
                self.expected_categorical = 0
                
                # Calculate categorical dimensions
                for feature in categorical_features:
                    # For each categorical feature, we get n binary columns (since we don't use drop='first')
                    categories = settings.CATEGORICAL_VALUES[feature]
                    if feature == 'Pclass':  # Pclass has 3 categories -> 2 binary columns
                        dims = 2
                    elif feature == 'Sex':  # Sex has 2 categories -> 1 binary column
                        dims = 1
                    elif feature == 'Embarked':  # Embarked has 3 categories -> 2 binary columns
                        dims = 2
                    self.expected_categorical += dims
                
                # Update total expected features
                self.expected_features = self.expected_numerical + self.expected_categorical
                
                # Verify model's feature count matches our expectations
                if not hasattr(self.model, 'n_features_in_'):
                    raise ValueError("Loaded model has no feature count information")
                
                if self.model.n_features_in_ != self.expected_features:
                    raise ValueError(
                        f"Loaded model expects {self.model.n_features_in_} features but our preprocessing "
                        f"pipeline generates {self.expected_features} features "
                        f"({self.expected_numerical} numerical + {self.expected_categorical} categorical). "
                        "This indicates a mismatch between the model and preprocessing pipeline."
                    )
                
                logger.info(f"Model loaded successfully with {self.expected_features} features "
                          f"({self.expected_numerical} numerical + {self.expected_categorical} categorical)")
                
            except Exception as e:
                logger.warning(f"Failed to load model from MLflow: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error during model loading: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

def get_model() -> TitanicModel:
    """Get model instance, loading from disk if model exists."""
    model = TitanicModel()
    try:
        model.load()
    except FileNotFoundError:
        logger.warning("Model file not found, returning uninitialized model")
    return model
