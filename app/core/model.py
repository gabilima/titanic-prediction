"""
Prediction model manager.

This module handles model loading, prediction, and monitoring.
"""

from typing import Dict, Any, List
import time
from collections import deque
import asyncio
from functools import partial
from app.ml.model import TitanicModel
from app.ml.pipeline.pipeline import load_pipeline, preprocess_input
from app.core.logging import get_logger
from app.core.config import settings
import pandas as pd

logger = get_logger(__name__)

class ModelManager:
    """
    Manages the prediction model lifecycle.
    """
    
    def __init__(self):
        self.model = TitanicModel()
        self.pipeline = None
        self.version = "latest"
        self._prediction_times = deque(maxlen=60)  # Stores timestamps of last 60 predictions
        self._initialized = False
        try:
            self._load_model_and_pipeline()
            self._initialized = True
        except Exception as e:
            logger.warning(f"Model and pipeline not loaded during initialization: {str(e)}")
    
    def _load_model_and_pipeline(self):
        """Loads the model and pipeline."""
        try:
            self.model.load()  # Try to load the model
            self.pipeline = load_pipeline()  # Load the pipeline
            logger.info("Model and pipeline loaded successfully")
        except Exception as e:
            error_msg = f"Error loading model or pipeline: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes a prediction asynchronously.
        
        Args:
            features: Dictionary of input features
            
        Returns:
            Dictionary containing prediction results
            
        Raises:
            RuntimeError: If model or pipeline is not loaded
        """
        # Record prediction timestamp
        start_time = time.time()
        
        try:
            # Ensure model and pipeline are loaded
            if not self._initialized:
                self._load_model_and_pipeline()
                self._initialized = True
            
            # Run prediction in a thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            prediction = await loop.run_in_executor(
                None, 
                partial(self._predict_internal, features)
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Convert prediction to text
            prediction_text = "Survived" if prediction >= 0.5 else "Did not survive"
            
            return {
                "prediction": prediction,
                "prediction_text": prediction_text,
                "processing_time_ms": processing_time,
                "model_version": self.get_version()
            }
            
        except Exception as e:
            logger.error(
                "Prediction error",
                error=str(e),
                features=features
            )
            raise
    
    def _predict_internal(self, features: Dict[str, Any]) -> float:
        """Internal prediction method."""
        try:
            # Ensure model and pipeline are loaded
            if not self._initialized:
                self._load_model_and_pipeline()
                self._initialized = True
            
            # Preprocess features
            X = preprocess_input(self.pipeline, features)
            
            # Reshape if needed - ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Make prediction            probas = self.model.predict_proba(X)
            
            # Log prediction details for debugging
            logger.info(f"Probabilities shape: {probas.shape}")
            logger.info(f"Probabilities: {probas}")
            
            # Handle both binary and single-class cases
            if probas.shape[1] == 1:
                # Single class case - return the probability of that class
                return float(probas[0, 0])
            else:
                # Binary classification case - return probability of survival (class 1)
                return float(probas[0, 1])
            
        except Exception as e:
            logger.error(
                "Prediction error",
                error=str(e),
                features=features
            )
            raise
    
    def get_version(self) -> str:
        """Returns the model version."""
        return self.version
    
    def get_predictions_per_minute(self) -> int:
        """Calculates predictions per minute."""
        now = time.time()
        # Remove predictions older than 1 minute
        while self._prediction_times and now - self._prediction_times[0] > 60:
            self._prediction_times.popleft()
        return len(self._prediction_times)

# Global model manager instance
model = ModelManager()

def get_model() -> TitanicModel:
    """Get the global model instance."""
    return model.model

def get_pipeline() -> dict:
    """Get the global pipeline instance."""
    return model.pipeline

def get_version() -> str:
    """Get the global model version."""
    return model.get_version()

def get_predictions_per_minute() -> int:
    """Get the global predictions per minute."""
    return model.get_predictions_per_minute() 