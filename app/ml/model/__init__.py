from typing import Optional
from fastapi import Depends
from app.ml.model.model import TitanicModel
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global model instance
_model: Optional[TitanicModel] = None

def get_model() -> TitanicModel:
    """Get or create the model instance."""
    global _model
    
    if _model is None:
        logger.info("Initializing model...")
        _model = TitanicModel()
        _model.load(settings.MODEL_PATH)
        logger.info("Model initialized successfully")
    
    return _model 