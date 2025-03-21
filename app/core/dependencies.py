"""FastAPI dependencies."""
from typing import Optional

from app.core.model import TitanicModel
from app.ml.pipeline import load_pipeline

# Global instances
_model: Optional[TitanicModel] = None
_pipeline = None

def get_model() -> TitanicModel:
    """Get or create model instance."""
    global _model
    if _model is None:
        _model = TitanicModel()
        _model.load()  # Load from MLflow or local file
    return _model

def get_pipeline():
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = load_pipeline()
    return _pipeline 