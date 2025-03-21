from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import time

class PredictionRequest(BaseModel):
    """Schema for Titanic prediction request."""
    Pclass: str = Field(..., pattern="^[123]$")
    Sex: str = Field(..., pattern="^(male|female)$")
    Age: float = Field(..., ge=0, le=120)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: str = Field(..., pattern="^[CQS]$")
    request_time_ms: float = Field(default_factory=lambda: time.time() * 1000)

    @field_validator('Sex')
    def validate_sex(cls, v: str) -> str:
        v = v.lower()
        if v not in ('male', 'female'):
            raise ValueError("sex must be 'male' or 'female'")
        return v
    
    @field_validator('Embarked')
    def validate_embarked(cls, v: str) -> str:
        if v not in ('C', 'Q', 'S'):
            raise ValueError("embarked must be 'C', 'Q', or 'S'")
        return v.upper()
    
    @field_validator('Pclass')
    def validate_pclass(cls, v: str) -> str:
        if v not in ('1', '2', '3'):
            raise ValueError("pclass must be '1', '2', or '3'")
        return v

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""
    passengers: List[PredictionRequest]

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    prediction: float
    prediction_text: str
    survival_probability: float = Field(..., description="Probability of survival (between 0 and 1)")
    model_version: str
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response."""
    predictions: List[PredictionResponse]
    batch_processing_time_ms: float

class SystemMetrics(BaseModel):
    """Schema for system metrics."""
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    thread_count: int = Field(..., description="Number of threads")
    open_files: int = Field(..., description="Number of open files")
    predictions_per_minute: int = Field(..., description="Predictions rate per minute")

class ABTestConfig(BaseModel):
    """Configuration model for A/B testing."""
    model_weights: Dict[str, float] = Field(..., description="Dictionary mapping model versions to their weights")
    description: Optional[str] = Field(None, description="Optional description of the A/B test configuration")

class HealthResponse(BaseModel):
    """Schema for health check response."""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(..., description="Health status of the service")
    version: str = Field(..., description="Version of the service")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready for predictions")
    model_version: Optional[str] = Field(None, description="Version of the loaded model, if any")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if not healthy")
    metrics: SystemMetrics = Field(..., description="System metrics")
