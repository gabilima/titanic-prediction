"""
FastAPI Application for Titanic Survival Prediction

This module provides the main FastAPI application with endpoints for prediction
and A/B testing functionality.
"""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import uvicorn
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel
import mlflow

from app.metrics_exporter import MetricsExporter
from app.api.schemas.prediction import PredictionRequest, PredictionResponse, ABTestConfig

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.api.endpoints import predict, health
from app.monitoring.metrics import record_request
from app.ml.ab_testing import ABTest
from app.ml.mlflow_utils import MLflowManager

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Initialize MLflow manager and A/B testing
mlflow_manager = MLflowManager()
ab_test = ABTest(mlflow_manager=mlflow_manager)

# Initialize metrics exporter without predictor (we'll set it during startup)
metrics_exporter = None

async def collect_metrics_periodically():
    """Collect metrics periodically."""
    while True:
        try:
            if metrics_exporter:
                await metrics_exporter.collect_feature_metrics()
            await asyncio.sleep(300)  # Collect every 5 minutes
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")
            await asyncio.sleep(60)  # Retry after 1 minute on error

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for the FastAPI application."""
    # Startup: load model and resources
    logger.info("Starting application, initializing resources")
    try:
        # Initialize MLflow manager
        app.state.mlflow_manager = MLflowManager()
        logger.info("MLflow manager initialized successfully")

        # Load model from MLflow
        try:
            model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}/{settings.MLFLOW_MODEL_STAGE}"
            app.state.model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Model loaded successfully from {model_uri}")
        except Exception as e:
            logger.warning(f"Could not load model from MLflow registry: {str(e)}")
            logger.info("Falling back to local model file")
            from app.core.model import model
            model.model.load()
            app.state.model = model.model
            logger.info("Local model loaded successfully")
        
        # Initialize metrics exporter with model
        global metrics_exporter
        metrics_exporter = MetricsExporter(app.state.model)
        app.state.metrics_exporter = metrics_exporter
        
        # Start metrics collection
        asyncio.create_task(collect_metrics_periodically())
        
        # Set up initial A/B test with production model
        latest_prod_version = app.state.mlflow_manager.get_latest_versions(
            k=1,
            stages=["Production"]
        )
        if latest_prod_version:
            version = latest_prod_version[0]["version"]
            ab_test.setup_test(
                model_weights={version: 1.0},
                description="Initial production deployment"
            )
            logger.info(f"Initialized A/B test with production model version {version}")
        else:
            logger.warning("No production model found for initial A/B test setup")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

    yield
    # Shutdown: clean up resources
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request timing and logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(time.time()))
    request_logger = get_logger("api.request")
    
    start_time = time.time()
    request_logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={"trace_id": request_id}
    )
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    process_time_ms = round(process_time * 1000, 2)
    
    # Record metrics
    record_request(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        latency_seconds=process_time
    )
    
    request_logger.info(
        f"Request completed in {process_time_ms}ms",
        extra={
            "trace_id": request_id,
            "status_code": response.status_code,
            "latency_ms": process_time_ms
        }
    )
    
    response.headers["X-Process-Time"] = str(process_time_ms)
    response.headers["X-Request-ID"] = request_id
    return response

# Include routers
app.include_router(predict.router, prefix=settings.API_V1_STR)
app.include_router(health.router, prefix=settings.API_V1_STR)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint returning basic API info."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.API_VERSION,
        "description": settings.DESCRIPTION,
        "docs_url": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict survival probability for a Titanic passenger.
    
    Args:
        request: Passenger information for prediction
    
    Returns:
        Prediction response including survival probability and model details
    """
    try:
        # Select model version based on A/B test configuration
        model_version = ab_test.select_model_version()
        model = mlflow_manager.load_model(model_version)
        
        # Prepare features for prediction
        features = [
            request.Pclass,
            1 if request.Sex.lower() == "male" else 0,
            request.Age,
            request.SibSp,
            request.Parch,
            request.Fare
        ]
        
        # Make prediction
        prediction = float(model.predict_proba([features])[0][1])
        
        # Log prediction for A/B testing
        ab_test.log_prediction(
            model_version=model_version,
            prediction=prediction,
            metadata={
                "features": features,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return PredictionResponse(
            prediction=prediction,
            prediction_text="Survived" if prediction >= 0.5 else "Did not survive",
            survival_probability=prediction,
            model_version=model_version,
            processing_time_ms=time.time() * 1000 - request.request_time_ms
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/ab-test/configure")
async def configure_ab_test(config: ABTestConfig):
    """
    Configure A/B test with new model weights.
    
    Args:
        config: A/B test configuration including model weights
    """
    try:
        ab_test.setup_test(
            model_weights=config.model_weights,
            description=config.description
        )
        return {"message": "A/B test configuration updated successfully"}
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to configure A/B test: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to configure A/B test: {str(e)}"
        )

@app.get("/ab-test/results")
async def get_ab_test_results():
    """Get current A/B test results."""
    try:
        results = ab_test.get_test_results()
        return {
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get A/B test results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get A/B test results: {str(e)}"
        )

@app.post("/model/promote/{version}/{stage}")
async def promote_model(version: str, stage: str):
    """
    Promote a model version to a new stage.
    
    Args:
        version: Model version to promote
        stage: Target stage (Production/Staging)
    """
    try:
        if stage not in ["Production", "Staging"]:
            raise ValueError("Stage must be either 'Production' or 'Staging'")
        
        mlflow_manager.promote_model(
            version=version,
            stage=stage
        )
        return {
            "message": f"Model version {version} promoted to {stage}",
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to promote model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to promote model: {str(e)}"
        )

# Start the application
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS
    )
