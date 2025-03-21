"""
Prediction endpoints for the Titanic survival prediction service.
"""

import uuid
import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.core.logging import get_logger, log_prediction
from app.api.schemas.prediction import (
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse
)
from app.core.model import ModelManager
from app.core.model import TitanicModel
from app.core.dependencies import get_model, get_pipeline

router = APIRouter()
logger = get_logger(__name__)
model_manager = ModelManager()

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model: TitanicModel = Depends(get_model),
    pipeline: Pipeline = Depends(get_pipeline)
) -> PredictionResponse:
    """
    Make a single prediction.
    """
    logger.info("Processing prediction request")
    
    # Transform input data
    X = pipeline.transform(request.model_dump())
    
    # Get prediction probabilities
    probas = model.predict_proba(X)
    logger.info(f"Probabilities shape: {probas.shape}")
    logger.info(f"Probabilities: {probas}")
    
    # Convert probability to binary prediction (1 if probability > 0.5, else 0)
    prediction = int(probas[0, 1] > 0.5)
    prediction_text = "Survived" if prediction == 1 else "Did not survive"
    
    logger.info("Prediction completed")
    return PredictionResponse(
        prediction=prediction,
        prediction_text=prediction_text,
        survival_probability=float(probas[0, 1]),
        model_version="latest",
        processing_time_ms=time.time() * 1000 - request.request_time_ms
    )

@router.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    model: TitanicModel = Depends(get_model),
    pipeline: Pipeline = Depends(get_pipeline)
) -> BatchPredictionResponse:
    """
    Make predictions for a batch of passengers.
    """
    logger.info("Processing batch prediction request")
    
    start_time = time.time()
    predictions = []
    
    for passenger in request.passengers:
        # Transform input data
        X = pipeline.transform(passenger.model_dump())
        
        # Get prediction probabilities
        probas = model.predict_proba(X)
        logger.info(f"Probabilities shape: {probas.shape}")
        logger.info(f"Probabilities: {probas}")
        
        # Convert probability to binary prediction (1 if probability > 0.5, else 0)
        prediction = int(probas[0, 1] > 0.5)
        prediction_text = "Survived" if prediction == 1 else "Did not survive"
        
        predictions.append(
            PredictionResponse(
                prediction=prediction,
                prediction_text=prediction_text,
                survival_probability=float(probas[0, 1]),
                model_version="latest",
                processing_time_ms=time.time() * 1000 - start_time
            )
        )
    
    logger.info("Batch prediction completed")
    return BatchPredictionResponse(
        predictions=predictions,
        batch_processing_time_ms=time.time() * 1000 - start_time
    )
