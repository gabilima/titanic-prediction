import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.core.config import settings
from app.ml.model import TitanicModel
from app.ml.pipeline import create_preprocessing_pipeline, save_pipeline
from app.core.model import ModelManager

@pytest.fixture
def setup_test_model():
    # Create test data with correct features
    test_data = pd.DataFrame({
        'Pclass': [1, 2, 3] * 34,
        'Sex': ['male', 'female'] * 51,
        'Age': np.random.randint(1, 80, 102),
        'SibSp': np.random.randint(0, 5, 102),
        'Parch': np.random.randint(0, 5, 102),
        'Fare': np.random.uniform(10, 100, 102),
        'Embarked': ['S', 'C', 'Q'] * 34
    })
    
    # Create and fit pipeline
    pipeline = create_preprocessing_pipeline()
    pipeline.fit(test_data)
    
    # Save pipeline with explicit path
    pipeline_path = settings.MODEL_PATH / settings.FEATURE_PIPELINE_FILENAME
    print(f"Saving pipeline to {pipeline_path}")
    save_pipeline(pipeline, pipeline_path)
    
    # Verify pipeline was saved
    if not pipeline_path.exists():
        raise RuntimeError(f"Failed to save pipeline to {pipeline_path}")
    
    # Transform data
    X = pipeline.transform(test_data)
    y = np.random.randint(0, 2, len(test_data))
    
    # Train and save model
    model = TitanicModel()
    model.train(X, y)
    model_path = settings.MODEL_PATH / settings.DEFAULT_MODEL_FILENAME
    print(f"Saving model to {model_path}")
    model.save(model_path)
    
    # Verify model was saved
    if not model_path.exists():
        raise RuntimeError(f"Failed to save model to {model_path}")
    
    # Reinitialize ModelManager to ensure it loads the new model
    app.state.model = ModelManager()
    app.state.model._load_model_and_pipeline()  # Explicitly load model and pipeline
    
    # Verify pipeline was loaded
    if app.state.model.pipeline is None:
        raise RuntimeError("Failed to load pipeline in ModelManager")
    
    yield

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client, setup_test_model):
    response = client.get(f"{settings.API_V1_STR}/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_prediction_endpoint(client, setup_test_model):
    test_input = {
        'Pclass': '1',
        'Sex': 'male',
        'Age': 30.0,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 50.0,
        'Embarked': 'S'
    }
    
    # Log the request payload for debugging
    print(f"Sending prediction request with payload: {test_input}")
    
    response = client.post(f"{settings.API_V1_STR}/predict", json=test_input)
    
    # Log the response for debugging
    print(f"Received response with status code: {response.status_code}")
    print(f"Response content: {response.content}")
    
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}. Response: {response.json()}"
    data = response.json()
    assert "prediction" in data
    assert "prediction_text" in data
    assert "model_version" in data
    assert "processing_time_ms" in data
    
    # Verify prediction format
    assert isinstance(data["prediction"], (int, float))
    assert data["prediction"] in [0, 1]
    assert isinstance(data["prediction_text"], str)
    assert data["prediction_text"] in ["Survived", "Did not survive"]
