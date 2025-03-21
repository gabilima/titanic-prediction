"""Integration tests for the Titanic prediction service"""

import os
import json
import sys
import shutil
import pytest
import pandas as pd
from pathlib import Path
import mlflow
from fastapi.testclient import TestClient
import tempfile

from app.main import app
from app.core.config import settings
from app.ml.data_validation import (
    DataValidator,
    DatasetMetadata,
    FeatureDefinition,
    FeatureType
)
from app.ml.training import train_model

# Add project root to Python path to import local modules
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Setup test client
client = TestClient(app)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        'Pclass': 1,
        'Sex': 'male',
        'Age': 29.0,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 100.0,
        'Embarked': 'S'  # Added required field
    }

@pytest.fixture
def feature_definitions():
    """Fixture providing feature definitions for testing"""
    return {
        "Pclass": FeatureDefinition(
            name="Pclass",
            feature_type=FeatureType.NUMERIC,
            min_value=1,
            max_value=3
        ),
        "Sex": FeatureDefinition(
            name="Sex",
            feature_type=FeatureType.CATEGORICAL,
            allowed_values=["male", "female"]
        ),
        "Age": FeatureDefinition(
            name="Age",
            feature_type=FeatureType.NUMERIC,
            min_value=0,
            max_value=120
        ),
        "SibSp": FeatureDefinition(
            name="SibSp",
            feature_type=FeatureType.NUMERIC,
            min_value=0
        ),
        "Parch": FeatureDefinition(
            name="Parch",
            feature_type=FeatureType.NUMERIC,
            min_value=0
        ),
        "Fare": FeatureDefinition(
            name="Fare",
            feature_type=FeatureType.NUMERIC,
            min_value=0
        ),
        "Embarked": FeatureDefinition(
            name="Embarked",
            feature_type=FeatureType.CATEGORICAL,
            allowed_values=["C", "Q", "S"]
        )
    }

@pytest.fixture
def registry_path():
    """Provide a temporary path for the feature store registry."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup after tests
    if os.path.exists(temp_path):
        os.unlink(temp_path)

class TestEndToEndPipeline:
    """End-to-end tests for the complete ML pipeline"""
    
    def test_data_validation_and_training(self, feature_definitions):
        """Test data validation followed by model training"""
        # Setup
        metadata = DatasetMetadata(
            name="test_dataset",
            version="1.0",
            feature_definitions=feature_definitions,
            target_column="Survived"
        )
        validator = DataValidator(metadata)
        
        # Load test data
        data_path = Path(__file__).parent / "test_data" / "train_sample.csv"
        data = pd.read_csv(data_path)
        
        # Validate data
        report = validator.validate_dataset(data)
        assert report.invalid_rows == 0, "Data validation failed"
        
        # Train model
        model_path = "test_model_dir"
        experiment_name = "test_experiment"
        
        train_model(
            data=data,
            experiment_name=experiment_name,
            model_path=model_path
        )
        
        # Verify model directory was created
        assert os.path.exists(model_path)
        
        # Cleanup
        try:
            shutil.rmtree(model_path)
        except Exception as e:
            print(f"Warning: Could not remove model directory: {e}")

class TestAPIIntegration:
    """Integration tests for the API endpoints"""
    
    def test_health_check(self):
        """Test the health check endpoint"""
        response = client.get(f"{settings.API_V1_STR}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "metrics" in data
    
    def test_single_prediction(self, sample_data):
        """Test the single prediction endpoint"""
        # Update sample data to use string for Pclass
        sample_data['Pclass'] = str(sample_data['Pclass'])
        
        response = client.post(
            f"{settings.API_V1_STR}/predict",
            json=sample_data
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "prediction_text" in data
        assert "model_version" in data
        assert "processing_time_ms" in data
    
    def test_batch_prediction(self, sample_data):
        """Test the batch prediction endpoint"""
        # Create a batch with two identical passengers
        batch_data = {
            "passengers": [
                {
                    'Pclass': '1',  # Changed from integer to string
                    'Sex': 'male',
                    'Age': 30.0,
                    'SibSp': 0,
                    'Parch': 0,
                    'Fare': 50.0,
                    'Embarked': 'S'
                },
                {
                    'Pclass': '2',  # Changed from integer to string
                    'Sex': 'female',
                    'Age': 25.0,
                    'SibSp': 1,
                    'Parch': 0,
                    'Fare': 30.0,
                    'Embarked': 'C'
                }
            ]
        }
        
        # Log the request payload for debugging
        print(f"Sending batch prediction request with payload: {json.dumps(batch_data, indent=2)}")
        
        response = client.post(
            f"{settings.API_V1_STR}/batch_predict",
            json=batch_data
        )
        
        # Log the response for debugging
        print(f"Received response with status code: {response.status_code}")
        print(f"Response content: {response.content}")
        
        assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}. Response: {response.json()}"
        response_data = response.json()
        assert 'predictions' in response_data
        assert 'batch_processing_time_ms' in response_data
        assert len(response_data['predictions']) == 2
        
        # Verify prediction format for each passenger
        for pred in response_data["predictions"]:
            assert "prediction" in pred
            assert "prediction_text" in pred
            assert "model_version" in pred
            assert "processing_time_ms" in pred
            assert isinstance(pred["prediction"], (int, float))
            assert pred["prediction"] in [0, 1]
            assert isinstance(pred["prediction_text"], str)
            assert pred["prediction_text"] in ["Survived", "Did not survive"]
    
    def test_invalid_input(self):
        """Test API response with invalid input"""
        invalid_data = {
            "Pclass": "invalid",
            "Sex": "unknown",
            "Age": -1
        }
        response = client.post(
            f"{settings.API_V1_STR}/predict",
            json=invalid_data
        )
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

class TestMLflowIntegration:
    """Integration tests for MLflow tracking"""
    
    def test_experiment_tracking(self):
        """Test MLflow experiment tracking"""
        # Setup MLflow
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        experiment_name = "test_tracking"
        
        # Create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Log metrics
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("test_param", 1)
            mlflow.log_metric("test_metric", 0.95)
        
        # Verify experiment was tracked
        experiment = mlflow.get_experiment_by_name(experiment_name)
        assert experiment is not None
        
        # Get latest run
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        assert len(runs) > 0
        latest_run = runs.iloc[0]
        
        # Verify metrics were logged
        assert latest_run["metrics.test_metric"] == 0.95
        assert latest_run["params.test_param"] == "1"

class TestFeatureStore:
    """Test feature store functionality"""
    
    def test_feature_storage_and_retrieval(self, sample_data, registry_path):
        """Test storing and retrieving features from the feature store"""
        from feature_store.feature_store import FeatureStore
        
        # Initialize feature store with registry path
        store = FeatureStore(registry_path=registry_path)
        
        # Store features
        feature_group = "passenger_features"
        passenger_id = "test_passenger"
        
        store.store_features(
            feature_group=feature_group,
            entity_id=passenger_id,
            features=sample_data
        )
        
        # Retrieve features
        retrieved_features = store.get_features(
            feature_group=feature_group,
            entity_id=passenger_id
        )
        
        # Verify features were stored correctly
        assert retrieved_features == sample_data
    
    def test_feature_versioning(self, sample_data, registry_path):
        """Test feature versioning in the feature store"""
        from feature_store.feature_store import FeatureStore
        
        store = FeatureStore(registry_path=registry_path)
        feature_group = "passenger_features"
        passenger_id = "test_passenger"
        
        # Store features with version 1
        store.store_features(
            feature_group=feature_group,
            entity_id=passenger_id,
            features=sample_data,
            version="1.0"
        )
        
        # Modify features and store as version 2
        modified_data = sample_data.copy()
        modified_data["Age"] = 30.0  # Use uppercase "Age" to match the original data
        store.store_features(
            feature_group=feature_group,
            entity_id=passenger_id,
            features=modified_data,
            version="2.0"
        )
        
        # Retrieve specific versions
        v1_features = store.get_features(
            feature_group=feature_group,
            entity_id=passenger_id,
            version="1.0"
        )
        v2_features = store.get_features(
            feature_group=feature_group,
            entity_id=passenger_id,
            version="2.0"
        )
        
        # Verify versioning
        assert v1_features["Age"] == 29.0  # Use uppercase "Age" in assertions
        assert v2_features["Age"] == 30.0 