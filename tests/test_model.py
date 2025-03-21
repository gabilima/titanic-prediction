import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from app.ml.model import TitanicModel, get_model
from app.ml.pipeline import create_preprocessing_pipeline, save_pipeline
from app.core.config import settings
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create test data with correct features
    test_data = pd.DataFrame({
        'Pclass': [1, 2, 3] * 34,  # 102 samples with balanced classes
        'Sex': ['male', 'female'] * 51,  # 102 samples alternating
        'Age': np.random.randint(1, 80, 102),
        'SibSp': np.random.randint(0, 5, 102),
        'Parch': np.random.randint(0, 5, 102),
        'Fare': np.random.uniform(10, 100, 102),
        'Embarked': ['S', 'C', 'Q'] * 34  # 102 samples with balanced ports
    })

    # Create and fit preprocessing pipeline
    pipeline = create_preprocessing_pipeline()
    pipeline.fit(test_data)
    save_pipeline(pipeline)

    # Transform data
    X = pipeline.transform(test_data)
    y = np.random.randint(0, 2, 102)  # Binary target
    
    # Verify feature count
    assert X.shape[1] == 9, f"Expected 9 features but got {X.shape[1]}"
    
    return X, y

def test_model_initialization():
    model = TitanicModel()
    assert isinstance(model.model, RandomForestClassifier)

def test_model_training(sample_data):
    X, y = sample_data
    model = TitanicModel()
    model.train(X, y)
    assert model.model is not None
    assert model.model.n_features_in_ == X.shape[1]  # Verifica se o número de features está correto

def test_model_prediction(sample_data):
    X, y = sample_data
    model = TitanicModel()
    model.train(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == len(X)
    assert all((p >= 0).all() and (p <= 1).all() for p in probs)

def test_model_evaluation(sample_data):
    X, y = sample_data
    model = TitanicModel()
    model.train(X, y)
    metrics = model.evaluate(X, y)
    assert isinstance(metrics, dict)
    expected_metrics = {'accuracy', 'precision', 'recall', 'f1_score'}
    assert all(metric in metrics for metric in expected_metrics)
    assert all(isinstance(metrics[metric], float) for metric in expected_metrics)

def test_model_save_load(sample_data, tmp_path):
    """Test saving and loading a model."""
    X, y = sample_data  # This already has the correct 9 features from the fixture
    model = TitanicModel()
    model.train(X, y)

    # Test save
    save_path = Path(tmp_path) / "test_model.joblib"
    try:
        # First try MLflow save
        saved_path = model.save()
        assert saved_path is not None
    except Exception:
        # Fallback to local file save
        saved_path = model.save(save_path)
        assert saved_path.exists()

    # Test load
    loaded_model = TitanicModel()
    try:
        # First try MLflow load
        loaded_model.load()
    except Exception:
        # Fallback to local file load
        loaded_model.load(save_path)

    # Verify predictions are consistent
    X_test = X[:1]  # Use first sample for prediction test
    original_pred = model.predict(X_test)
    loaded_pred = loaded_model.predict(X_test)
    assert np.array_equal(original_pred, loaded_pred)

    # Verify feature count
    assert loaded_model.model.n_features_in_ == 9, f"Expected 9 features but got {loaded_model.model.n_features_in_}"

def test_get_model(tmp_path):
    """Test getting a trained model."""
    # Create test data with all required features
    test_data = pd.DataFrame({
        'Pclass': [1, 2, 3],  # Will create 2 one-hot encoded features
        'Sex': ['male', 'female', 'male'],  # Will create 1 one-hot encoded feature
        'Age': [30.0, 25.0, 40.0],  # Numerical
        'SibSp': [0, 1, 2],  # Numerical
        'Parch': [0, 0, 1],  # Numerical
        'Fare': [50.0, 30.0, 15.0],  # Numerical
        'Embarked': ['S', 'C', 'Q']  # Will create 2 one-hot encoded features
    })

    # Create and fit preprocessing pipeline
    pipeline = create_preprocessing_pipeline()
    pipeline.fit(test_data)
    save_pipeline(pipeline)

    # Transform data and train model
    X = pipeline.transform(test_data)
    y = np.array([1, 0, 1])  # Dummy target
    
    # Verify feature count
    assert X.shape[1] == 9, f"Expected 9 features but got {X.shape[1]}"
    
    model = TitanicModel()
    model.train(X, y)

    # Save model
    save_path = Path(tmp_path) / settings.DEFAULT_MODEL_FILENAME
    model.save(save_path)

    # Load model and verify feature count
    loaded_model = get_model()
    assert isinstance(loaded_model.model, RandomForestClassifier)
    
    # Test prediction with transformed data
    pred = loaded_model.predict(X[:1])
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert pred[0] in [0, 1]
