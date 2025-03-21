import pytest
import pandas as pd
import numpy as np
from app.ml.pipeline import create_preprocessing_pipeline, train_preprocessing_pipeline
from app.core.config import settings

@pytest.fixture
def sample_data():
    # Create sample data with expected features
    data = pd.DataFrame({
        'Pclass': [1, 2, 3],
        'Sex': ['male', 'female', 'male'],
        'Age': [22, 35, np.nan],
        'SibSp': [1, 0, 2],
        'Parch': [0, 0, 1],
        'Fare': [7.25, 71.2833, 8.05],
        'Embarked': ['S', 'C', 'Q'],
        'PassengerId': [1, 2, 3],
        'Survived': [0, 1, 0]
    })
    return data

def test_create_preprocessing_pipeline():
    pipeline = create_preprocessing_pipeline()
    assert pipeline is not None
    assert len(pipeline.steps) > 0

def test_train_preprocessing_pipeline(sample_data):
    # Prepare input data
    X = sample_data.drop(['Survived', 'PassengerId'], axis=1)
    
    # Train pipeline
    pipeline = train_preprocessing_pipeline(X)
    assert pipeline is not None
    
    # Transform data
    transformed_data = pipeline.transform(X)
    assert transformed_data is not None
    assert isinstance(transformed_data, np.ndarray)
    
    # Check output shape (should match expected feature count after one-hot encoding)
    expected_features = len(settings.CATEGORICAL_FEATURES) + len(settings.NUMERICAL_FEATURES)
    assert transformed_data.shape[1] >= expected_features

def test_pipeline_handles_missing_values(sample_data):
    # Prepare input data with missing values
    X = sample_data.drop(['Survived', 'PassengerId'], axis=1)
    
    # Train and transform
    pipeline = train_preprocessing_pipeline(X)
    transformed_data = pipeline.transform(X)
    
    # Check that there are no missing values in the output
    assert not np.isnan(transformed_data).any()

def test_pipeline_categorical_encoding(sample_data):
    X = sample_data.drop(['Survived', 'PassengerId'], axis=1)
    pipeline = train_preprocessing_pipeline(X)
    transformed_data = pipeline.transform(X)
    
    # Values should be numeric after transformation
    assert np.issubdtype(transformed_data.dtype, np.number)

import pandas as pd
from app.ml.pipeline import create_preprocessing_pipeline, train_preprocessing_pipeline

# Load sample data
data = pd.read_csv("data/raw/train.csv")
X = data.drop(['Survived', 'PassengerId'], axis=1, errors='ignore')

# Test pipeline creation
pipeline = create_preprocessing_pipeline()
assert pipeline is not None

# Test pipeline training
trained_pipeline = train_preprocessing_pipeline(X)
transformed_data = trained_pipeline.transform(X)
# Verify output shape matches expectations
