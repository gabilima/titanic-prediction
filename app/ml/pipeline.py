import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, cast

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from app.core.config import settings
from app.core.logging import get_logger, LoggingContextTimer, timing_decorator

# Initialize logger
logger = get_logger(__name__)


def _safe_to_frame(data: Union[Dict[str, Any], np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """Safely convert input data to DataFrame with explicit dtypes."""
    if isinstance(data, pd.DataFrame):
        return data
    
    # Convert PredictionRequest or any Pydantic model to dict
    if hasattr(data, 'model_dump'):
        data = data.model_dump()
    elif hasattr(data, 'dict'):  # Fallback for older Pydantic versions
        data = data.dict()
    
    # Convert single dict to list of dicts
    if isinstance(data, dict):
        data = [data]
        
    # Create DataFrame with explicit index
    df = pd.DataFrame(data, index=range(len(data)))
    
    # Convert object columns to string, but preserve numeric types
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric first
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # If not numeric, convert to string
                df[col] = df[col].astype(str)
    
    logger.info(f"Converted input data to DataFrame with shape {df.shape}")
    return df

class DefaultValueImputer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to impute missing values using predefined defaults from settings.
    """
    
    def __init__(self):
        self.defaults = settings.FEATURES_DEFAULTS
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        for feature, default_value in self.defaults.items():
            if feature in settings.REQUIRED_FEATURES and feature in X_copy.columns and X_copy[feature].isnull().any():
                logger.info(f"Imputing {X_copy[feature].isnull().sum()} missing values for {feature} with {default_value}")
                X_copy[feature] = X_copy[feature].fillna(default_value)
        return X_copy


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to select and validate required features.
    """
    
    def __init__(self):
        self.feature_names = settings.REQUIRED_FEATURES
        self.feature_names_lower = [f.lower() for f in settings.REQUIRED_FEATURES]
        self.feature_map = dict(zip(self.feature_names_lower, self.feature_names))
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert to DataFrame using the safe converter
        X = _safe_to_frame(X)
        
        # Create a mapping of lowercase column names to actual column names
        column_map = {str(col).lower(): col for col in X.columns}
        
        # Create a new DataFrame with correctly cased column names
        new_data = {}
        missing_features = []
        
        for feat_lower, feat in self.feature_map.items():
            if feat_lower in column_map:
                new_data[feat] = X[column_map[feat_lower]]
            elif feat in settings.FEATURES_DEFAULTS:
                new_data[feat] = settings.FEATURES_DEFAULTS[feat]
                logger.warning(f"Adding missing feature {feat} with default value {settings.FEATURES_DEFAULTS[feat]}")
            else:
                missing_features.append(feat)
        
        if missing_features:
            error_msg = f"Missing required features: {missing_features}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Return DataFrame with required features in the correct order
        return pd.DataFrame(new_data)[self.feature_names]


def convert_to_string(X):
    """Convert input to string type."""
    return X.astype(str)

@timing_decorator("create_preprocessing_pipeline")
def create_preprocessing_pipeline() -> Pipeline:
    """
    Create the feature preprocessing pipeline for the Titanic dataset.
    
    Returns:
        Pipeline: A scikit-learn Pipeline for data preprocessing
    """
    logger.info("Creating preprocessing pipeline")
    
    # Get feature groups from settings
    numerical_features = [f for f in settings.NUMERICAL_FEATURES if f in settings.REQUIRED_FEATURES]
    # Ensure Pclass is treated as categorical
    if "Pclass" in numerical_features:
        numerical_features.remove("Pclass")
    logger.info(f"Numerical features: {numerical_features}")
    
    # Define categorical features in consistent order
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    logger.info(f"Categorical features: {categorical_features}")
    
    # Numerical features preprocessing
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Create transformers for each categorical feature
    categorical_transformers = []
    for feature in categorical_features:
        transformer = Pipeline(steps=[
            ('to_string', FunctionTransformer(convert_to_string)),
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(
                categories=[settings.CATEGORICAL_VALUES[feature]], 
                drop='first', 
                sparse=False,
                handle_unknown='ignore'  # Add this to handle unknown categories
            ))
        ])
        categorical_transformers.append((f"{feature.lower()}", transformer, [feature]))
        logger.info(f"Created transformer for {feature} with categories: {settings.CATEGORICAL_VALUES[feature]}")
    
    # Combine transformers
    transformers = [('num', numerical_transformer, numerical_features)]
    transformers.extend(categorical_transformers)
    
    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop columns not explicitly listed
        verbose_feature_names_out=False  # Don't prefix feature names
    )
    
    # Create the full preprocessing pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('selector', FeatureSelector()),
        ('default_imputer', DefaultValueImputer()),
        ('column_transformer', column_transformer)
    ])
    
    # Calculate expected features
    expected_numerical = numerical_features
    expected_categorical = []
    
    # Add categorical features in consistent order
    for feature in categorical_features:
        # Skip first category since drop='first'
        categories = settings.CATEGORICAL_VALUES[feature][1:]
        feature_names = [f"{feature}_{cat}" for cat in categories]
        expected_categorical.extend(feature_names)
    
    total_features = len(expected_numerical) + len(expected_categorical)
    logger.info(f"Preprocessing pipeline will generate {total_features} features:")
    logger.info(f"- Numerical features ({len(expected_numerical)}): {expected_numerical}")
    logger.info(f"- Categorical features ({len(expected_categorical)}): {expected_categorical}")
    
    return preprocessing_pipeline


@timing_decorator("save_pipeline")
def save_pipeline(pipeline: Pipeline, filepath: Optional[Path] = None) -> Path:
    """
    Save the preprocessing pipeline to disk.
    
    Args:
        pipeline: The pipeline to save
        filepath: Optional path to save the pipeline. If not provided, uses default path.
        
    Returns:
        Path: The path where the pipeline was saved
        
    Raises:
        RuntimeError: If error occurs during saving
    """
    try:
        if filepath is None:
            filepath = settings.MODEL_PATH / settings.FEATURE_PIPELINE_FILENAME

        with LoggingContextTimer(logger, "pipeline_saving"):
            joblib.dump(pipeline, filepath)
            logger.info(f"Pipeline saved to {filepath}")
            return filepath

    except Exception as e:
        error_msg = f"Error saving pipeline: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def load_pipeline() -> Pipeline:
    """
    Load the preprocessing pipeline from disk.
    
    Returns:
        Pipeline: The loaded preprocessing pipeline
        
    Raises:
        FileNotFoundError: If pipeline file not found
        RuntimeError: If error occurs during loading
    """
    try:
        pipeline_path = settings.MODEL_PATH / settings.FEATURE_PIPELINE_FILENAME
        if not pipeline_path.exists():
            error_msg = f"Pipeline file not found at {pipeline_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        with LoggingContextTimer(logger, "pipeline_loading"):
            pipeline = joblib.load(pipeline_path)
            logger.info(f"Pipeline loaded successfully from {pipeline_path}")
            return pipeline
            
    except Exception as e:
        error_msg = f"Error loading pipeline: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def preprocess_input(pipeline: Pipeline, input_data: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess input data using the pipeline.
    
    Args:
        pipeline: The preprocessing pipeline
        input_data: Input data as dictionary
        
    Returns:
        np.ndarray: Preprocessed input data as a 2D array
    """
    try:
        # Convert input data to DataFrame using _safe_to_frame
        X = _safe_to_frame(input_data)
        logger.info(f"Input data shape: {X.shape}")
        logger.info(f"Input data columns: {X.columns.tolist()}")
        logger.info(f"Input data types:\n{X.dtypes}")
        
        # Get the column transformer
        column_transformer = pipeline.named_steps['column_transformer']
        
        # Get numerical and categorical features
        numerical_features = []
        categorical_features = []
        
        for name, transformer, features in column_transformer.transformers_:
            if name == 'num':
                numerical_features = features
                logger.info(f"Numerical features to transform: {features}")
            elif name in ['pclass', 'sex', 'embarked']:
                categorical_features.append(features[0])
                if hasattr(transformer, 'named_steps') and 'encoder' in transformer.named_steps:
                    encoder = transformer.named_steps['encoder']
                    logger.info(f"Categories for {features[0]}: {encoder.categories_[0]}")
        
        logger.info(f"Categorical features to transform: {categorical_features}")
        
        # Transform the data step by step
        logger.info("Starting transformation pipeline...")
        
        # 1. Feature selection
        X_selected = pipeline.named_steps['selector'].transform(X)
        logger.info(f"After feature selection - shape: {X_selected.shape}, columns: {X_selected.columns.tolist()}")
        
        # 2. Default imputation
        X_imputed = pipeline.named_steps['default_imputer'].transform(X_selected)
        logger.info(f"After default imputation - shape: {X_imputed.shape}")
        
        # 3. Column transformation
        X_transformed = pipeline.named_steps['column_transformer'].transform(X_imputed)
        logger.info(f"After column transformation - shape: {X_transformed.shape}")
        
        # Get and log feature names
        feature_names = get_feature_names(pipeline)
        logger.info(f"Final feature names ({len(feature_names)}): {feature_names}")
        
        # Verify and adjust feature dimensionality
        X_verified = verify_feature_dimensionality(X_transformed)
        
        # Ensure output is 2D array
        if len(X_verified.shape) == 1:
            X_verified = X_verified.reshape(1, -1)
            logger.info(f"Reshaped output to 2D array with shape: {X_verified.shape}")
        
        if isinstance(X_verified, np.ndarray):
            logger.info(f"Final output is numpy array with shape: {X_verified.shape}")
        else:
            logger.info(f"Final output type: {type(X_verified)}")
        
        return X_verified
        
    except Exception as e:
        error_msg = f"Error preprocessing input data: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


@timing_decorator("train_preprocessing_pipeline")
def train_preprocessing_pipeline(training_data: pd.DataFrame) -> Pipeline:
    """
    Train the preprocessing pipeline on the training data.
    
    Args:
        training_data: Training data as DataFrame
        
    Returns:
        Pipeline: The fitted preprocessing pipeline
    """
    logger.info(f"Training preprocessing pipeline on data with shape {training_data.shape}")
    
    # Select only required features
    training_data = training_data[settings.REQUIRED_FEATURES]
    
    # Create the pipeline
    pipeline = create_preprocessing_pipeline()
    # Fit the pipeline
    pipeline.fit(training_data)
    
    # Get transformed feature names and count
    feature_names = get_feature_names(pipeline)
    feature_count = len(feature_names)
    logger.info(f"Pipeline trained with {feature_count} output features")
    
    return pipeline


def get_feature_names(pipeline: Union[Pipeline, ColumnTransformer]) -> List[str]:
    """
    Get the feature names after preprocessing transformations.
    
    Args:
        pipeline: The preprocessing pipeline
        
    Returns:
        List[str]: List of feature names after transformations
    """
    # Extract the column transformer
    if isinstance(pipeline, Pipeline):
        column_transformer = pipeline.named_steps['column_transformer']
    else:
        # Already a ColumnTransformer
        column_transformer = pipeline
    
    # Get transformed feature names in the correct order
    transformed_features = []
    
    # First add numerical features (they keep their names)
    for name, _, features in column_transformer.transformers_:
        if name == 'num':
            logger.info(f"Adding numerical features: {features}")
            transformed_features.extend(features)
    
    # Then add categorical features in a consistent order
    categorical_order = ['Pclass', 'Sex', 'Embarked']  # Define explicit order
    for cat_feature in categorical_order:
        for name, transformer, features in column_transformer.transformers_:
            if name.lower() == cat_feature.lower() and features[0] == cat_feature:
                # Get the OneHotEncoder for this feature
                encoder = transformer.named_steps['encoder']
                # Skip the first category since drop='first'
                feature_names = [f"{cat_feature}_{category}" for category in encoder.categories_[0][1:]]
                logger.info(f"Adding categorical feature {cat_feature} features: {feature_names}")
                transformed_features.extend(feature_names)
    
    logger.info(f"Total features ({len(transformed_features)}): {transformed_features}")
    return transformed_features


def verify_feature_dimensionality(X: np.ndarray) -> np.ndarray:
    """
    Verify that the transformed feature set has the expected dimensionality.
    
    Args:
        X: The transformed feature array
        
    Returns:
        np.ndarray: Feature array with correct dimensionality
    """
    # Calculate expected number of features based on configuration
    numerical_features = [f for f in settings.NUMERICAL_FEATURES if f in settings.REQUIRED_FEATURES]
    # Ensure Pclass is treated as categorical
    if "Pclass" in numerical_features:
        numerical_features.remove("Pclass")
    
    categorical_features = ['Pclass', 'Sex', 'Embarked']  # Define explicit order
    
    # Calculate expected dimensions
    numerical_dims = len(numerical_features)  # Age, Fare, SibSp, Parch
    categorical_dims = sum(len(settings.CATEGORICAL_VALUES[f]) - 1 for f in categorical_features)  # Subtract 1 for drop='first'
    expected_features = numerical_dims + categorical_dims
    
    current_features = X.shape[1]
    logger.info(f"Current feature count: {current_features}, Expected: {expected_features}")
    logger.info(f"Numerical features ({numerical_dims}): {numerical_features}")
    logger.info(f"Categorical features with dimensions:")
    for f in categorical_features:
        dims = len(settings.CATEGORICAL_VALUES[f]) - 1
        logger.info(f"  - {f}: {dims} dimensions (categories: {settings.CATEGORICAL_VALUES[f]})")
    
    if current_features == expected_features:
        logger.info(f"Feature dimensionality is correct: {current_features}")
        return X
        
    if current_features < expected_features:
        logger.warning(f"Feature dimensionality mismatch: got {current_features}, expected {expected_features}")
        # Pad with zeros to match expected dimensionality
        padding = np.zeros((X.shape[0], expected_features - current_features))
        X_padded = np.hstack((X, padding))
        logger.info(f"Padded features to match expected dimensionality: {X_padded.shape[1]}")
        return X_padded
        
    if current_features > expected_features:
        logger.warning(f"Feature dimensionality mismatch: got {current_features}, expected {expected_features}")
        # Truncate to match expected dimensionality
        X_truncated = X[:, :expected_features]
        logger.info(f"Truncated features to match expected dimensionality: {X_truncated.shape[1]}")
        return X_truncated
