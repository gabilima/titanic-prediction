import os
import pandas as pd
from pathlib import Path
from app.core.config import settings
from feature_store.feature_pipeline import (
    ImputationTransformer,
    EncodingTransformer,
    ScalingTransformer,
    DependencyManager
)

def build_pipeline(sample_data_path: str = "data/raw/train.csv"):
    """Build and save the feature pipeline."""
    # Load sample data
    df = pd.read_csv(sample_data_path)
    
    # Initialize transformers
    imputer = ImputationTransformer(
        name="imputer",
        numeric_cols=settings.NUMERICAL_FEATURES,
        categorical_cols=settings.CATEGORICAL_FEATURES
    )
    
    encoder = EncodingTransformer(
        name="encoder",
        categorical_cols=settings.CATEGORICAL_FEATURES,
        method="one-hot"
    )
    
    scaler = ScalingTransformer(
        name="scaler",
        numeric_cols=settings.NUMERICAL_FEATURES,
        method="standard"
    )
    
    # Create dependency manager
    pipeline = DependencyManager()
    
    # Add transformers in order
    pipeline.add_transformer(imputer)
    pipeline.add_transformer(encoder)
    pipeline.add_transformer(scaler)
    
    # Fit transformers
    df_transformed = df.copy()
    for transformer_name in pipeline.get_execution_order():
        transformer = pipeline._transformers[transformer_name]
        df_transformed = transformer.fit_transform(df_transformed)
    
    # Save pipeline components
    os.makedirs(settings.MODEL_PATH, exist_ok=True)
    pipeline_path = settings.MODEL_PATH / settings.FEATURE_PIPELINE_FILENAME
    
    # Save each transformer
    for transformer in pipeline._transformers.values():
        transformer_path = settings.MODEL_PATH / f"{transformer.name}.joblib"
        transformer.save(str(transformer_path))
    
    print(f"Feature pipeline saved to {pipeline_path}")
    
    return pipeline

if __name__ == "__main__":
    build_pipeline() 