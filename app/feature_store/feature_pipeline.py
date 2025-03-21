import os
import logging
import schedule
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
from abc import ABC, abstractmethod
import great_expectations as ge
import mlflow
from joblib import dump, load
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("feature_pipeline")

# Metrics tracking
METRICS = {
    "feature_computation_time": [],
    "feature_validation_errors": 0,
    "pipeline_runs": 0,
    "feature_update_count": 0,
    "dependency_failures": 0
}


class FeatureTransformer(ABC):
    """Base class for all feature transformers."""

    def __init__(self, name: str, dependencies: List[str] = None):
        self.name = name
        self.dependencies = dependencies or []
        self._is_fitted = False
        self.stats = {}
        self.version = datetime.now().strftime("%Y%m%d%H%M%S")

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'FeatureTransformer':
        """Fit the transformer on the data."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data in one step."""
        self.fit(df)
        return self.transform(df)

    def save(self, path: str) -> None:
        """Save the transformer to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump(self, path)
        logger.info(f"Saved transformer {self.name} to {path}")

    @classmethod
    def load(cls, path: str) -> 'FeatureTransformer':
        """Load the transformer from disk."""
        transformer = load(path)
        logger.info(f"Loaded transformer {transformer.name} from {path}")
        return transformer


class ImputationTransformer(FeatureTransformer):
    """Handles missing value imputation for numeric and categorical features."""

    def __init__(
        self, name: str, numeric_cols: List[str], categorical_cols: List[str], 
        numeric_strategy: str = "median", categorical_strategy: str = "most_frequent"
    ):
        super().__init__(name)
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_values = {}

    def fit(self, df: pd.DataFrame) -> 'ImputationTransformer':
        """Compute imputation values based on the data."""
        for col in self.numeric_cols:
            if col in df.columns:
                if self.numeric_strategy == "mean":
                    self.fill_values[col] = df[col].mean()
                elif self.numeric_strategy == "median":
                    self.fill_values[col] = df[col].median()
                else:  # Use 0 as default
                    self.fill_values[col] = 0
        
        for col in self.categorical_cols:
            if col in df.columns:
                if self.categorical_strategy == "most_frequent":
                    self.fill_values[col] = df[col].mode()[0] if not df[col].mode().empty else "unknown"
                else:
                    self.fill_values[col] = "unknown"
        
        self._is_fitted = True
        self.stats = {
            "fitted_columns": list(self.fill_values.keys()),
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.categorical_cols
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply imputation to the data."""
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        result = df.copy()
        for col, value in self.fill_values.items():
            if col in result.columns:
                result[col] = result[col].fillna(value)
        
        return result


class EncodingTransformer(FeatureTransformer):
    """Handles categorical encoding with one-hot or label encoding."""

    def __init__(self, name: str, categorical_cols: List[str], method: str = "one-hot"):
        super().__init__(name)
        self.categorical_cols = categorical_cols
        self.method = method
        self.mappings = {}
        self.categories = {}

    def fit(self, df: pd.DataFrame) -> 'EncodingTransformer':
        """Learn encoding mappings from the data."""
        for col in self.categorical_cols:
            if col in df.columns:
                unique_values = df[col].dropna().unique()
                self.categories[col] = unique_values.tolist()
                
                if self.method == "label":
                    self.mappings[col] = {val: idx for idx, val in enumerate(unique_values)}
        
        self._is_fitted = True
        self.stats = {
            "categorical_columns": self.categorical_cols,
            "encoding_method": self.method,
            "unique_categories": {col: len(cats) for col, cats in self.categories.items()}
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding to the data."""
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        result = df.copy()
        
        if self.method == "one-hot":
            for col in self.categorical_cols:
                if col in result.columns:
                    # Only create dummies for known categories from training
                    dummies = pd.get_dummies(
                        result[col], 
                        prefix=col, 
                        dummy_na=False
                    )
                    
                    # Ensure all categories from training are present
                    for cat in self.categories.get(col, []):
                        col_name = f"{col}_{cat}"
                        if col_name not in dummies.columns:
                            dummies[col_name] = 0
                    
                    result = pd.concat([result, dummies], axis=1)
                    result = result.drop(col, axis=1)
        
        elif self.method == "label":
            for col, mapping in self.mappings.items():
                if col in result.columns:
                    # Apply mapping with a default for unseen categories
                    result[col] = result[col].map(mapping).fillna(-1).astype(int)
        
        return result


class ScalingTransformer(FeatureTransformer):
    """Handles feature scaling using various methods."""

    def __init__(self, name: str, numeric_cols: List[str], method: str = "standard"):
        super().__init__(name)
        self.numeric_cols = numeric_cols
        self.method = method
        self.stats_values = {}

    def fit(self, df: pd.DataFrame) -> 'ScalingTransformer':
        """Compute scaling parameters from the data."""
        for col in self.numeric_cols:
            if col in df.columns:
                if self.method == "standard":
                    mean = df[col].mean()
                    std = df[col].std()
                    self.stats_values[col] = {"mean": mean, "std": std if std > 0 else 1}
                
                elif self.method == "minmax":
                    min_val = df[col].min()
                    max_val = df[col].max()
                    self.stats_values[col] = {
                        "min": min_val, 
                        "max": max_val if max_val > min_val else min_val + 1
                    }
                
                elif self.method == "robust":
                    median = df[col].median()
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    self.stats_values[col] = {
                        "median": median, 
                        "iqr": iqr if iqr > 0 else 1
                    }
        
        self._is_fitted = True
        self.stats = {
            "numeric_columns": self.numeric_cols,
            "scaling_method": self.method,
            "scaling_parameters": {
                col: {k: float(v) for k, v in params.items()} 
                for col, params in self.stats_values.items()
            }
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling to the data."""
        if not self._is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        result = df.copy()
        
        for col, params in self.stats_values.items():
            if col in result.columns:
                if self.method == "standard":
                    result[col] = (result[col] - params["mean"]) / params["std"]
                
                elif self.method == "minmax":
                    result[col] = (result[col] - params["min"]) / (params["max"] - params["min"])
                
                elif self.method == "robust":
                    result[col] = (result[col] - params["median"]) / params["iqr"]
        
        return result


class FeatureGenerator(FeatureTransformer):
    """Generates new features from existing ones."""

    def __init__(self, name: str, transformations: Dict[str, Callable], dependencies: List[str] = None):
        super().__init__(name, dependencies)
        self.transformations = transformations

    def fit(self, df: pd.DataFrame) -> 'FeatureGenerator':
        """Nothing to fit for feature generation."""
        self._is_fitted = True
        self.stats = {
            "generated_features": list(self.transformations.keys())
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature generation to the data."""
        result = df.copy()
        
        for new_col, transform_func in self.transformations.items():
            try:
                result[new_col] = transform_func(result)
                logger.info(f"Generated feature: {new_col}")
            except Exception as e:
                logger.error(f"Error generating feature {new_col}: {str(e)}")
                result[new_col] = np.nan
                METRICS["dependency_failures"] += 1
        
        return result


class FeatureValidator:
    """Validates features using Great Expectations."""
    
    def __init__(self, expectations_path: Optional[str] = None):
        self.expectations_path = expectations_path
        self.expectations = {}
        
        if expectations_path and os.path.exists(expectations_path):
            try:
                # Load expectations from JSON file if it exists
                import json
                with open(expectations_path, 'r') as f:
                    self.expectations = json.load(f)
            except Exception as e:
                logger.error(f"Error loading expectations: {str(e)}")

    def add_expectation(self, column: str, expectation_type: str, **kwargs):
        """Add an expectation for a column."""
        if column not in self.expectations:
            self.expectations[column] = []
        
        self.expectations[column].append({
            "type": expectation_type,
            "params": kwargs
        })
    
    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate a dataframe against the defined expectations."""
        ge_df = ge.from_pandas(df)
        results = {"success": True, "failures": []}
        
        # Apply column-specific expectations
        for column, expectations_list in self.expectations.items():
            if column in df.columns:
                for expectation in expectations_list:
                    exp_type = expectation["type"]
                    params = expectation["params"]
                    
                    try:
                        method_name = f"expect_{exp_type}"
                        if hasattr(ge_df, method_name):
                            method = getattr(ge_df, method_name)
                            result = method(column, **params)
                            
                            if not result.success:
                                results["success"] = False
                                results["failures"].append({
                                    "column": column,
                                    "expectation": exp_type,
                                    "details": result.result
                                })
                                METRICS["feature_validation_errors"] += 1
                    except Exception as e:
                        logger.error(f"Error validating {column} with {exp_type}: {str(e)}")
                        results["success"] = False
                        results["failures"].append({
                            "column": column,
                            "expectation": exp_type,
                            "error": str(e)
                        })
                        METRICS["feature_validation_errors"] += 1
        
        # Apply dataframe-level expectations
        if "_dataframe_" in self.expectations:
            for expectation in self.expectations["_dataframe_"]:
                exp_type = expectation["type"]
                params = expectation["params"]
                
                try:
                    method_name = f"expect_{exp_type}"
                    if hasattr(ge_df, method_name):
                        method = getattr(ge_df, method_name)
                        result = method(**params)
                        
                        if not result.success:
                            results["success"] = False
                            results["failures"].append({
                                "column": "_dataframe_",
                                "expectation": exp_type,
                                "details": result.result
                            })
                            METRICS["feature_validation_errors"] += 1
                except Exception as e:
                    logger.error(f"Error validating dataframe with {exp_type}: {str(e)}")
                    results["success"] = False
                    results["failures"].append({
                        "column": "_dataframe_",
                        "expectation": exp_type,
                        "error": str(e)
                    })
                    METRICS["feature_validation_errors"] += 1
        
        return results

    def save_expectations(self, path: Optional[str] = None):
        """Save expectations to a JSON file."""
        save_path = path or self.expectations_path
        if save_path:
            try:
                import json
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(self.expectations, f, indent=2)
                logger.info(f"Saved expectations to {save_path}")
            except Exception as e:
                logger.error(f"Error saving expectations: {str(e)}")


class DependencyManager:
    """Manages dependencies between feature transformers."""
    
    def __init__(self):
        self.graph = {}
        self.execution_order = []
        self._transformers = {}
    
    def add_transformer(self, transformer: FeatureTransformer):
        """Add a transformer to the dependency graph."""
        self.graph[transformer.name] = {
            "transformer": transformer,
            "dependencies": transformer.dependencies
        }
        self._transformers[transformer.name] = transformer
    
    def get_execution_order(self) -> List[str]:
        """Get the execution order of transformers."""
        visited = set()
        temp = set()
        order = []
        
        def visit(node: str):
            if node in temp:
                raise ValueError(f"Circular dependency detected: {node}")
            if node in visited:
                return
            
            temp.add(node)
            
            # Visit dependencies first
            for dep in self.graph[node]["dependencies"]:
                if dep in self.graph:
                    visit(dep)
            
            temp.remove(node)
            visited.add(node)
            order.append(node)
        
        # Visit all nodes
        for node in self.graph:
            if node not in visited:
                visit(node)
        
        self.execution_order = order
        return order


