"""
Data Validation Module for Training Pipeline

This module provides data validation utilities for the training pipeline, including:
- Input data schema validation using Pydantic
- Data quality checks
- Feature validation rules
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)

class FeatureType(str, Enum):
    """Enumeration of supported feature types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

class FeatureDefinition(BaseModel):
    """Definition of a feature's properties and validation rules"""
    name: str
    feature_type: FeatureType
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[str]] = None
    description: Optional[str] = None
    
    class Config:
        extra = "forbid"

class DatasetMetadata(BaseModel):
    """Metadata for a training dataset"""
    name: str
    version: str
    creation_date: datetime = Field(default_factory=datetime.now)
    feature_definitions: Dict[str, FeatureDefinition]
    target_column: str
    description: Optional[str] = None

class DataValidationReport(BaseModel):
    """Report containing data validation results"""
    validation_date: datetime = Field(default_factory=datetime.now)
    dataset_name: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    validation_errors: Dict[str, List[str]]
    warnings: List[str]
    feature_statistics: Dict[str, Dict[str, Any]]

class DataValidator:
    """
    Data validator for training datasets
    
    This class provides methods to validate training data against defined schemas
    and perform data quality checks.
    
    Example:
        ```python
        # Define feature definitions
        feature_defs = {
            "Pclass": FeatureDefinition(
                name="Pclass",
                feature_type=FeatureType.CATEGORICAL,
                allowed_values=["1", "2", "3"]
            ),
            "Age": FeatureDefinition(
                name="Age",
                feature_type=FeatureType.NUMERIC,
                min_value=0,
                max_value=120
            )
        }
        
        # Create metadata
        metadata = DatasetMetadata(
            name="titanic_training",
            version="1.0",
            feature_definitions=feature_defs,
            target_column="Survived"
        )
        
        # Create validator
        validator = DataValidator(metadata)
        
        # Validate dataset
        report = validator.validate_dataset(training_data)
        ```
    """
    
    def __init__(self, metadata: DatasetMetadata):
        self.metadata = metadata
        self.logger = logging.getLogger(__name__)
    
    def validate_dataset(self, data: pd.DataFrame) -> DataValidationReport:
        """
        Validate a training dataset against the defined schema
        
        Args:
            data: DataFrame containing the training data
            
        Returns:
            DataValidationReport containing validation results
        """
        validation_errors = {}
        warnings = []
        feature_statistics = {}
        
        # Check required columns
        missing_cols = [
            feat for feat, def_ in self.metadata.feature_definitions.items()
            if def_.required and feat not in data.columns
        ]
        if missing_cols:
            validation_errors["missing_columns"] = missing_cols
        
        # Validate each feature
        for feature_name, feature_def in self.metadata.feature_definitions.items():
            if feature_name not in data.columns:
                continue
                
            feature_errors = []
            feature_stats = self._compute_feature_statistics(data[feature_name], feature_def)
            feature_statistics[feature_name] = feature_stats
            
            # Validate according to feature type
            if feature_def.feature_type == FeatureType.NUMERIC:
                errors = self._validate_numeric_feature(data[feature_name], feature_def)
                feature_errors.extend(errors)
            
            elif feature_def.feature_type == FeatureType.CATEGORICAL:
                errors = self._validate_categorical_feature(data[feature_name], feature_def)
                feature_errors.extend(errors)
            
            if feature_errors:
                validation_errors[feature_name] = feature_errors
        
        # Create validation report
        invalid_rows = len(validation_errors)
        report = DataValidationReport(
            dataset_name=self.metadata.name,
            total_rows=len(data),
            valid_rows=len(data) - invalid_rows,
            invalid_rows=invalid_rows,
            validation_errors=validation_errors,
            warnings=warnings,
            feature_statistics=feature_statistics
        )
        
        self._log_validation_results(report)
        return report
    
    def _compute_feature_statistics(
        self, feature_data: pd.Series, feature_def: FeatureDefinition
    ) -> Dict[str, Any]:
        """Compute statistics for a feature"""
        stats = {
            "missing_count": feature_data.isnull().sum(),
            "missing_percentage": feature_data.isnull().mean() * 100
        }
        
        if feature_def.feature_type == FeatureType.NUMERIC:
            stats.update({
                "mean": feature_data.mean(),
                "std": feature_data.std(),
                "min": feature_data.min(),
                "max": feature_data.max(),
                "median": feature_data.median()
            })
        elif feature_def.feature_type == FeatureType.CATEGORICAL:
            stats.update({
                "unique_values": feature_data.nunique(),
                "value_counts": feature_data.value_counts().to_dict()
            })
        
        return stats
    
    def _validate_numeric_feature(
        self, feature_data: pd.Series, feature_def: FeatureDefinition
    ) -> List[str]:
        """Validate a numeric feature"""
        errors = []
        
        # Check for non-numeric values
        if not pd.to_numeric(feature_data, errors='coerce').notnull().all():
            errors.append(f"Contains non-numeric values")
        
        # Check value range
        if feature_def.min_value is not None:
            if feature_data.min() < feature_def.min_value:
                errors.append(f"Values below minimum ({feature_def.min_value})")
        
        if feature_def.max_value is not None:
            if feature_data.max() > feature_def.max_value:
                errors.append(f"Values above maximum ({feature_def.max_value})")
        
        return errors
    
    def _validate_categorical_feature(
        self, feature_data: pd.Series, feature_def: FeatureDefinition
    ) -> List[str]:
        """Validate a categorical feature"""
        errors = []
        
        if feature_def.allowed_values:
            invalid_values = set(feature_data.dropna().unique()) - set(feature_def.allowed_values)
            if invalid_values:
                errors.append(f"Contains invalid values: {invalid_values}")
        
        return errors
    
    def _log_validation_results(self, report: DataValidationReport):
        """Log validation results"""
        self.logger.info(
            f"Data validation completed for dataset '{report.dataset_name}'\n"
            f"Total rows: {report.total_rows}\n"
            f"Valid rows: {report.valid_rows}\n"
            f"Invalid rows: {report.invalid_rows}"
        )
        
        if report.validation_errors:
            self.logger.warning("Validation errors found:")
            for feature, errors in report.validation_errors.items():
                for error in errors:
                    self.logger.warning(f"  - {feature}: {error}")
        
        if report.warnings:
            self.logger.warning("Validation warnings:")
            for warning in report.warnings:
                self.logger.warning(f"  - {warning}") 