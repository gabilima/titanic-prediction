import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field
import hashlib
import sqlite3
import redis
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("feature_store")

class FeatureType(str, Enum):
    """Enum for feature data types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"
    BOOLEAN = "boolean"
    TEXT = "text"


class FeatureMetadata(BaseModel):
    """Metadata for a feature"""
    name: str
    description: str
    feature_type: FeatureType
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    owner: Optional[str] = None
    tags: List[str] = []
    version: str
    statistics: Dict[str, Any] = {}
    transformation_logic: Optional[str] = None
    is_active: bool = True


class Feature(BaseModel):
    """Feature entity"""
    metadata: FeatureMetadata
    data: Optional[Any] = None
    
    def get_hash(self) -> str:
        """Generate a hash for the feature definition for versioning"""
        feature_def = {
            "name": self.metadata.name,
            "type": self.metadata.feature_type,
            "transformation": self.metadata.transformation_logic
        }
        return hashlib.md5(json.dumps(feature_def, sort_keys=True).encode()).hexdigest()[:10]


class FeatureValidator:
    """Validate features based on data quality metrics and constraints"""
    
    def __init__(self):
        self.validators = {
            FeatureType.NUMERIC: self._validate_numeric,
            FeatureType.CATEGORICAL: self._validate_categorical,
            FeatureType.TIMESTAMP: self._validate_timestamp,
            FeatureType.BOOLEAN: self._validate_boolean,
            FeatureType.TEXT: self._validate_text
        }
    
    def validate(self, feature: Feature, data: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """Validate feature data based on its type"""
        if feature.metadata.feature_type in self.validators:
            return self.validators[feature.metadata.feature_type](data)
        return False, {"error": f"No validator for type {feature.metadata.feature_type}"}
    
    def _validate_numeric(self, data: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """Validate numeric features"""
        stats = {
            "mean": data.mean(),
            "std": data.std(),
            "min": data.min(),
            "max": data.max(),
            "missing_rate": data.isna().mean()
        }
        
        # Example validation rules
        valid = True
        reasons = []
        
        if stats["missing_rate"] > 0.2:
            valid = False
            reasons.append(f"High missing rate: {stats['missing_rate']:.2f}")
            
        if np.abs(stats["mean"]) > 1e10:
            valid = False
            reasons.append(f"Extreme mean value: {stats['mean']}")
            
        return valid, {
            "valid": valid,
            "reasons": reasons,
            "statistics": stats
        }
    
    def _validate_categorical(self, data: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """Validate categorical features"""
        value_counts = data.value_counts()
        stats = {
            "unique_values": len(value_counts),
            "top_values": value_counts.head(5).to_dict(),
            "missing_rate": data.isna().mean()
        }
        
        valid = True
        reasons = []
        
        if stats["missing_rate"] > 0.2:
            valid = False
            reasons.append(f"High missing rate: {stats['missing_rate']:.2f}")
            
        if stats["unique_values"] > 100:
            valid = False
            reasons.append(f"Too many unique values: {stats['unique_values']}")
            
        return valid, {
            "valid": valid,
            "reasons": reasons,
            "statistics": stats
        }
    
    def _validate_timestamp(self, data: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """Validate timestamp features"""
        stats = {
            "min_date": data.min(),
            "max_date": data.max(),
            "missing_rate": data.isna().mean()
        }
        
        valid = True
        reasons = []
        
        if stats["missing_rate"] > 0.2:
            valid = False
            reasons.append(f"High missing rate: {stats['missing_rate']:.2f}")
            
        return valid, {
            "valid": valid, 
            "reasons": reasons,
            "statistics": stats
        }
    
    def _validate_boolean(self, data: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """Validate boolean features"""
        value_counts = data.value_counts(normalize=True)
        stats = {
            "distribution": value_counts.to_dict(),
            "missing_rate": data.isna().mean()
        }
        
        valid = True
        reasons = []
        
        if stats["missing_rate"] > 0.2:
            valid = False
            reasons.append(f"High missing rate: {stats['missing_rate']:.2f}")
            
        return valid, {
            "valid": valid,
            "reasons": reasons,
            "statistics": stats
        }
    
    def _validate_text(self, data: pd.Series) -> Tuple[bool, Dict[str, Any]]:
        """Validate text features"""
        stats = {
            "avg_length": data.str.len().mean(),
            "max_length": data.str.len().max(),
            "missing_rate": data.isna().mean()
        }
        
        valid = True
        reasons = []
        
        if stats["missing_rate"] > 0.2:
            valid = False
            reasons.append(f"High missing rate: {stats['missing_rate']:.2f}")
            
        return valid, {
            "valid": valid,
            "reasons": reasons,
            "statistics": stats
        }


class FeatureRegistry:
    """Registry for feature management and discovery"""
    
    def __init__(self, registry_path: str = "registry/feature_registry.json"):
        self.registry_path = registry_path
        self.features: Dict[str, Dict[str, Feature]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """Load feature registry from file if exists"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                for feature_name, versions in registry_data.items():
                    self.features[feature_name] = {}
                    for version, feature_data in versions.items():
                        self.features[feature_name][version] = Feature(**feature_data)
                logger.info(f"Loaded feature registry with {len(self.features)} features")
            except Exception as e:
                logger.error(f"Failed to load feature registry: {e}")
                self.features = {}
    
    def _save_registry(self):
        """Save feature registry to file"""
        try:
            registry_data = {}
            for feature_name, versions in self.features.items():
                registry_data[feature_name] = {}
                for version, feature in versions.items():
                    registry_data[feature_name][version] = feature.model_dump()
            
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, default=str)
            logger.info("Feature registry saved successfully")
        except Exception as e:
            logger.error(f"Failed to save feature registry: {e}")
    
    def register_feature(self, feature: Feature) -> str:
        """Register a new feature or new version of existing feature"""
        feature_name = feature.metadata.name
        
        # If feature doesn't exist yet, initialize it
        if feature_name not in self.features:
            self.features[feature_name] = {}
        
        # Generate version hash if not provided
        if not feature.metadata.version:
            feature.metadata.version = feature.get_hash()
        
        # Add to registry
        self.features[feature_name][feature.metadata.version] = feature
        self._save_registry()
        
        logger.info(f"Registered feature {feature_name} version {feature.metadata.version}")
        return feature.metadata.version
    
    def get_feature(self, name: str, version: Optional[str] = None) -> Optional[Feature]:
        """Get a feature by name and optionally version"""
        if name not in self.features:
            return None
        
        if version:
            return self.features[name].get(version)
        
        # Return latest version if no version specified
        versions = sorted(self.features[name].keys(), 
                          key=lambda v: self.features[name][v].metadata.updated_at,
                          reverse=True)
        if versions:
            return self.features[name][versions[0]]
        
        return None
    
    def list_features(self) -> List[Dict[str, Any]]:
        """List all features in the registry"""
        result = []
        for feature_name, versions in self.features.items():
            for version, feature in versions.items():
                result.append({
                    "name": feature_name,
                    "version": version,
                    "type": feature.metadata.feature_type,
                    "updated_at": feature.metadata.updated_at,
                    "owner": feature.metadata.owner,
                    "tags": feature.metadata.tags,
                    "active": feature.metadata.is_active
                })
        return result
    
    def delete_feature(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a feature or specific version"""
        if name not in self.features:
            return False
        
        if version:
            if version in self.features[name]:
                del self.features[name][version]
                if not self.features[name]:  # If no versions left
                    del self.features[name]
                self._save_registry()
                return True
            return False
        
        # Delete all versions if no version specified
        del self.features[name]
        self._save_registry()
        return True


class OfflineFeatureStore:
    """Offline feature store using SQLite"""
    
    def __init__(self, db_path: str = "feature_store.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Create feature_metadata table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_metadata (
            name TEXT,
            version TEXT,
            description TEXT,
            feature_type TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            owner TEXT,
            tags TEXT,
            transformation_logic TEXT,
            is_active INTEGER,
            PRIMARY KEY (name, version)
        )
        ''')
        
        # Create feature_data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_data (
            entity_id TEXT,
            feature_name TEXT,
            feature_version TEXT,
            value TEXT,
            timestamp TIMESTAMP,
            PRIMARY KEY (entity_id, feature_name, feature_version),
            FOREIGN KEY (feature_name, feature_version) REFERENCES feature_metadata(name, version)
        )
        ''')
        
        # Create feature_statistics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_statistics (
            feature_name TEXT,
            feature_version TEXT,
            statistics TEXT,
            FOREIGN KEY (feature_name, feature_version) REFERENCES feature_metadata(name, version)
        )
        ''')
        
        self.conn.commit()
    
    def store_feature_data(self, entity_ids: List[str], feature: Feature, values: List[Any]):
        """Store feature data for multiple entities"""
        cursor = self.conn.cursor()
        
        # First ensure the feature metadata is stored
        cursor.execute('''
        INSERT OR REPLACE INTO feature_metadata
        (name, version, description, feature_type, created_at, updated_at, owner, tags, transformation_logic, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feature.metadata.name,
            feature.metadata.version,
            feature.metadata.description,
            feature.metadata.feature_type,
            feature.metadata.created_at,
            feature.metadata.updated_at,
            feature.metadata.owner,
            json.dumps(feature.metadata.tags),
            feature.metadata.transformation_logic,
            int(feature.metadata.is_active)
        ))
        
        # Then store the feature data
        timestamp = datetime.now()
        for entity_id, value in zip(entity_ids, values):
            cursor.execute('''
            INSERT OR REPLACE INTO feature_data
            (entity_id, feature_name, feature_version, value, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                entity_id,
                feature.metadata.name,
                feature.metadata.version,
                json.dumps(value),
                timestamp
            ))
        
        # Store statistics if available
        if feature.metadata.statistics:
            cursor.execute('''
            INSERT OR REPLACE INTO feature_statistics
            (feature_name, feature_version, statistics)
            VALUES (?, ?, ?)
            ''', (
                feature.metadata.name,
                feature.metadata.version,
                json.dumps(feature.metadata.statistics)
            ))
        
        self.conn.commit()
    
    def get_feature_data(self, entity_ids: List[str], feature_name: str, feature_version: Optional[str] = None) -> Dict[str, Any]:
        """Get feature data for multiple entities"""
        cursor = self.conn.cursor()
        
        query = '''
        SELECT entity_id, value FROM feature_data
        WHERE entity_id IN ({}) AND feature_name = ?
        '''.format(','.join(['?'] * len(entity_ids)))
        
        params = entity_ids + [feature_name]
        
        if feature_version:
            query += ' AND feature_version = ?'
            params.append(feature_version)
        else:
            # Get the latest version for each entity_id
            query = '''
            SELECT fd.entity_id, fd.value FROM feature_data fd
            INNER JOIN (
                SELECT entity_id, MAX(timestamp) as max_timestamp
                FROM feature_data
                WHERE entity_id IN ({}) AND feature_name = ?
                GROUP BY entity_id
            ) latest
            ON fd.entity_id = latest.entity_id AND fd.timestamp = latest.max_timestamp
            WHERE fd.feature_name = ?
            '''.format(','.join(['?'] * len(entity_ids)))
            params = entity_ids + [feature_name, feature_name]
        
        cursor.execute(query, params)
        result = cursor.fetchall()
        
        return {entity_id: value for entity_id, value in result}


class FeatureStore:
    """Feature store for managing feature data"""
    
    def __init__(self, registry_path: str = "registry/feature_registry.json"):
        """Initialize feature store with Redis and SQLite backends"""
        self.registry_path = registry_path
        self.features = {}
        self.expectations = self._load_expectations()
        self.registry = FeatureRegistry()
        self.offline_store = OfflineFeatureStore()
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def _load_expectations(self) -> Dict[str, Any]:
        """Load feature expectations from JSON file."""
        try:
            with open('feature_store/expectations.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("expectations.json not found. Feature validation will be limited.")
            return {}

    def validate_feature_value(self, feature_name: str, value: Any) -> bool:
        """Validate a feature value against defined expectations."""
        if not self.expectations or 'features' not in self.expectations:
            return True

        feature_expectations = self.expectations.get('features', {}).get(feature_name)
        if not feature_expectations:
            return True

        try:
            # Type validation
            expected_type = feature_expectations.get('type')
            if expected_type == 'integer' and not isinstance(value, int):
                return False
            elif expected_type == 'float' and not isinstance(value, (int, float)):
                return False
            elif expected_type == 'string' and not isinstance(value, str):
                return False

            # Range validation
            if 'min_value' in feature_expectations and value < feature_expectations['min_value']:
                return False
            if 'max_value' in feature_expectations and value > feature_expectations['max_value']:
                return False

            # Category validation
            if 'categories' in feature_expectations and value not in feature_expectations['categories']:
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating feature {feature_name}: {str(e)}")
            return False

    def store_features(self, feature_group: str, entity_id: str, features: Dict[str, Any], version: str = "1.0"):
        """Store features for an entity"""
        try:
            # Create feature metadata
            for feature_name, value in features.items():
                feature_type = self._infer_feature_type(value)
                metadata = FeatureMetadata(
                    name=f"{feature_group}.{feature_name}",
                    description=f"Feature {feature_name} for {feature_group}",
                    feature_type=feature_type,
                    version=version
                )
                feature = Feature(metadata=metadata, data=value)
                self.registry.register_feature(feature)
            
            # Store in offline store
            self.offline_store.store_feature_data(
                entity_ids=[entity_id],
                feature=feature,
                values=[features]
            )
            
            # Cache in Redis
            cache_key = f"{feature_group}:{entity_id}:{version}"
            self.redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(features)
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to store features: {e}")
            return False
    
    def get_features(self, feature_group: str, entity_id: str, version: str = "1.0") -> Optional[Dict[str, Any]]:
        """Retrieve features for an entity"""
        try:
            # Try cache first
            cache_key = f"{feature_group}:{entity_id}:{version}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            # Fall back to offline store
            feature_name = f"{feature_group}.{entity_id}"
            data = self.offline_store.get_feature_data(
                entity_ids=[entity_id],
                feature_name=feature_name,
                feature_version=version
            )
            
            if data and entity_id in data:
                # Update cache
                self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(data[entity_id])
                )
                return data[entity_id]
            
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve features: {e}")
            return None
    
    def _infer_feature_type(self, value: Any) -> FeatureType:
        """Infer feature type from value"""
        if isinstance(value, (int, float)):
            return FeatureType.NUMERIC
        elif isinstance(value, bool):
            return FeatureType.BOOLEAN
        elif isinstance(value, datetime):
            return FeatureType.TIMESTAMP
        elif isinstance(value, str):
            return FeatureType.TEXT
        else:
            return FeatureType.CATEGORICAL


