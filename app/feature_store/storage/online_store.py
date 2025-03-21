import json
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple

import redis
from redis.exceptions import RedisError
import numpy as np
import pandas as pd
from prometheus_client import Counter, Histogram, Gauge

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Prometheus metrics
FEATURE_RETRIEVAL_TIME = Histogram(
    "feature_store_retrieval_seconds", 
    "Time spent retrieving features from online store",
    ["feature_set"]
)
FEATURE_SYNC_TIME = Histogram(
    "feature_store_sync_seconds", 
    "Time spent synchronizing features to online store",
    ["feature_set"]
)
CACHE_HIT_COUNTER = Counter(
    "feature_store_cache_hits_total", 
    "Number of cache hits in feature store",
    ["feature_set"]
)
CACHE_MISS_COUNTER = Counter(
    "feature_store_cache_misses_total", 
    "Number of cache misses in feature store",
    ["feature_set"]
)
FEATURE_STORE_SIZE = Gauge(
    "feature_store_size_bytes", 
    "Size of feature store in bytes",
    ["feature_set"]
)
FEATURE_STORE_KEYS = Gauge(
    "feature_store_keys_total", 
    "Number of keys in feature store",
    ["feature_set"]
)

class RedisOnlineFeatureStore:
    """
    Redis-based online feature store for real-time feature serving.
    
    This class provides an interface to store and retrieve features from Redis,
    with support for caching, synchronization, and monitoring.
    """
    
    def __init__(
        self, 
        redis_host: str = settings.REDIS_HOST,
        redis_port: int = settings.REDIS_PORT,
        redis_db: int = settings.REDIS_FEATURE_STORE_DB,
        redis_password: Optional[str] = settings.REDIS_PASSWORD,
        ttl: int = settings.FEATURE_STORE_TTL,
        prefix: str = "features:",
        max_connections: int = 10,
        socket_timeout: int = 5,
    ):
        """
        Initialize Redis connection pool for the online feature store.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database to use
            redis_password: Redis password (if required)
            ttl: Time-to-live for cached features in seconds
            prefix: Key prefix for features in Redis
            max_connections: Maximum number of Redis connections
            socket_timeout: Socket timeout for Redis connections
        """
        self.pool = redis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            decode_responses=True,
        )
        self.ttl = ttl
        self.prefix = prefix
        logger.info(f"Initialized Redis online feature store at {redis_host}:{redis_port}/{redis_db}")
    
    def _get_redis_client(self) -> redis.Redis:
        """Get a Redis client from the connection pool."""
        return redis.Redis(connection_pool=self.pool)
    
    def _get_key(self, feature_set: str, entity_id: str) -> str:
        """
        Construct a Redis key for a specific feature set and entity ID.
        
        Args:
            feature_set: Name of the feature set
            entity_id: Unique identifier for the entity
            
        Returns:
            Formatted Redis key
        """
        return f"{self.prefix}{feature_set}:{entity_id}"
    
    def get_features(
        self, 
        feature_set: str, 
        entity_ids: Union[str, List[str]],
        features: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve features for one or more entities from the online store.
        
        Args:
            feature_set: Name of the feature set
            entity_ids: Single entity ID or list of entity IDs
            features: Optional list of specific features to retrieve (returns all if None)
            
        Returns:
            Dictionary mapping entity IDs to their feature values
        """
        start_time = time.time()
        client = self._get_redis_client()
        
        # Convert single entity_id to list for consistent processing
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        
        result = {}
        pipeline = client.pipeline()
        
        # Queue all get requests in pipeline for efficiency
        for entity_id in entity_ids:
            key = self._get_key(feature_set, entity_id)
            pipeline.get(key)
        
        # Execute pipeline
        responses = pipeline.execute()
        
        # Process responses
        for entity_id, response in zip(entity_ids, responses):
            if response:
                CACHE_HIT_COUNTER.labels(feature_set=feature_set).inc()
                try:
                    feature_values = json.loads(response)
                    # Filter to requested features if specified
                    if features:
                        feature_values = {k: v for k, v in feature_values.items() if k in features}
                    result[entity_id] = feature_values
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode feature values for {entity_id} in {feature_set}")
                    result[entity_id] = {}
            else:
                CACHE_MISS_COUNTER.labels(feature_set=feature_set).inc()
                result[entity_id] = {}
        
        elapsed = time.time() - start_time
        FEATURE_RETRIEVAL_TIME.labels(feature_set=feature_set).observe(elapsed)
        logger.debug(f"Retrieved features for {len(entity_ids)} entities from {feature_set} in {elapsed:.3f}s")
        
        return result
    
    def save_features(
        self, 
        feature_set: str, 
        features_data: Dict[str, Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Save features for one or more entities to the online store.
        
        Args:
            feature_set: Name of the feature set
            features_data: Dictionary mapping entity IDs to their feature values
            ttl: Optional custom TTL override (in seconds)
            
        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        client = self._get_redis_client()
        
        if ttl is None:
            ttl = self.ttl
        
        try:
            pipeline = client.pipeline()
            
            for entity_id, features in features_data.items():
                key = self._get_key(feature_set, entity_id)
                # Serialize with handling for numpy and pandas types
                serialized = self._serialize_features(features)
                pipeline.set(key, serialized, ex=ttl)
            
            pipeline.execute()
            success = True
            
        except RedisError as e:
            logger.error(f"Failed to save features to Redis: {str(e)}")
            success = False
        
        elapsed = time.time() - start_time
        FEATURE_SYNC_TIME.labels(feature_set=feature_set).observe(elapsed)
        logger.debug(f"Saved {len(features_data)} entities to {feature_set} in {elapsed:.3f}s")
        
        # Update metrics on feature store size
        self._update_size_metrics(feature_set)
        
        return success
    
    def _serialize_features(self, features: Dict[str, Any]) -> str:
        """
        Serialize feature values, handling numpy and pandas types.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            JSON-serialized feature string
        """
        def serialize_value(value):
            if isinstance(value, (np.ndarray, np.number)):
                return value.tolist()
            elif isinstance(value, pd.Series):
                return value.to_dict()
            elif isinstance(value, pd.DataFrame):
                return value.to_dict(orient='records')
            else:
                return value
        
        serializable = {k: serialize_value(v) for k, v in features.items()}
        return json.dumps(serializable)
    
    def delete_features(self, feature_set: str, entity_ids: Union[str, List[str]]) -> int:
        """
        Delete features for one or more entities from the online store.
        
        Args:
            feature_set: Name of the feature set
            entity_ids: Single entity ID or list of entity IDs
            
        Returns:
            Number of keys deleted
        """
        client = self._get_redis_client()
        
        # Convert single entity_id to list for consistent processing
        if isinstance(entity_ids, str):
            entity_ids = [entity_ids]
        
        keys = [self._get_key(feature_set, entity_id) for entity_id in entity_ids]
        deleted = client.delete(*keys)
        
        # Update metrics
        self._update_size_metrics(feature_set)
        
        return deleted
    
    def sync_from_dataframe(
        self, 
        feature_set: str, 
        df: pd.DataFrame,
        entity_id_column: str,
        feature_columns: Optional[List[str]] = None,
        batch_size: int = 1000,
        ttl: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Synchronize features from a pandas DataFrame to the online store.
        
        Args:
            feature_set: Name of the feature set
            df: DataFrame containing features
            entity_id_column: Column name to use as entity ID
            feature_columns: Optional list of columns to use as features (uses all except entity_id_column if None)
            batch_size: Number of entities to process in each batch
            ttl: Optional custom TTL override (in seconds)
            
        Returns:
            Tuple of (number of entities processed, number of batches)
        """
        start_time = time.time()
        
        # Determine which columns to use as features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != entity_id_column]
        
        # Validate that entity_id_column exists
        if entity_id_column not in df.columns:
            raise ValueError(f"Entity ID column '{entity_id_column}' not found in DataFrame")
        
        # Process in batches
        total_rows = len(df)
        batch_count = (total_rows + batch_size - 1) // batch_size
        
        for batch_idx in range(batch_count):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Convert batch to dictionary format for save_features
            features_data = {}
            for _, row in batch_df.iterrows():
                entity_id = str(row[entity_id_column])
                features = {col: row[col] for col in feature_columns}
                features_data[entity_id] = features
            
            # Save this batch
            self.save_features(feature_set, features_data, ttl=ttl)
            
            logger.debug(f"Synchronized batch {batch_idx+1}/{batch_count} to feature set {feature_set}")
        
        elapsed = time.time() - start_time
        logger.info(f"Synchronized {total_rows} entities to {feature_set} in {batch_count} batches ({elapsed:.3f}s)")
        
        return total_rows, batch_count
    
    def _update_size_metrics(self, feature_set: str) -> None:
        """
        Update Prometheus metrics about feature store size.
        
        Args:
            feature_set: Name of the feature set to measure
        """
        try:
            client = self._get_redis_client()
            pattern = f"{self.prefix}{feature_set}:*"
            
            # Count keys
            cursor = 0
            key_count = 0
            while True:
                cursor, keys = client.scan(cursor, pattern, 1000)
                key_count += len(keys)
                if cursor == 0:
                    break
            
            # Update key count metric
            FEATURE_STORE_KEYS.labels(feature_set=feature_set).set(key_count)
            
            # Estimate memory usage (this is an expensive operation)
            # Only do it periodically in production
            size_bytes = 0
            for key in client.scan_iter(pattern, 100):
                key_size = client.memory_usage(key)
                if key_size:
                    size_bytes += key_size
            
            # Update size metric
            FEATURE_STORE_SIZE.labels(feature_set=feature_set).set(size_bytes)
            
        except RedisError as e:
            logger.error(f"Failed to update size metrics: {str(e)}")
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the Redis connection.
        
        Returns:
            Dictionary with health status information
        """
        client = self._get_redis_client()
        start_time = time.time()
        
        try:
            # Simple ping to check connection
            ping_response = client.ping()
            
            # Get some basic stats
            info = client.info()
            
            result = {
                "status": "healthy" if ping_response else "degraded",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "used_memory": info.get("used_memory_human", "unknown"),
                "clients_connected": info.get("connected_clients", -1),
                "uptime_seconds": info.get("uptime_in_seconds", -1),
            }
        except RedisError as e:
            result = {
                "status": "unhealthy",
                "error": str(e),
                "latency_ms": round((time.time() - start_time) * 1000, 2),
            }
        
        return result
    
    def clear_feature_set(self, feature_set: str) -> int:
        """
        Clear all entities in a feature set.
        
        Args:
            feature_set: Name of the feature set to clear
            
        Returns:
            Number of keys deleted
        """
        client = self._get_redis_client()
        pattern = f"{self.prefix}{feature_set}:*"
        
        # Use scan and pipeline for efficient deletion
        cursor = 0
        deleted_count = 0
        
        while True:
            cursor, keys = client.scan(cursor, pattern, 1000)
            if keys:
                deleted_count += client.delete(*keys)
            if cursor == 0:
                break
        
        logger.info(f"Cleared {deleted_count} entities from feature set {feature_set}")
        return deleted_count
    
    def get_entity_ids(self, feature_set: str, limit: int = 1000) -> List[str]:
        """
        Get a list of entity IDs for a feature set.
        
        Args:
            feature_set: Name of the feature set
            limit: Maximum number of entity IDs to return
            
        Returns:
            List of entity IDs
        """
        client = self._get_redis_client()
        pattern = f"{self.prefix}{feature_set}:*"
        
        

