from enhanced_feature_store import (
    EnhancedFeatureStore,
    CacheConfig,
    FeatureMetadata,
    FeatureType,
    FeatureValue
)
from datetime import datetime, timedelta

def main():
    # Initialize feature store with cache configuration
    cache_config = CacheConfig(
        redis_url="redis://localhost:6379/0",
        sqlite_path="feature_cache.db"
    )
    
    feature_store = EnhancedFeatureStore(
        cache_config=cache_config,
        registry_path="feature_registry.json",
        lineage_path="feature_lineage.json"
    )
    
    # Example 1: Register a new feature
    age_feature = FeatureMetadata(
        name="passenger_age",
        description="Passenger age in years",
        feature_type=FeatureType.NUMERIC,
        group="passenger_demographics",
        owner="data_science_team",
        tags=["demographic", "core"],
        version="v1",
        validation_rules={
            "min_value": 0,
            "max_value": 120,
            "allow_null": False
        }
    )
    
    feature_store.register_feature(
        metadata=age_feature,
        dependencies={"passenger_id"},
        transformation="SELECT age FROM passengers WHERE passenger_id = :id"
    )
    
    # Example 2: Register a derived feature
    survival_prob_feature = FeatureMetadata(
        name="survival_probability",
        description="Predicted probability of survival",
        feature_type=FeatureType.NUMERIC,
        group="model_predictions",
        owner="ml_team",
        tags=["prediction", "model_output"],
        version="v1",
        validation_rules={
            "min_value": 0.0,
            "max_value": 1.0,
            "allow_null": False
        }
    )
    
    feature_store.register_feature(
        metadata=survival_prob_feature,
        dependencies={"passenger_age", "passenger_class", "passenger_sex"},
        transformation="model.predict_proba(features)[0][1]"
    )
    
    # Example 3: Store feature values
    age_value = FeatureValue(
        value=25,
        timestamp=datetime.now(),
        version="v1"
    )
    
    feature_store.cache.set("passenger_age", age_value)
    
    # Example 4: Retrieve feature with caching
    cached_age = feature_store.get_feature(
        feature_name="passenger_age",
        version="v1"
    )
    print(f"Retrieved age value: {cached_age.value if cached_age else 'Not found'}")
    
    # Example 5: List features by group
    demographic_features = feature_store.list_features(group="passenger_demographics")
    print("\nDemographic features:")
    for feature in demographic_features:
        print(f"- {feature.name} ({feature.feature_type})")
    
    # Example 6: Get feature lineage
    lineage = feature_store.get_feature_lineage("survival_probability")
    if lineage:
        print("\nFeature lineage for survival_probability:")
        print(f"Dependencies: {lineage.dependencies}")
        print(f"Transformation: {lineage.transformation}")
    
    # Example 7: Invalidate cache for a specific feature version
    feature_store.invalidate_cache(
        feature_name="passenger_age",
        version="v1"
    )

if __name__ == "__main__":
    main() 