-- Feature Store Tables

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
);

CREATE TABLE IF NOT EXISTS feature_data (
    entity_id TEXT,
    feature_name TEXT,
    feature_version TEXT,
    value TEXT,
    timestamp TIMESTAMP,
    PRIMARY KEY (entity_id, feature_name, feature_version),
    FOREIGN KEY (feature_name, feature_version) REFERENCES feature_metadata(name, version)
);

CREATE TABLE IF NOT EXISTS feature_statistics (
    feature_name TEXT,
    feature_version TEXT,
    statistics TEXT,
    FOREIGN KEY (feature_name, feature_version) REFERENCES feature_metadata(name, version)
);

-- Monitoring Tables
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    model_version TEXT,
    input_data TEXT,
    prediction REAL,
    confidence REAL,
    latency_ms INTEGER,
    timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT,
    metric_name TEXT,
    metric_value REAL,
    timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feature_drift (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT,
    feature_version TEXT,
    drift_score REAL,
    p_value REAL,
    timestamp TIMESTAMP,
    FOREIGN KEY (feature_name, feature_version) REFERENCES feature_metadata(name, version)
);

-- Indexes
CREATE INDEX idx_feature_data_lookup ON feature_data(entity_id, feature_name, feature_version);
CREATE INDEX idx_model_predictions_request ON model_predictions(request_id);
CREATE INDEX idx_model_metrics_version ON model_metrics(model_version, metric_name);
CREATE INDEX idx_feature_drift_monitoring ON feature_drift(feature_name, feature_version, timestamp); 