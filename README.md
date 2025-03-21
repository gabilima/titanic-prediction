# Titanic Survival Prediction Service

## Project Overview
A production-ready machine learning service that predicts Titanic passenger survival probabilities. This project demonstrates MLOps best practices, featuring a robust API, monitoring, and deployment architecture.

## Solution Architecture

### Feature Store
The feature store provides a robust feature management system:

- **Feature Registry**: Centralized metadata management for features
  - Version control (1.0, 2.0) for each feature
  - Feature type tracking (numeric, text)
  - Creation and update timestamps
  - Feature ownership and tagging
  - Transformation logic tracking

- **Storage Layer**:
  - Online Store: Redis-based for fast access
    - Configurable TTL (default: 3600s)
    - Optimized for real-time predictions
  - Offline Store: SQLite-based for historical data
    - Complete feature history
    - Feature statistics storage
    - Drift detection data

### Machine Learning Pipeline
The solution implements a production-grade ML pipeline:

1. **Data Processing**
   - Feature engineering with scikit-learn pipelines
   - Input validation using Pydantic models
   - Data quality checks and constraints
   - Feature versioning and tracking

2. **Model Architecture**
   - Random Forest Classifier implementation
   - Model versioning through MLflow
   - A/B testing capabilities
   - Configurable model parameters
   - Prediction caching with TTL

3. **Monitoring & Observability**
   - Prometheus metrics integration
   - Feature drift detection
   - Model performance tracking
   - Request/response logging
   - Custom metrics:
     - Prediction latency (p50, p95, p99)
     - Request counts
     - Feature drift scores
     - Model confidence metrics

### API Layer
RESTful API with comprehensive endpoints:

1. **Prediction Endpoints**
   ```http
   POST /api/v1/predict
   POST /api/v1/batch_predict
   ```

2. **Feature Management**
   ```http
   GET /features/{feature_name}
   POST /features
   ```

3. **Monitoring**
   ```http
   GET /health
   GET /metrics
   GET /api/v1/model/metrics
   ```

4. **A/B Testing**
   ```http
   POST /api/v1/ab-test/configure
   ```

### Infrastructure

1. **Kubernetes Deployment**
   - Horizontal Pod Autoscaling (HPA)
   - Pod Disruption Budget (PDB)
   - Network Policies
   - Service Monitor integration
   - Ingress configuration
   - Resource management

2. **Security**
   - API key authentication
   - Rate limiting (100 req/min)
   - Network policy restrictions
   - Secret management
   - CORS configuration

## Quick Start

### Local Development
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/dev.txt

# Run the application
uvicorn app.main:app --reload
```

### Docker Deployment
```bash
# Build the image
docker build -t titanic-prediction .

# Run the container
docker run -p 8000:8000 titanic-prediction
```

### Example API Usage

1. **Single Prediction**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
-H "Content-Type: application/json" \
-d '{
    "pclass": 1,
    "sex": "female",
    "age": 29,
    "sibsp": 0,
    "parch": 0,
    "fare": 100,
    "embarked": "S"
}'
```

2. **Feature Management**
```bash
# Get feature metadata
curl -X GET http://localhost:8000/features/passenger_features.age

# Store new features
curl -X POST http://localhost:8000/features \
-H "Content-Type: application/json" \
-d '{
    "feature_group": "passenger_features",
    "entity_id": "123",
    "features": {
        "age": 25,
        "sex": "female",
        "pclass": 1
    }
}'
```

## Project Structure
```
titanic-prediction/
├── app/                      # Main application code
│   ├── api/                  # API endpoints and routes
│   ├── core/                # Core functionality
│   ├── feature_store/       # Feature management
│   ├── ml/                  # ML models and training
│   ├── monitoring/          # Monitoring and metrics
│   └── main.py             # Application entry point
├── config/                  # Configuration files
│   └── settings.yaml       # Application settings
├── docs/                    # Documentation
│   ├── architecture/       # Architecture docs
│   └── api/                # API specification
├── kubernetes/              # K8s manifests
├── migrations/              # Database migrations
├── scripts/                # Utility scripts
├── terraform/              # Infrastructure as code
├── tests/                  # Test suites
└── requirements/           # Dependencies
```

## Configuration

Key configurations in `config/settings.yaml`:

```yaml
app:
  name: titanic-prediction
  version: 1.0.0
  environment: ${ENV:development}

server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 120

feature_store:
  registry_path: app/feature_store/registry/feature_registry.json
  offline_store:
    type: sqlite
    path: data/feature_store.db
  online_store:
    type: redis
    host: ${REDIS_HOST:localhost}
    port: ${REDIS_PORT:6379}

model:
  path: models/
  version: ${MODEL_VERSION:latest}
  batch_size: 32
  cache_predictions: true
```

## Testing

The project includes comprehensive test suites:

1. **Unit Tests**
   - Model validation
   - Feature processing
   - API endpoints
   - Core utilities

2. **Integration Tests**
   - End-to-end workflows
   - API integration
   - Database operations
   - Feature store operations

3. **Performance Tests**
   - Load testing
   - Latency benchmarks
   - Resource utilization

## Monitoring & Metrics

Available metrics at `/metrics`:
```json
{
  "prediction_requests_total": 1000,
  "prediction_latency_ms": {
    "p50": 45,
    "p95": 120,
    "p99": 200
  },
  "feature_drift_scores": {
    "age": 0.02,
    "fare": 0.05
  }
}
```

## Error Handling

Standard error response format:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input: age must be a positive number",
    "details": {
      "field": "age",
      "constraint": "positive_number"
    }
  }
}
```