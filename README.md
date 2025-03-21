# Titanic Survival Prediction Service

## Project Overview
A production-ready machine learning service that predicts Titanic passenger survival probabilities. This project demonstrates MLOps best practices, featuring a robust API, monitoring, and deployment architecture.

## Solution Description

### Machine Learning Pipeline
The solution implements a machine learning pipeline for the Titanic survival prediction problem:

1. **Data Processing**
   - Feature engineering with scikit-learn pipelines
   - Input data validation using Pydantic
   - Feature store implementation for feature management
   - Basic data quality checks

2. **Model Architecture**
   - Random Forest Classifier with configurable parameters
   - Basic A/B testing implementation
   - Model versioning via MLflow
   - Configurable model loading strategies

3. **Production Infrastructure**
   - FastAPI service with health checks
   - Kubernetes deployment configuration
   - Prometheus metrics integration
   - MLflow experiment tracking

4. **Quality Assurance**
   - Unit and integration tests
   - Model performance monitoring
   - Input validation
   - Health checks and fallbacks

### Key Features
- **API Endpoints**: Single and batch prediction endpoints
- **Monitoring**: Prometheus metrics for model and system performance
- **Scalability**: Kubernetes configuration with HPA
- **Testing**: Comprehensive test suite
- **Infrastructure**: Terraform configurations for deployment
- **MLOps**: Basic model versioning and A/B testing

### Production Behavior Testing

1. **Health Check**
   ```bash
   # Check service health
   curl http://localhost:8000/health
   
   # View metrics
   curl http://localhost:8000/metrics
   ```

2. **Model Monitoring**
   ```bash
   # Check model metrics
   curl http://localhost:8000/api/v1/model/metrics
   ```

3. **A/B Testing**
   ```bash
   # Configure A/B test
   curl -X POST http://localhost:8000/api/v1/ab-test/configure \
   -H "Content-Type: application/json" \
   -d '{
       "control_version": "v1.0.0",
       "treatment_version": "v1.0.1",
       "traffic_split": 0.5
   }'
   ```

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
```bash
# Single prediction
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

# Batch prediction
curl -X POST http://localhost:8000/api/v1/batch_predict \
-H "Content-Type: application/json" \
-d '{
    "passengers": [
        {
            "pclass": 1,
            "sex": "female",
            "age": 29,
            "sibsp": 0,
            "parch": 0,
            "fare": 100,
            "embarked": "S"
        }
    ]
}'
```

## Architecture & Components

### Core Components
- FastAPI service with prediction endpoints
- Scikit-learn Random Forest model
- MLflow for experiment tracking
- Prometheus metrics collection
- Kubernetes deployment configuration

### Project Structure
```
titanic-prediction/
├── app/                     # Main application code
│   ├── api/                 # API endpoints definitions
│   ├── core/                # Core configuration
│   ├── ml/                  # ML model implementation
│   └── monitoring/          # Metrics collection
├── feature_store/           # Feature management
├── kubernetes/              # K8s deployment configs
├── terraform/               # Infrastructure as code
├── tests/                   # Test suite
└── requirements/            # Dependencies
```

## Technical Stack
- **Python 3.9**: Base runtime environment
- **FastAPI**: Web framework
- **Pydantic**: Data validation
- **Scikit-learn**: ML model implementation
- **MLflow**: Model tracking
- **Prometheus Client**: Metrics collection
- **Docker & Kubernetes**: Containerization and orchestration
- **Terraform**: Infrastructure provisioning

## Monitoring & Metrics

### Available Metrics
- Request counts and latencies
- Model prediction counts and latencies
- System resource utilization
- Model performance metrics

### Health Checks
- Service health status
- Model availability
- System resource thresholds
- Prediction latency monitoring

## Feature Store

The feature store implementation provides:
- Feature computation and storage
- Basic feature validation
- Feature monitoring capabilities
- Online and offline feature access

## Testing

The project includes:
- Unit tests for core components
- Integration tests for API endpoints
- Model validation tests
- Basic performance tests

## Configuration

Key configuration options available in `app/core/config.py`:
- Model parameters
- API settings
- Monitoring configuration
- Feature store settings
- Environment-specific configurations

## Deployment

### Kubernetes
- Basic deployment configuration
- Horizontal Pod Autoscaling
- Service and ingress setup
- Resource limits and requests

### Infrastructure
- Terraform configurations for deployment
- Basic networking setup
- Resource provisioning