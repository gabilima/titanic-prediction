# Titanic Survival Prediction Service

## Project Overview
A production-ready machine learning service that predicts Titanic passenger survival probabilities. This project demonstrates MLOps best practices, featuring a robust API, monitoring, and deployment architecture.

## Solution Architecture

### Feature Store
The feature store provides feature management capabilities:

- **Feature Processing**:
  - Scikit-learn based feature transformations
  - Input validation and type checking
  - Data quality checks
  - Feature versioning through MLflow

- **Storage Layer**:
  - File-based feature storage
  - MLflow artifact storage for model features
  - Feature pipeline versioning

### Machine Learning Pipeline
The solution implements a production-grade ML pipeline:

1. **Data Processing**
   - Feature engineering with scikit-learn pipelines
   - Input validation using Pydantic models
   - Data quality checks
   - Feature versioning and tracking

2. **Model Architecture**
   - Random Forest Classifier implementation
   - Model versioning through MLflow
   - A/B testing capabilities
   - Configurable model parameters

3. **Monitoring & Observability**
   - Prometheus metrics integration
   - Basic feature monitoring
   - Model performance tracking
   - Request/response logging
   - Custom metrics:
     - Prediction latency
     - Request counts
     - Model version tracking

### API Layer
RESTful API with comprehensive endpoints:

1. **Prediction Endpoints**
   ```http
   POST /api/v1/predict
   ```

2. **Health & Monitoring**
   ```http
   GET /health
   GET /api/v1/health
   GET /metrics
   GET /api/v1/model/metrics
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

3. **Environment Variables**
   ```
   ENV=production
   MODEL_VERSION=latest
   ENABLE_PREDICTION_CACHE=true
   CACHE_TTL_SECONDS=3600
   BATCH_SIZE_LIMIT=100
   ENABLE_RESPONSE_COMPRESSION=true
   COMPRESSION_MINIMUM_SIZE=1000
   MLFLOW_TRACKING_URI=file:///app/mlruns
   ```

## Complete Setup Guide

### Prerequisites
- Python 3.9+
- Docker
- Kubernetes cluster (for production deployment)
- MLflow
- Git

### 1. Local Development Setup
```bash
# Clone the repository
git clone https://github.com/gabilima/titanic-prediction.git
cd titanic-prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements/dev.txt
```

### 2. Model Training and Registration
```bash
# Build the feature pipeline
python scripts/build_pipeline.py

# Register the model in MLflow
python scripts/register_model.py

# Set up MLOps components
python scripts/setup_mlops.py
```

### 3. Local Testing
```bash
# Run unit tests
pytest tests/test_model.py tests/test_pipeline.py tests/test_api.py

# Run integration tests
pytest tests/test_integration.py

# Run the application locally
uvicorn app.main:app --reload
```

### 4. Docker Build and Test
```bash
# Build the Docker image
docker build -t titanic-prediction:latest .

# Run the container
docker run -d --name titanic-api -p 8000:8000 titanic-prediction:latest

# Test the API
curl -X POST http://localhost:8000/api/v1/predict \
-H "Content-Type: application/json" \
-d '{
    "PassengerId": "123",
    "Pclass": "1",
    "Name": "John Doe",
    "Sex": "male",
    "Age": "30",
    "SibSp": "0",
    "Parch": "0",
    "Ticket": "A123",
    "Fare": "100",
    "Cabin": "C123",
    "Embarked": "S"
}'
```

### 5. Production Deployment

#### 5.1 Infrastructure Setup with Terraform
```bash
# Initialize Terraform
cd terraform
terraform init

# Plan the deployment
terraform plan -out=tfplan

# Apply the infrastructure
terraform apply tfplan
```

#### 5.2 Kubernetes Deployment
```bash
# Apply Kubernetes configurations
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
kubectl apply -f kubernetes/hpa.yaml
kubectl apply -f kubernetes/pdb.yaml
kubectl apply -f kubernetes/network-policy.yaml

# Apply monitoring configurations
kubectl apply -f kubernetes/service-monitor.yaml
kubectl apply -f kubernetes/metrics-service.yaml
kubectl apply -f kubernetes/prometheus-rules.yaml
kubectl apply -f kubernetes/grafana-dashboards.yaml

# Verify deployment
kubectl get pods -l app=titanic-prediction
kubectl get svc titanic-prediction
kubectl get ingress titanic-prediction
```

### 6. Monitoring Setup

#### 6.1 Access Monitoring Dashboards
```bash
# Port forward Grafana service
kubectl port-forward svc/grafana 3000:3000

# Access Grafana
# Open http://localhost:3000 in your browser
# Default credentials:
# Username: admin
# Password: Check kubernetes/secrets.yaml
```

#### 6.2 View Application Metrics
```bash
# Check application metrics
curl http://localhost:8000/metrics

# Check application health
curl http://localhost:8000/api/v1/health
```

### 7. Testing in Production

#### 7.1 Load Testing
```bash
# Install k6 for load testing
brew install k6  # On macOS
# or
docker pull grafana/k6  # Using Docker

# Run load tests
cd tests/performance
k6 run load_test.js
```

#### 7.2 API Testing
```bash
# Test prediction endpoint
curl -X POST http://localhost:8000/api/v1/predict \
-H "Content-Type: application/json" \
-d '{
    "PassengerId": "123",
    "Pclass": "1",
    "Name": "John Doe",
    "Sex": "male",
    "Age": "30",
    "SibSp": "0",
    "Parch": "0",
    "Ticket": "A123",
    "Fare": "100",
    "Cabin": "C123",
    "Embarked": "S"
}'

# Monitor prediction latency
curl http://localhost:8000/api/v1/model/metrics
```

### 8. Maintenance

#### 8.1 Updating the Model
```bash
# Register new model version
python scripts/register_model.py --new-version

# Update deployment
kubectl rollout restart deployment titanic-prediction
```

#### 8.2 Backup and Recovery
```bash
# Backup MLflow data
tar -czf mlflow_backup.tar.gz mlruns mlartifacts

# Backup Kubernetes configs
kubectl get all -o yaml > k8s_backup.yaml
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
├── kubernetes/              # K8s manifests
├── scripts/                # Utility scripts
├── terraform/              # Infrastructure as code
├── tests/                  # Test suites
└── requirements/           # Dependencies
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