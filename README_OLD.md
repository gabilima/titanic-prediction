# Titanic Survival Prediction Service

## Project Overview
This service provides an API for predicting the survival probability of Titanic passengers based on their characteristics. Built as a production-ready machine learning service, it demonstrates best practices in ML model deployment, API design, containerization, orchestration, and monitoring.

## Features
- **Production-ready ML Service**: Scalable, containerized, and monitored ML service
- **Model Prediction API**: REST endpoints for single and batch predictions
- **Input Validation**: Comprehensive validation of prediction request parameters
- **Model Serving**: Efficient serving of RandomForest models
- **Containerization**: Docker-based packaging for consistent deployment
- **Orchestration**: Kubernetes deployment with horizontal scaling
- **Monitoring**: Prometheus metrics for tracking predictions, latency, and errors
- **Health Checks**: Endpoint for service health monitoring

## Technology Stack
- **Python 3.9**: Base runtime environment
- **FastAPI 0.103.1**: High-performance web framework
- **Pydantic 2.4.2**: Data validation and settings management
- **Scikit-learn 1.2.2**: Machine learning toolkit for RandomForest model
- **MLflow 2.3.1**: Model registry and versioning
- **Prometheus Client 0.17.0**: Metrics collection and exposure
- **Uvicorn 0.23.2**: ASGI server for FastAPI
- **Docker**: Containerization platform
- **Kubernetes**: Container orchestration

## Project Structure
```
.
├── app
│   ├── api                 # API layer with endpoints and schemas
│   │   ├── endpoints       # API route handlers
│   │   └── schemas         # Request/response Pydantic models
│   ├── core                # Core application configuration
│   ├── ml                  # ML model implementation
│   └── monitoring          # Prometheus metrics collection
├── kubernetes              # Kubernetes deployment configuration
│   ├── deployment.yaml     # Main deployment specification
│   ├── hpa.yaml            # Horizontal Pod Autoscaler
│   └── service-monitor.yaml # Prometheus ServiceMonitor
├── requirements            # Dependency specifications
│   ├── dev.txt             # Development dependencies
│   ├── prod.txt            # Production dependencies
│   └── test.txt            # Testing dependencies
├── tests                   # Test suite
├── Dockerfile              # Container definition
└── README.md               # Project documentation
```

## API Reference

### Health Check
- **Endpoint**: `/health`
- **Method**: GET
- **Response**: 
  ```json
  {
    "status": "ok",
    "version": "1.0.0"
  }
  ```

### Single Prediction Endpoint
- **Endpoint**: `/predict`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "pclass": 1,
    "sex": "female",
    "age": 29,
    "sibsp": 0,
    "parch": 0,
    "fare": 211.3375,
    "embarked": "S"
  }
  ```
- **Response**:
  ```json
  {
    "request_id": "3e9b1bc8-2194-4c7c-ae1b-96c3b26d9cc0",
    "survival_probability": 0.87,
    "survival_prediction": true,
    "processing_time_ms": 5.32
  }
  ```

### Batch Prediction Endpoint
- **Endpoint**: `/predict/batch`
- **Method**: POST
- **Request Body**:
  ```json
  {
    "passengers": [
      {
        "pclass": 1,
        "sex": "female",
        "age": 29,
        "sibsp": 0,
        "parch": 0,
        "fare": 211.3375,
        "embarked": "S"
      },
      {
        "pclass": 3,
        "sex": "male",
        "age": 25,
        "sibsp": 0,
        "parch": 0,
        "fare": 7.225,
        "embarked": "C"
      }
    ]
  }
  ```
- **Response**:
  ```json
  {
    "request_id": "f8b54d9c-e3fe-4deb-92a2-c0d8f984a8e7",
    "predictions": [
      {
        "survival_probability": 0.87,
        "survival_prediction": true
      },
      {
        "survival_probability": 0.12,
        "survival_prediction": false
      }
    ],
    "processing_time_ms": 8.76
  }
  ```

### Input Validation Rules
- `pclass`: Integer between 1-3
- `sex`: String, either "male" or "female"
- `age`: Positive float or integer
- `sibsp`: Non-negative integer
- `parch`: Non-negative integer
- `fare`: Positive float or integer
- `embarked`: String, one of "C", "Q", or "S"

## Setup & Deployment

### Local Development
1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements/dev.txt
   ```
4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```
5. Access the API at [http://localhost:8000](http://localhost:8000) and the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)

### Docker
1. Build the Docker image:
   ```bash
   docker build -t titanic-prediction-service:latest .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 titanic-prediction-service:latest
   ```
3. Access the API at [http://localhost:8000](http://localhost:8000)

### Kubernetes
1. Apply the Kubernetes configuration:
   ```bash
   kubectl apply -f kubernetes/
   ```
2. The deployment includes:
   - 2 replicas by default
   - Resource requests: 100m CPU, 512Mi Memory
   - Resource limits: 500m CPU, 1Gi Memory
   - Horizontal Pod Autoscaler (HPA) that scales from 2 to 10 replicas based on 80% CPU and 80% memory utilization
   - Service exposure via ClusterIP
   - Prometheus ServiceMonitor for metrics collection (15s scrape interval)

## Monitoring
The service exposes Prometheus metrics at the `/metrics` endpoint. Key metrics include:

- **prediction_requests_total**: Counter of total prediction requests
- **prediction_latency_seconds**: Histogram of prediction latency
- **prediction_errors_total**: Counter of prediction errors

The Kubernetes deployment includes a ServiceMonitor configuration for automatic discovery by Prometheus. The metrics are scraped every 15 seconds.

To visualize the metrics:
1. Ensure Prometheus is configured to discover ServiceMonitors
2. Access Prometheus and query the available metrics
3. Set up Grafana dashboards for visualization

Example Prometheus query to monitor prediction success rate:
```
sum(rate(prediction_requests_total{status="success"}[5m])) 
/ 
sum(rate(prediction_requests_total[5m]))
```

# Titanic Survival Prediction API

## 1. Project Overview

This service provides machine learning-based predictions for passenger survival probability based on the Titanic dataset. It offers both single-instance and batch prediction capabilities through RESTful API endpoints. The service is designed to be scalable, monitorable, and deployable in production Kubernetes environments.

## 2. Features

- **Prediction Endpoints**:
  - Single prediction for individual passenger data
  - Batch prediction for multiple passengers
  - Input validation with meaningful error messages
- **Production-Ready Deployment**:
  - Dockerized application
  - Kubernetes deployment with horizontal pod autoscaling
  - Resource management and limits
  - Health check endpoints
- **Monitoring & Observability**:
  - Prometheus metrics for request tracking
  - Latency monitoring
  - Error rate tracking
  - ServiceMonitor integration
- **MLOps Integration**:
  - MLflow model registry integration
  - Structured prediction logging
  - Model version management

## 3. Technology Stack

The service is built using the following technologies (exact versions from requirements/prod.txt):

- **Python 3.9** (base runtime)
- **FastAPI 0.103.1** (web framework)
- **Pydantic 2.4.2** (data validation)
- **Scikit-learn 1.2.2** (ML model implementation)
- **MLflow 2.3.1** (model management)
- **Prometheus-client 0.17.0** (metrics collection)
- **Uvicorn 0.23.2** (ASGI server)
- **Docker** (containerization)
- **Kubernetes** (orchestration)

## 4. Project Structure

```
├── app/                       # Main application code
│   ├── api/                   # API implementation
│   │   ├── endpoints/         # API route handlers
│   │   └── schemas/           # Request/response models
│   ├── core/                  # Core application code
│   │   └── config.py          # Application configuration
│   ├── ml/                    # Machine learning code
│   │   └── model.py           # Model implementation
│   └── monitoring/            # Monitoring implementation
│       └── metrics.py         # Prometheus metrics
├── kubernetes/                # Kubernetes manifests
│   ├── deployment.yaml        # Main application deployment
│   ├── hpa.yaml               # Horizontal Pod Autoscaler
│   └── service-monitor.yaml   # Prometheus ServiceMonitor
├── requirements/              # Application dependencies
│   ├── dev.txt                # Development dependencies
│   ├── prod.txt               # Production dependencies
│   └── test.txt               # Testing dependencies
├── tests/                     # Test suite
├── Dockerfile                 # Docker image definition
└── README.md                  # This documentation
```

## 5. API Documentation

### Health Check Endpoint

```
GET /health
```

Returns the health status of the service.

**Response:**
```json
{
  "status": "ok"
}
```

### Single Prediction Endpoint

```
POST /api/v1/predict
```

Make a prediction for a single passenger.

**Request Schema:**
```json
{
  "pclass": 3,
  "sex": "male",
  "age": 22.0,
  "sibsp": 1,
  "parch": 0,
  "fare": 7.25,
  "embarked": "S"
}
```

**Response Schema:**
```json
{
  "request_id": "c29b5441-d45e-4544-a34a-7cebb364dbd7",
  "survival_probability": 0.17,
  "survival_prediction": 0,
  "processing_time_ms": 5.23
}
```

### Batch Prediction Endpoint

```
POST /api/v1/predict/batch
```

Make predictions for multiple passengers.

**Request Schema:**
```json
{
  "passengers": [
    {
      "pclass": 3,
      "sex": "male",
      "age": 22.0,
      "sibsp": 1,
      "parch": 0,
      "fare": 7.25,
      "embarked": "S"
    },
    {
      "pclass": 1,
      "sex": "female",
      "age": 38.0,
      "sibsp": 1,
      "parch": 0,
      "fare": 71.28,
      "embarked": "C"
    }
  ]
}
```

**Response Schema:**
```json
{
  "request_id": "a19c5ed3-fd75-4c4f-a6d5-fbd7e31299bf",
  "predictions": [
    {
      "survival_probability": 0.17,
      "survival_prediction": 0
    },
    {
      "survival_probability": 0.95,
      "survival_prediction": 1
    }
  ],
  "processing_time_ms": 7.81
}
```

### Input Validation Rules

The prediction endpoints enforce the following validation rules:

- `pclass`: Integer between 1-3
- `sex`: String, must be "male" or "female"
- `age`: Float, optional (can be null)
- `sibsp`: Integer, number of siblings/spouses aboard
- `parch`: Integer, number of parents/children aboard
- `fare`: Float, passenger fare
- `embarked`: String, must be one of "C", "Q", or "S"

## 6. Setup Instructions

### Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements/dev.txt
   ```
3. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```
4. Access the API documentation at http://localhost:8000/docs

### Docker

1. Build the Docker image:
   ```bash
   docker build -t titanic-prediction-api:latest .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 titanic-prediction-api:latest
   ```

### Kubernetes

1. Apply the Kubernetes manifests:
   ```bash
   kubectl apply -f kubernetes/
   ```

This will create:
- A deployment with 2 replicas
- A service exposing the API
- A Horizontal Pod Autoscaler to scale between 2-10 replicas
- A ServiceMonitor for Prometheus integration

**Deployment Configuration:**
- Resource requests: 100m CPU, 512Mi Memory
- Resource limits: 500m CPU, 1Gi Memory
- HPA scales based on 80% CPU and 80% Memory utilization

## 7. Monitoring

### Prometheus Metrics

The service exposes the following metrics at the `/metrics` endpoint:

- `http_requests_total`: Total count of HTTP requests (labels: endpoint, method, status)
- `http_request_duration_seconds`: Histogram of request durations (labels: endpoint, method)
- `prediction_requests_total`: Count of prediction requests (labels: endpoint, prediction_result)
- `prediction_processing_time_seconds`: Histogram of prediction processing times

### ServiceMonitor Configuration

A Prometheus ServiceMonitor is configured to scrape metrics every 15 seconds. To view the configuration:

```bash
kubectl get servicemonitor titanic-prediction-api -o yaml
```

Key configuration:
- Scrape interval: 15s
- Endpoint path: /metrics
- Selector matches app: titanic-prediction-api

To view metrics in Prometheus:
1. Access your Prometheus instance
2. Navigate to the "Targets" section to confirm the ServiceMonitor is working
3. Use the Prometheus query interface to query metrics (e.g., `prediction_requests_total`)

# Titanic Survival Prediction API

This service provides a machine learning API that predicts passenger survival on the Titanic based on passenger attributes. The API is built with FastAPI, uses a scikit-learn Random Forest model, and includes monitoring via Prometheus.

## Table of Contents

- [API Endpoints](#api-endpoints)
  - [Single Prediction](#single-prediction)
  - [Batch Prediction](#batch-prediction)
  - [Health Check](#health-check)
- [Request/Response Schemas](#requestresponse-schemas)
- [Example Usage](#example-usage)
- [Input Validation](#input-validation)
- [Monitoring](#monitoring)
- [Deployment](#deployment)

## API Endpoints

The service provides two main prediction endpoints and a health check endpoint:

### Single Prediction

Make a prediction for a single passenger:

```
POST /api/v1/predictions
```

### Batch Prediction

Make predictions for multiple passengers in a single request:

```
POST /api/v1/batch-predictions
```

### Health Check

Verify the service is running:

```
GET /health
```

## Request/Response Schemas

### Single Prediction Request

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 22,
  "sibsp": 1,
  "parch": 0,
  "fare": 53.1,
  "embarked": "S"
}
```

### Batch Prediction Request

```json
{
  "passengers": [
    {
      "pclass": 1,
      "sex": "female",
      "age": 22,
      "sibsp": 1,
      "parch": 0,
      "fare": 53.1,
      "embarked": "S"
    },
    {
      "pclass": 3,
      "sex": "male",
      "age": 28,
      "sibsp": 0,
      "parch": 0,
      "fare": 8.05,
      "embarked": "S"
    }
  ]
}
```

### Single Prediction Response

```json
{
  "request_id": "3a9cd031-c9a9-4113-a75b-90f35d2d59a2",
  "survival_prediction": true,
  "survival_probability": 0.87,
  "processing_time_ms": 12
}
```

### Batch Prediction Response

```json
{
  "request_id": "98dac87e-77a0-4554-b5b1-96324fda70e3",
  "predictions": [
    {
      "survival_prediction": true,
      "survival_probability": 0.87
    },
    {
      "survival_prediction": false,
      "survival_probability": 0.23
    }
  ],
  "processing_time_ms": 25
}
```

### Health Check Response

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 1234
}
```

## Example Usage

### Single Prediction Example

```bash
curl -X POST "http://localhost:8000/api/v1/predictions" \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 22,
    "sibsp": 1,
    "parch": 0,
    "fare": 53.1,
    "embarked": "S"
  }'
```

### Batch Prediction Example

```bash
curl -X POST "http://localhost:8000/api/v1/batch-predictions" \
  -H "Content-Type: application/json" \
  -d '{
    "passengers": [
      {
        "pclass": 1,
        "sex": "female",
        "age": 22,
        "sibsp": 1,
        "parch": 0,
        "fare": 53.1,
        "embarked": "S"
      },
      {
        "pclass": 3,
        "sex": "male",
        "age": 28,
        "sibsp": 0,
        "parch": 0,
        "fare": 8.05,
        "embarked": "S"
      }
    ]
  }'
```

## Input Validation

The following validation rules are applied to input data:

- `pclass`: Must be 1, 2, or 3 (passenger class)
- `sex`: Must be "male" or "female"
- `age`: Must be a positive number
- `sibsp`: Number of siblings/spouses aboard
- `parch`: Number of parents/children aboard
- `fare`: Passenger fare (must be positive)
- `embarked`: Port of embarkation, must be one of:
  - "C" (Cherbourg)
  - "Q" (Queenstown)
  - "S" (Southampton)

If validation fails, the API returns a 422 error with details about the validation failures.

## Monitoring

The service exposes Prometheus metrics at the `/metrics` endpoint, which include:

- `prediction_requests_total`: Total number of prediction requests
- `prediction_latency_seconds`: Prediction request latency in seconds
- `prediction_errors_total`: Total number of prediction errors

The Kubernetes deployment includes a ServiceMonitor with a 15-second scrape interval for Prometheus.

## Deployment

### Docker

Build and run the Docker container:

```bash
docker build -t titanic-prediction-api .
docker run -p 8000:8000 titanic-prediction-api
```

### Kubernetes

Deploy to Kubernetes:

```bash
kubectl apply -f kubernetes/
```

The Kubernetes deployment includes:
- 2 replicas by default
- Resource limits: 500m CPU, 1Gi Memory
- Resource requests: 100m CPU, 512Mi Memory
- HPA configuration scaling from 2 to 10 replicas based on 80% CPU and Memory utilization
- Prometheus ServiceMonitor integration
- Liveness and readiness probes
- Rolling update strategy for zero-downtime deployments

# Titanic Prediction Service

## 1. Project Overview

This service provides machine learning prediction capabilities for the Titanic survival dataset. It offers a RESTful API built with FastAPI that serves predictions from a trained Random Forest model. The service is containerized with Docker and designed for Kubernetes deployments with proper monitoring and scaling capabilities.

The prediction service takes passenger information (such as age, sex, class, etc.) as input and returns the probability of survival. It's designed to be scalable, reliable, and observable in production environments.

## 2. Project Stack & Dependencies

The service uses the following technologies:

* **Python 3.9**: Base runtime environment
* **FastAPI 0.103.1**: Web framework for building APIs
* **Pydantic 2.4.2**: Data validation and settings management
* **Scikit-learn 1.2.2**: Machine learning framework for the prediction model
* **Prometheus Client 0.17.0**: Metrics collection and exposition
* **MLflow 2.3.1**: ML model management and tracking
* **Uvicorn**: ASGI server for running the FastAPI application
* **Docker**: Containerization
* **Kubernetes**: Container orchestration and scaling

Key dependencies are specified in the requirements directory:
- `requirements/prod.txt`: Production dependencies
- `requirements/dev.txt`: Development-specific dependencies (includes testing tools)
- `requirements/test.txt`: Testing-specific dependencies

## 3. Project Structure

```
.
├── app/                          # Main application code
│   ├── api/                      # API endpoints
│   │   └── endpoints/            # API route handlers
│   │       └── predict.py        # Prediction endpoint
│   ├── core/                     # Core application code
│   │   └── config.py             # Configuration management
│   ├── ml/                       # Machine learning code
│   │   └── model.py              # Model implementation
│   ├── monitoring/               # Monitoring components
│   │   └── metrics.py            # Prometheus metrics
│   └── main.py                   # Application entry point
├── kubernetes/                   # Kubernetes manifests
│   ├── deployment.yaml           # Deployment configuration
│   ├── hpa.yaml                  # Horizontal Pod Autoscaler config
│   └── service-monitor.yaml      # Prometheus ServiceMonitor
├── requirements/                 # Dependency specifications
│   ├── dev.txt                   # Development dependencies
│   ├── prod.txt                  # Production dependencies
│   └── test.txt                  # Testing dependencies
├── tests/                        # Test suite
├── Dockerfile                    # Docker container definition
└── README.md                     # Project documentation
```

## 4. Setup Instructions

### Local Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd titanic-prediction
```

2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements/dev.txt
```

4. Run the application locally:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. Access the API documentation:
```
http://localhost:8000/docs
```

### Docker Setup

1. Build the Docker image:
```bash
docker build -t titanic-prediction:latest .
```

2. Run the container:
```bash
docker run -p 8000:8000 titanic-prediction:latest
```

### Production Deployment

For production deployment, use the Kubernetes manifests in the `kubernetes/` directory:

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/
```

## 5. Kubernetes Deployment

The application is designed to run on Kubernetes with the following configuration:

* **Deployment**: 
  * 2 replicas by default
  * Resource requests: 100m CPU, 512Mi Memory
  * Resource limits: 500m CPU, 1Gi Memory
  * Rolling update strategy with zero downtime
  * Proper liveness and readiness probes

* **Horizontal Pod Autoscaler (HPA)**:
  * Scales from 2 to 10 replicas
  * Scales based on CPU (80%) and memory (80%) utilization

* **Service and Networking**:
  * Exposed as a service on port 8000
  * Health check endpoints for monitoring

## 6. Monitoring & Observability

Monitoring is implemented using Prometheus:

* **Metrics Exposure**:
  * Service exposes metrics on the `/metrics` endpoint
  * ServiceMonitor configured with 15-second scrape interval

* **Available Metrics**:
  * `prediction_requests_total`: Counter of total prediction requests
  * `prediction_latency_seconds`: Histogram of prediction latency
  * `prediction_errors_total`: Counter of prediction errors

* **Prometheus Integration**:
  * Prometheus annotations in the Kubernetes deployment
  * ServiceMonitor included in Kubernetes manifests

To view metrics, make requests to the service and then access:
```
http://localhost:8000/metrics
```

In a Kubernetes environment with Prometheus and Grafana, you can create dashboards to visualize these metrics.

## 7. Application Features

* **Prediction API**: REST endpoint for making predictions
* **Health Checks**: Endpoints for checking service health
* **Model Management**: Integration with MLflow for model versioning
* **Configuration Management**: Environment-based configuration via ConfigMap
* **Secrets Management**: Secure handling of sensitive information
* **Scalability**: Automatic scaling based on load
* **Observability**: Comprehensive metrics for monitoring

### API Usage Example

Make a prediction:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "male",
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 50,
    "embarked": "S"
  }'
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.2,
  "model_version": "1.0.0"
}
```

# Titanic Prediction Service

## 1. Project Overview

The Titanic Prediction Service is a machine learning model deployment that predicts survival probabilities for passengers based on the Titanic dataset. This service is implemented as a modern, production-ready microservice with the following key features:

- RESTful API built with FastAPI for low-latency prediction requests
- Prometheus monitoring integration for performance and operational metrics
- Containerized deployment using Docker for local development and testing
- Kubernetes configurations for scalable production deployment
- Machine learning model based on RandomForest algorithm

This service demonstrates a complete MLOps pipeline from model development to production deployment with monitoring.

## 2. Project Structure

```
titanic-prediction/
├── app/                      # Main application code
│   ├── api/                  # API routes and endpoints
│   │   ├── endpoints/        # API endpoints implementation
│   │   │   └── predict.py    # Prediction endpoint
│   │   └── routes.py         # API route definitions
│   ├── core/                 # Core application components
│   │   └── config.py         # Application configuration
│   ├── ml/                   # Machine learning components
│   │   └── model.py          # Model implementation
│   ├── monitoring/           # Monitoring implementation
│   │   └── metrics.py        # Prometheus metrics
│   ├── schemas/              # Pydantic schemas
│   │   └── prediction.py     # Prediction request/response schemas
│   └── main.py               # Application entry point
├── kubernetes/               # Kubernetes deployment manifests
│   ├── deployment.yaml       # Deployment configuration
│   ├── service.yaml          # Service configuration
│   └── hpa.yaml              # Horizontal Pod Autoscaler
├── requirements/             # Project dependencies
│   ├── dev.txt               # Development dependencies
│   ├── prod.txt              # Production dependencies
│   └── test.txt              # Testing dependencies
├── tests/                    # Test suite
│   ├── conftest.py           # Test fixtures
│   ├── test_api.py           # API endpoint tests
│   └── test_model.py         # Model tests
├── Dockerfile                # Docker configuration
└── README.md                 # Project documentation
```

## 3. Technology Stack

- **Python 3.9**: Core programming language
- **FastAPI**: High-performance web framework for building APIs
- **Prometheus Client**: For collecting and exposing metrics
- **Scikit-learn**: Machine learning library implementing the RandomForest model
- **Pydantic**: Data validation and settings management
- **Pytest**: Testing framework
- **Docker**: Container platform for packaging the application
- **Kubernetes**: Container orchestration for production deployment

## 4. Setup Instructions

### Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/titanic-prediction.git
   cd titanic-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements/dev.txt
   ```

4. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

5. The API documentation will be available at http://localhost:8000/docs

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t titanic-prediction:latest .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 titanic-prediction:latest
   ```

### Production Deployment with Kubernetes

1. Deploy to Kubernetes:
   ```bash
   kubectl apply -f kubernetes/
   ```

2. The service will be deployed with the configurations defined in the Kubernetes manifests, including:
   - Deployment with specified resource limits
   - Service to expose the API
   - Horizontal Pod Autoscaler for automatic scaling

## 5. API Endpoints

### Prediction Endpoint

**Endpoint**: `/api/v1/predict`

**Method**: POST

**Request Body**:
```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22.0,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

**Response**:
```json
{
  "prediction": 0,
  "probability": 0.12,
  "model_version": "1.0.0"
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"Pclass": 3, "Sex": "male", "Age": 22.0, "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S"}'
```

## 6. Monitoring

The service integrates with Prometheus for metrics collection and monitoring:

### Available Metrics

- **prediction_requests_total**: Counter of prediction requests
- **prediction_latency_seconds**: Histogram of prediction request latency
- **model_prediction_errors_total**: Counter of prediction errors

### Accessing Metrics

The metrics are exposed at the `/metrics` endpoint for Prometheus to scrape:

```bash
curl http://localhost:8000/metrics
```

### Setting up Prometheus

1. Configure Prometheus to scrape the metrics endpoint:
   ```yaml
   scrape_configs:
     - job_name: 'titanic-prediction'
       scrape_interval: 15s
       static_configs:
         - targets: ['titanic-prediction-service:8000']
   ```

2. For Kubernetes deployments, a ServiceMonitor can be used to automatically configure Prometheus scraping.

### Visualizing Metrics

The metrics can be visualized using Grafana by:
1. Configuring Prometheus as a data source
2. Creating dashboards to visualize the prediction latency, request count, and error rates

## 7. Testing

### Running Tests

To run the test suite:

```bash
pytest
```

For test coverage:

```bash
pytest --cov=app
```

### Test Structure

- **Unit Tests**: Testing individual components
  - `test_model.py`: Tests for the RandomForest model
  
- **Integration Tests**: Testing the API endpoints
  - `test_api.py`: Tests for the prediction API endpoints

### Load Testing

For load testing the API, you can use tools like Locust or Apache JMeter.

## License

[MIT](LICENSE)

# Titanic Prediction Service

## Project Overview

The Titanic Prediction Service is a scalable, production-ready machine learning service that predicts survival probability for passengers based on the classic Titanic dataset. This service provides both synchronous and asynchronous prediction endpoints, includes robust monitoring, and is designed for high availability in a containerized environment.

This solution provides:
- Fast, reliable predictions via RESTful API
- Scalable architecture that can handle varying loads
- Comprehensive monitoring and observability
- CI/CD pipeline integration for model training and deployment
- Feature management via a dedicated feature store
- Infrastructure as Code for repeatable deployments

## Technology Stack

### FastAPI for API Service
- **What**: FastAPI serves as the web framework for our prediction service
- **Why**: Selected for its high performance, built-in async support, automatic OpenAPI documentation, and type validation
- **Integration**: Connects to the model service layer and exposes endpoints for predictions

### MLflow for Model Management
- **What**: MLflow handles model versioning, registration, and deployment
- **Why**: Provides a central model registry, experiment tracking, and standardized packaging that makes model lifecycle management seamless
- **Integration**: The training pipeline logs models to MLflow, while the prediction service loads registered models for inference

### Feature Store Implementation
- **What**: A custom feature store that manages feature computation and transformation
- **Why**: Ensures consistent feature engineering between training and inference, prevents training-serving skew
- **Integration**: Used by both the training pipeline for model building and the prediction service for online transformations

### Kubernetes and HPA (Horizontal Pod Autoscaler)
- **What**: Orchestration platform for container deployment with auto-scaling capabilities
- **Why**: Enables efficient resource utilization, high availability, scalability, and resilience
- **Integration**: Deploys all components (API, model service, monitoring) and scales them based on CPU/memory metrics

### Prometheus Monitoring
- **What**: Time-series database for metrics collection
- **Why**: Industry standard for collecting and querying operational and performance metrics
- **Integration**: Each service exposes a /metrics endpoint that Prometheus scrapes; alerts based on these metrics

### Terraform IaC (Infrastructure as Code)
- **What**: Declarative infrastructure provisioning
- **Why**: Ensures consistent, repeatable, and version-controlled infrastructure deployments
- **Integration**: Used to provision cloud resources, Kubernetes clusters, and supporting services

## Project Structure

```
titanic-prediction/
├── app/                     # Main application code
│   ├── api/                 # API endpoints definitions
│   ├── models/              # ML model implementations
│   ├── feature_store/       # Feature engineering and storage
│   ├── schemas/             # Data validation schemas
│   └── utils/               # Utility functions
├── kubernetes/              # Kubernetes deployment manifests
│   ├── base/                # Base Kubernetes resources
│   ├── overlays/            # Environment-specific overlays
│   └── hpa/                 # Horizontal Pod Autoscaler configs
├── terraform/               # Infrastructure as Code
│   ├── modules/             # Reusable Terraform modules
│   └── environments/        # Environment-specific configurations
├── monitoring/              # Monitoring configurations
│   ├── prometheus/          # Prometheus configurations
│   ├── grafana/             # Grafana dashboards
│   └── alerts/              # Alert rules
├── tests/                   # Test suites
├── training/                # Model training code
├── docker-compose.yml       # Local development setup
├── Dockerfile               # Container definition
└── README.md                # This file
```

## Detailed Setup Instructions

### Local Development Setup

1. **Clone the repository and install dependencies**:
   ```bash
   git clone https://github.com/your-org/titanic-prediction.git
   cd titanic-prediction
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env file with your configurations
   ```

3. **Run the application locally**:
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Access the API documentation**:
   - Open your browser and go to http://localhost:8000/docs

### Docker Deployment

1. **Build the Docker image**:
   ```bash
   docker build -t titanic-prediction:latest .
   ```

2. **Run the Docker container**:
   ```bash
   docker run -p 8000:8000 \
     -e DATABASE_URL=postgresql://user:password@db:5432/titanic \
     -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
     titanic-prediction:latest
   ```

3. **Run with Docker Compose (includes dependencies)**:
   ```bash
   docker-compose up -d
   ```
   This will start the API service, PostgreSQL, Redis, and MLflow.

### Kubernetes Deployment

1. **Ensure you have access to a Kubernetes cluster**:
   ```bash
   kubectl config current-context
   ```

2. **Deploy using kustomize**:
   ```bash
   # For development environment
   kubectl apply -k kubernetes/overlays/dev
   
   # For production environment
   kubectl apply -k kubernetes/overlays/prod
   ```

3. **Verify deployment**:
   ```bash
   kubectl get pods -n titanic-prediction
   kubectl get svc -n titanic-prediction
   ```

4. **Configure Horizontal Pod Autoscaler**:
   ```bash
   kubectl apply -f kubernetes/hpa/prediction-api-hpa.yaml
   ```

5. **Check HPA status**:
   ```bash
   kubectl get hpa -n titanic-prediction
   ```

### Model Training and Deployment

1. **Run the training pipeline**:
   ```bash
   # From the project root
   python -m training.train \
     --data-path data/titanic.csv \
     --experiment-name titanic-survival \
     --run-name "random_forest_v1"
   ```

2. **Register the model in MLflow**:
   ```bash
   python -m training.register \
     --run-id <run_id_from_training> \
     --model-name titanic-survival-predictor
   ```

3. **Transition the model to production**:
   ```bash
   python -m training.transition \
     --model-name titanic-survival-predictor \
     --version 1 \
     --stage Production
   ```

4. **Deploy the updated model**:
   ```bash
   # Kubernetes will automatically pick up the new model
   # Or trigger a deployment rollout:
   kubectl rollout restart deployment/titanic-prediction -n titanic-prediction
   ```

## Testing Instructions

### Local Testing

1. **Run unit tests**:
   ```bash
   pytest tests/unit/
   ```

2. **Run integration tests**:
   ```bash
   pytest tests/integration/
   ```

3. **Test the API endpoints manually**:
   ```bash
   # Synchronous prediction
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 211.3375, "embarked": "S"}'
   
   # Asynchronous prediction request
   curl -X POST http://localhost:8000/predict/async \
     -H "Content-Type: application/json" \
     -d '{"pclass": 3, "sex": "male", "age": 34.5, "sibsp": 0, "parch": 0, "fare": 7.8292, "embarked": "Q"}'
   
   # Check async prediction result
   curl -X GET http://localhost:8000/predict/status/<prediction_id>
   ```

4. **Load testing**:
   ```bash
   # Install locust
   pip install locust
   
   # Run load test
   locust -f tests/load/locustfile.py
   ```
   Access the Locust web interface at http://localhost:8089

### Production Environment Testing

1. **Smoke test after deployment**:
   ```bash
   # Get the service endpoint
   export SERVICE_URL=$(kubectl get svc -n titanic-prediction titanic-prediction-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
   
   # Test the health endpoint
   curl http://$SERVICE_URL/health
   
   # Test a prediction
   curl -X POST http://$SERVICE_URL/predict \
     -H "Content-Type: application/json" \
     -d '{"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 211.3375, "embarked": "S"}'
   ```

2. **Monitor logs during testing**:
   ```bash
   kubectl logs -f -l app=titanic-prediction-api -n titanic-prediction
   ```

3. **Test model deployment process**:
   ```bash
   # Follow the model training and deployment steps above
   # Then verify the new model is being used:
   curl http://$SERVICE_URL/model-info
   ```

4. **A/B testing between model versions**:
   ```bash
   # Configure traffic split in the service configuration
   kubectl apply -f kubernetes/overlays/prod/ab-testing-config.yaml
   
   # Verify traffic is being split correctly using metrics in Grafana
   ```

## Monitoring Setup

### Metrics Configuration

1. **Application metrics configuration**:
   - The application automatically exposes metrics via `/metrics` endpoint
   - Key metrics include:
     - `prediction_request_duration_seconds`: Histogram of request times
     - `prediction_requests_total`: Counter of total requests
     - `prediction_errors_total`: Counter of errors
     - `active_requests`: Gauge of current active requests
     - `model_prediction_values`: Distribution of prediction values
     - `feature_drift_score`: Gauge of current feature drift

2. **Enable metrics in your environment**:
   ```bash
   # In Kubernetes
   kubectl apply -f monitoring/prometheus/service-monitor.yaml
   
   # Or configure custom scrape config
   kubectl apply -f monitoring/prometheus/scrape-config.yaml
   ```

### Dashboards

1. **Install Grafana dashboard**:
   ```bash
   kubectl apply -f monitoring/grafana/dashboards/titanic-prediction-dashboard.yaml
   ```

2. **Available dashboards**:
   - **Overview Dashboard**: General service health and performance
   - **ML Model Dashboard**: Model performance metrics and drift indicators
   - **Operational Dashboard**: Resource usage and scaling metrics

3. **Sample dashboard components**:
   - Request rates and latencies
   - Error rates and types
   - Model prediction distribution
   - Feature drift over time
   - Resource usage (CPU/memory)
   - HPA scaling events

4. **Access Grafana**:
   ```bash
   # Port forward Grafana service
   kubectl port-forward svc/grafana 3000:3000 -n monitoring
   
   # Open in browser: http://localhost:3000
   # Default credentials: admin/admin
   ```

### Alerts

1. **Deploy alerting rules**:
   ```bash
   kubectl apply -f monitoring/alerts/prediction-service-alerts.yaml
   ```

2. **Configured alerts**:
   - **High Error Rate**: Fires when error rate exceeds 5% over 5 minutes
   - **High Latency**: Fires when 95th percentile latency exceeds 500ms
   - **Feature Drift Detected**: Fires when drift score exceeds threshold
   - **Model Accuracy Decline**: Fires when accuracy metrics drop
   - **Resource Saturation**: Fires when resources are consistently near limits

3. **Configure alert receivers**:
   ```bash
   # Edit AlertManager config
   kubectl edit configmap/alertmanager-config -n monitoring
   ```

4. **Test alert firing**:
   ```bash
   # Generate high error rate
   hey -n 1000 -c 50 -m POST -d '{"invalid": "data"}' http://$SERVICE_URL/predict
   
   # Verify alert fired in AlertManager
   kubectl port-forward svc/alertmanager 9093:9093 -n monitoring
   # Open in browser: http://localhost:9093
   ```

---

For more detailed information about specific components, check the documentation in the individual folders of this repository.

