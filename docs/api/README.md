# Titanic Prediction API Documentation

## Base URL
```
https://api.example.com/v1
```

## Authentication
All API requests require an API key to be included in the header:
```
Authorization: Bearer <your-api-key>
```

## Endpoints

### Prediction
#### Make a prediction
```http
POST /predict
```

Request body:
```json
{
  "passenger_id": 1,
  "pclass": 3,
  "name": "Braund, Mr. Owen Harris",
  "sex": "male",
  "age": 22.0,
  "sibsp": 1,
  "parch": 0,
  "ticket": "A/5 21171",
  "fare": 7.25,
  "cabin": "",
  "embarked": "S"
}
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.82,
  "model_version": "1.0.0",
  "request_id": "abc-123"
}
```

### Feature Management
#### Get feature by name
```http
GET /features/{feature_name}
```

Response:
```json
{
  "name": "passenger_features.age",
  "version": "1.0",
  "type": "numeric",
  "statistics": {
    "mean": 29.7,
    "std": 14.5,
    "min": 0.42,
    "max": 80.0
  }
}
```

#### Store features
```http
POST /features
```

Request body:
```json
{
  "feature_group": "passenger_features",
  "entity_id": "123",
  "features": {
    "age": 25,
    "sex": "female",
    "pclass": 1
  }
}
```

### Monitoring
#### Get model metrics
```http
GET /metrics
```

Response:
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

## Error Responses
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

## Rate Limiting
- 100 requests per minute per API key
- 429 Too Many Requests response when exceeded

## Data Types
| Field | Type | Description |
|-------|------|-------------|
| passenger_id | integer | Unique identifier |
| pclass | integer | Passenger class (1-3) |
| name | string | Passenger name |
| sex | string | Gender (male/female) |
| age | float | Age in years |
| sibsp | integer | Number of siblings/spouses |
| parch | integer | Number of parents/children |
| ticket | string | Ticket number |
| fare | float | Ticket fare |
| cabin | string | Cabin number |
| embarked | string | Port of embarkation | 