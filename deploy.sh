#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting deployment of Titanic Prediction Service...${NC}"

# Create namespace if it doesn't exist
echo "Creating namespace..."
kubectl create namespace titanic-prediction --dry-run=client -o yaml | kubectl apply -f -

# Apply core configurations
echo "Applying core configurations..."
kubectl apply -f kubernetes/core/deployment.yaml
kubectl apply -f kubernetes/core/service.yaml

# Apply security configurations
echo "Applying security configurations..."
kubectl apply -f kubernetes/security/network-policy.yaml

# Apply scaling configurations
echo "Applying scaling configurations..."
kubectl apply -f kubernetes/scaling/hpa.yaml

# Apply monitoring configurations
echo "Applying monitoring configurations..."
kubectl apply -f kubernetes/monitoring/prometheus-rules.yaml
kubectl apply -f kubernetes/monitoring/grafana-dashboards.yaml
kubectl apply -f kubernetes/monitoring/metrics-service.yaml
kubectl apply -f kubernetes/monitoring/service-monitor.yaml

# Apply storage configurations
echo "Applying storage configurations..."
kubectl apply -f kubernetes/storage/pv.yaml

# Apply model management configurations
echo "Applying model management configurations..."
kubectl apply -f kubernetes/model-management/register-model-job.yaml
kubectl apply -f kubernetes/model-management/copy-model-files-job.yaml
kubectl apply -f kubernetes/model-management/diagnostic-pod.yaml

# Wait for pods to be ready
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=titanic-prediction -n titanic-prediction --timeout=300s

# Check deployment status
echo "Checking deployment status..."
kubectl get deployment -n titanic-prediction titanic-prediction

# Check service status
echo "Checking service status..."
kubectl get service -n titanic-prediction titanic-prediction-service

# Check HPA status
echo "Checking HPA status..."
kubectl get hpa -n titanic-prediction titanic-prediction-hpa

echo -e "${GREEN}Deployment completed successfully!${NC}" 