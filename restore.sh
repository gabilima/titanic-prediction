#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Check if backup file is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Please provide a backup file path${NC}"
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

BACKUP_FILE=$1
TEMP_DIR="restore_temp"

echo -e "${GREEN}Starting restore process...${NC}"

# Create temporary directory
mkdir -p "${TEMP_DIR}"

# Extract backup
echo "Extracting backup..."
tar -xzf "${BACKUP_FILE}" -C "${TEMP_DIR}"

# Restore Kubernetes configurations
echo "Restoring Kubernetes configurations..."
kubectl apply -f "${TEMP_DIR}/titanic_backup_*/kubernetes_state.yaml"
kubectl apply -f "${TEMP_DIR}/titanic_backup_*/configmaps.yaml"
kubectl apply -f "${TEMP_DIR}/titanic_backup_*/secrets.yaml"

# Wait for pods to be ready
echo "Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=titanic-prediction -n titanic-prediction --timeout=300s

# Restore model files
echo "Restoring model files..."
kubectl cp "${TEMP_DIR}/titanic_backup_*/models" titanic-prediction/$(kubectl get pod -n titanic-prediction -l app=titanic-prediction -o jsonpath='{.items[0].metadata.name}'):/app/models

# Restore MLflow data
echo "Restoring MLflow data..."
kubectl cp "${TEMP_DIR}/titanic_backup_*/mlruns" titanic-prediction/$(kubectl get pod -n titanic-prediction -l app=titanic-prediction -o jsonpath='{.items[0].metadata.name}'):/app/models/mlruns

# Clean up
echo "Cleaning up..."
rm -rf "${TEMP_DIR}"

# Check deployment status
echo "Checking deployment status..."
kubectl get deployment -n titanic-prediction titanic-prediction

# Check service status
echo "Checking service status..."
kubectl get service -n titanic-prediction titanic-prediction-service

echo -e "${GREEN}Restore completed successfully!${NC}" 