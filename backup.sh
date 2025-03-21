#!/bin/bash

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="titanic_backup_${TIMESTAMP}"

echo -e "${GREEN}Starting backup process...${NC}"

# Create backup directory
mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"

# Backup Kubernetes configurations
echo "Backing up Kubernetes configurations..."
kubectl get all -n titanic-prediction -o yaml > "${BACKUP_DIR}/${BACKUP_NAME}/kubernetes_state.yaml"
kubectl get configmap -n titanic-prediction -o yaml > "${BACKUP_DIR}/${BACKUP_NAME}/configmaps.yaml"
kubectl get secret -n titanic-prediction -o yaml > "${BACKUP_DIR}/${BACKUP_NAME}/secrets.yaml"

# Backup model files
echo "Backing up model files..."
kubectl cp titanic-prediction/$(kubectl get pod -n titanic-prediction -l app=titanic-prediction -o jsonpath='{.items[0].metadata.name}'):/app/models "${BACKUP_DIR}/${BACKUP_NAME}/models"

# Backup MLflow data
echo "Backing up MLflow data..."
kubectl cp titanic-prediction/$(kubectl get pod -n titanic-prediction -l app=titanic-prediction -o jsonpath='{.items[0].metadata.name}'):/app/models/mlruns "${BACKUP_DIR}/${BACKUP_NAME}/mlruns"

# Create backup archive
echo "Creating backup archive..."
tar -czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" -C "${BACKUP_DIR}" "${BACKUP_NAME}"

# Clean up temporary directory
rm -rf "${BACKUP_DIR}/${BACKUP_NAME}"

# List available backups
echo -e "${GREEN}Available backups:${NC}"
ls -lh "${BACKUP_DIR}"/*.tar.gz

echo -e "${GREEN}Backup completed successfully!${NC}" 