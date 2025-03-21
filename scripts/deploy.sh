#!/bin/bash

# Configurações
IMAGE_NAME="titanic-prediction"
IMAGE_TAG="latest"
REGISTRY="docker.io/gabilima"  # Substitua pelo seu usuário do Docker Hub
NAMESPACE="titanic-prediction"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Iniciando deploy para produção...${NC}"

# 1. Construir a imagem Docker
echo -e "${YELLOW}Construindo imagem Docker...${NC}"
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
if [ $? -ne 0 ]; then
    echo -e "${RED}Erro ao construir imagem Docker${NC}"
    exit 1
fi

# 2. Tag e push da imagem
echo -e "${YELLOW}Enviando imagem para o registro...${NC}"
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
docker push ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
if [ $? -ne 0 ]; then
    echo -e "${RED}Erro ao enviar imagem para o registro${NC}"
    exit 1
fi

# 3. Criar namespace se não existir
echo -e "${YELLOW}Criando namespace...${NC}"
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# 4. Aplicar configurações do Kubernetes
echo -e "${YELLOW}Aplicando configurações do Kubernetes...${NC}"

# ConfigMaps e Secrets
echo -e "${YELLOW}Aplicando ConfigMaps e Secrets...${NC}"
kubectl apply -f kubernetes/config/ -n ${NAMESPACE}

# Core resources
echo -e "${YELLOW}Aplicando recursos core...${NC}"
kubectl apply -f kubernetes/core/ -n ${NAMESPACE}

# Monitoring
echo -e "${YELLOW}Aplicando configurações de monitoramento...${NC}"
kubectl apply -f kubernetes/monitoring/ -n ${NAMESPACE}

# Storage
echo -e "${YELLOW}Aplicando configurações de storage...${NC}"
kubectl apply -f kubernetes/storage/ -n ${NAMESPACE}

# 5. Verificar status do deploy
echo -e "${YELLOW}Verificando status do deploy...${NC}"
kubectl get pods -n ${NAMESPACE} -w

# 6. Verificar ingress
echo -e "${YELLOW}Verificando configuração do ingress...${NC}"
kubectl get ingress -n ${NAMESPACE}

echo -e "${GREEN}Deploy concluído com sucesso!${NC}"
echo -e "${YELLOW}Para verificar os logs:${NC}"
echo "kubectl logs -f deployment/${IMAGE_NAME} -n ${NAMESPACE}" 