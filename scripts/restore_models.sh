#!/bin/bash

# Configurações
NAMESPACE="titanic-prediction"
BACKUP_DIR="backups"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Verificar argumentos
if [ $# -ne 1 ]; then
    echo -e "${RED}Uso: $0 <arquivo_de_backup>${NC}"
    echo -e "${YELLOW}Exemplo: $0 backups/titanic_backup_20240321_123456.tar.gz${NC}"
    exit 1
fi

BACKUP_FILE=$1

# Verificar se o arquivo de backup existe
if [ ! -f "${BACKUP_FILE}" ]; then
    echo -e "${RED}Arquivo de backup não encontrado: ${BACKUP_FILE}${NC}"
    exit 1
fi

echo -e "${YELLOW}Iniciando restore dos modelos...${NC}"

# Verificar se o pod está rodando
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=titanic-prediction -o jsonpath="{.items[0].metadata.name}")
if [ -z "${POD_NAME}" ]; then
    echo -e "${RED}Nenhum pod encontrado no namespace ${NAMESPACE}${NC}"
    exit 1
fi

# Copiar backup para o pod
echo -e "${YELLOW}Copiando backup para o pod...${NC}"
kubectl cp ${BACKUP_FILE} ${NAMESPACE}/${POD_NAME}:/tmp/models_backup.tar.gz
if [ $? -ne 0 ]; then
    echo -e "${RED}Erro ao copiar backup para o pod${NC}"
    exit 1
fi

# Restaurar modelos
echo -e "${YELLOW}Restaurando modelos...${NC}"
kubectl exec -n ${NAMESPACE} ${POD_NAME} -- tar xzf /tmp/models_backup.tar.gz -C /app/models
if [ $? -ne 0 ]; then
    echo -e "${RED}Erro ao restaurar modelos${NC}"
    exit 1
fi

# Limpar backup temporário no pod
kubectl exec -n ${NAMESPACE} ${POD_NAME} -- rm /tmp/models_backup.tar.gz

# Reiniciar o pod para aplicar as mudanças
echo -e "${YELLOW}Reiniciando pod para aplicar as mudanças...${NC}"
kubectl rollout restart deployment titanic-prediction -n ${NAMESPACE}

echo -e "${GREEN}Restore concluído com sucesso!${NC}"
echo -e "${YELLOW}Aguardando pod reiniciar...${NC}"
kubectl rollout status deployment titanic-prediction -n ${NAMESPACE} 