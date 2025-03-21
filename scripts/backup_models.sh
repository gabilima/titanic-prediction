#!/bin/bash

# Configurações
NAMESPACE="titanic-prediction"
BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/titanic_backup_${TIMESTAMP}.tar.gz"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Iniciando backup dos modelos...${NC}"

# Criar diretório de backup se não existir
mkdir -p ${BACKUP_DIR}

# Verificar se o pod está rodando
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l app=titanic-prediction -o jsonpath="{.items[0].metadata.name}")
if [ -z "${POD_NAME}" ]; then
    echo -e "${RED}Nenhum pod encontrado no namespace ${NAMESPACE}${NC}"
    exit 1
fi

# Criar backup dos modelos
echo -e "${YELLOW}Criando backup dos modelos...${NC}"
kubectl exec -n ${NAMESPACE} ${POD_NAME} -- tar czf /tmp/models_backup.tar.gz -C /app/models .
if [ $? -ne 0 ]; then
    echo -e "${RED}Erro ao criar backup dos modelos${NC}"
    exit 1
fi

# Copiar backup para local
echo -e "${YELLOW}Copiando backup para local...${NC}"
kubectl cp ${NAMESPACE}/${POD_NAME}:/tmp/models_backup.tar.gz ${BACKUP_FILE}
if [ $? -ne 0 ]; then
    echo -e "${RED}Erro ao copiar backup para local${NC}"
    exit 1
fi

# Limpar backup temporário no pod
kubectl exec -n ${NAMESPACE} ${POD_NAME} -- rm /tmp/models_backup.tar.gz

echo -e "${GREEN}Backup concluído com sucesso!${NC}"
echo -e "${YELLOW}Arquivo de backup: ${BACKUP_FILE}${NC}" 