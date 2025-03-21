#!/bin/bash

# Configurações
NAMESPACE="titanic-prediction"
SECRET_NAME="titanic-prediction-tls-secret"
DOMAIN="titanic-prediction-api.example.com"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Configurando certificado TLS...${NC}"

# Verificar se os arquivos de certificado existem
if [ ! -f "certs/cert.pem" ] || [ ! -f "certs/key.pem" ]; then
    echo -e "${RED}Certificados não encontrados em certs/cert.pem e certs/key.pem${NC}"
    echo -e "${YELLOW}Por favor, coloque seus certificados na pasta certs/${NC}"
    exit 1
fi

# Criar secret com o certificado
echo -e "${YELLOW}Criando secret com o certificado...${NC}"
kubectl create secret tls ${SECRET_NAME} \
    --cert=certs/cert.pem \
    --key=certs/key.pem \
    -n ${NAMESPACE} \
    --dry-run=client -o yaml | kubectl apply -f -

# Verificar se o secret foi criado
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Certificado TLS configurado com sucesso!${NC}"
    echo -e "${YELLOW}Para verificar:${NC}"
    echo "kubectl get secret ${SECRET_NAME} -n ${NAMESPACE}"
else
    echo -e "${RED}Erro ao configurar certificado TLS${NC}"
    exit 1
fi 