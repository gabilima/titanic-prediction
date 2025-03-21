# Titanic Prediction API

API de predição de sobrevivência do Titanic usando machine learning, com arquitetura moderna e boas práticas de MLOps.

## Arquitetura da Solução

### Visão Geral
A solução é construída seguindo uma arquitetura moderna de MLOps, com os seguintes componentes principais:

#### 1. Camada de Aplicação
- **API REST**: Implementada com FastAPI, fornecendo endpoints para predições e monitoramento
- **Pipeline de ML**: Pipeline completo de features e modelo treinado
- **Monitoramento**: Métricas, logs e health checks

#### 2. Camada de Infraestrutura
- **Kubernetes**: Orquestração de containers e gerenciamento de recursos
- **Nginx Ingress**: Roteamento e balanceamento de carga
- **Persistent Volumes**: Armazenamento persistente para modelos
- **ConfigMaps/Secrets**: Gerenciamento de configurações e segredos

#### 3. Camada de Monitoramento
- **Prometheus**: Coleta de métricas
- **Logs Estruturados**: Logs em formato JSON para melhor análise
- **Health Checks**: Endpoints de verificação de saúde
- **Métricas de Sistema**: Monitoramento de CPU, memória e performance

#### 4. Camada de Segurança
- **TLS/HTTPS**: Comunicação segura via Ingress
- **CORS**: Controle de acesso configurado
- **Headers de Segurança**: Proteções básicas via Ingress

### Fluxo de Dados
1. Cliente faz requisição HTTPS para a API
2. Nginx Ingress roteia a requisição para o serviço
3. API processa a requisição usando o pipeline de ML
4. Resultados são retornados ao cliente
5. Métricas e logs são coletados para monitoramento

### Escalabilidade
- Deployment com 3 réplicas fixas
- Rolling updates para zero downtime
- Load balancing via Nginx Ingress

### Alta Disponibilidade
- 3 réplicas do serviço
- Rolling updates configurados
- Health checks e liveness probes
- Backup e restauração de modelos via scripts dedicados

## Estrutura do Projeto

```
titanic-prediction/
├── app/                    # Código da aplicação
│   ├── api/               # Endpoints da API
│   ├── core/              # Configurações e utilitários
│   ├── ml/                # Código relacionado ao modelo
│   └── monitoring/        # Métricas e monitoramento
├── kubernetes/            # Configurações do Kubernetes
│   ├── config/           # ConfigMaps e Secrets
│   ├── core/             # Recursos principais (deployment, service, etc)
│   ├── monitoring/       # Configurações de monitoramento
│   ├── security/         # Configurações de segurança
│   ├── storage/          # Configurações de storage
│   └── model-management/ # Gerenciamento de modelos
├── models/               # Modelos treinados
├── requirements/         # Dependências do projeto
├── scripts/             # Scripts de automação
└── tests/               # Testes automatizados
```

## Requisitos

- Python 3.11+
- Docker
- Kubernetes
- kubectl
- Terraform (opcional, para infraestrutura)

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/gabilima/titanic-prediction.git
cd titanic-prediction
```

2. Execute o script de setup:
```bash
chmod +x setup.sh
./setup.sh
```

Este script irá:
- Criar um ambiente virtual Python
- Instalar as dependências do projeto
- Configurar as variáveis de ambiente
- Ativar o ambiente virtual automaticamente

3. Ative o ambiente virtual (se ainda não estiver ativo):
```bash
source venv/bin/activate
```

4. Verifique a instalação:
```bash
python -c "import app; print('Instalação concluída com sucesso!')"
```

## Treinamento do Modelo

O projeto utiliza MLflow para gerenciamento do ciclo de vida do modelo. Para treinar e registrar o modelo:

1. Configure o ambiente MLflow:
```bash
export MLFLOW_TRACKING_URI="file:///app/mlruns"
```

2. Execute o script de setup MLOps:
```bash
python scripts/setup_mlops.py
```

Este script irá:
- Construir e salvar o pipeline de features
- Registrar o modelo no MLflow
- Inicializar o monitoramento de features
- Gerar um relatório inicial de qualidade dos dados

Os arquivos gerados serão:
- Pipeline de features: `models/feature_pipeline.joblib`
- Modelo treinado: `models/titanic_model.joblib`
- Relatório de monitoramento: `data/monitoring/initial_report.json`

## Desenvolvimento

Para rodar a aplicação localmente:

```bash
docker-compose up --build
```

A API estará disponível em `http://localhost:8000`

## Deploy em Produção

1. Configure o certificado TLS:
```bash
./scripts/setup_tls.sh
```

2. Faça o deploy da aplicação:
```bash
./scripts/deploy.sh
```

3. Verifique o status do deploy:
```bash
kubectl get pods -n titanic-prediction
```

## Endpoints da API

- `GET /api/v1/health`: Verifica a saúde da API
- `POST /api/v1/predict`: Faz predições de sobrevivência
- `POST /api/v1/batch_predict`: Faz predições em lote
- `GET /metrics`: Métricas do Prometheus
- `GET /docs`: Documentação Swagger/OpenAPI

## Monitoramento

A aplicação inclui:
- Métricas do Prometheus (CPU, memória, latência)
- Logs estruturados em JSON
- Health checks com métricas de sistema
- Monitoramento de performance da API

## Segurança

- TLS/HTTPS via Ingress
- CORS configurado
- Headers de segurança básicos
- Usuário não-root no container

## Gerenciamento de Modelos

### Backup
```bash
./scripts/backup_models.sh
```

### Restauração
```bash
./scripts/restore_models.sh <caminho-do-backup>
```

## Testes

Execute os testes:
```bash
pytest tests/
```

## Infraestrutura

O projeto usa:
- Kubernetes para orquestração
- Nginx Ingress Controller
- Persistent Volumes para modelos
- ConfigMaps e Secrets para configurações
- Prometheus para monitoramento