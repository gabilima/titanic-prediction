# Titanic Prediction API

API de predição de sobrevivência do Titanic usando machine learning, com arquitetura moderna e boas práticas de MLOps.

## Estrutura do Projeto

```
titanic-prediction/
├── app/                    # Código da aplicação
│   ├── api/               # Endpoints da API
│   ├── core/              # Configurações e utilitários
│   ├── feature_store/     # Gerenciamento de features
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

2. Instale as dependências:
```bash
pip install -r requirements/dev.txt
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
- `GET /metrics`: Métricas do Prometheus
- `GET /docs`: Documentação Swagger/OpenAPI

## Monitoramento

A aplicação inclui:
- Métricas do Prometheus
- Logs estruturados
- Health checks
- Métricas de performance
- Monitoramento de drift de features

## Segurança

- TLS/HTTPS
- Rate limiting
- CORS configurado
- Headers de segurança
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
- Prometheus e Grafana para monitoramento