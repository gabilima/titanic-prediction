# Guia para Data Scientists - Titanic Prediction Service

## Introdução

Este guia fornece instruções detalhadas para Data Scientists trabalharem com o serviço de predição Titanic. Ele cobre desde a experimentação inicial até o deployment de novos modelos em produção.

## Sumário

1. [Setup do Ambiente](#setup-do-ambiente)
2. [Estrutura do Projeto](#estrutura-do-projeto)
3. [Experimentação](#experimentação)
4. [Validação de Dados](#validação-de-dados)
5. [Feature Store](#feature-store)
6. [Treinamento de Modelos](#treinamento-de-modelos)
7. [Avaliação e Métricas](#avaliação-e-métricas)
8. [MLflow Tracking](#mlflow-tracking)
9. [Deployment de Modelos](#deployment-de-modelos)
10. [Monitoramento](#monitoramento)

## Setup do Ambiente

1. Clone o repositório e crie um ambiente virtual:
```bash
git clone https://github.com/yourusername/titanic-prediction.git
cd titanic-prediction
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

2. Instale as dependências de desenvolvimento:
```bash
pip install -r requirements/dev.txt
```

3. Configure o MLflow:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Estrutura do Projeto

```
titanic-prediction/
├── app/
│   ├── ml/                 # Implementação do modelo
│   └── api/               # API endpoints
├── notebooks/            # Notebooks para experimentação
├── feature_store/       # Feature store
├── mlflow/             # Configuração MLflow
├── tests/             # Testes
└── docs/             # Documentação
```

## Experimentação

Utilize os notebooks Jupyter fornecidos para experimentação:

1. **Análise Exploratória**: `notebooks/experimentation_example.ipynb`
   - Carregamento de dados
   - Validação inicial
   - Visualizações
   - Feature engineering

2. **Prototipagem de Modelos**: `notebooks/model_prototyping.ipynb`
   - Experimentação com diferentes algoritmos
   - Otimização de hiperparâmetros
   - Avaliação preliminar

## Validação de Dados

O projeto inclui um robusto sistema de validação de dados:

```python
from app.ml.data_validation import (
    DataValidator,
    DatasetMetadata,
    FeatureDefinition,
    FeatureType
)

# Define regras de validação
feature_definitions = {
    "Pclass": FeatureDefinition(
        name="Pclass",
        feature_type=FeatureType.CATEGORICAL,
        allowed_values=["1", "2", "3"]
    ),
    "Age": FeatureDefinition(
        name="Age",
        feature_type=FeatureType.NUMERIC,
        min_value=0,
        max_value=120
    )
}

# Cria metadata do dataset
metadata = DatasetMetadata(
    name="titanic_training",
    version="1.0",
    feature_definitions=feature_definitions,
    target_column="Survived"
)

# Valida dados
validator = DataValidator(metadata)
report = validator.validate_dataset(data)
```

## Feature Store

O feature store fornece:
- Armazenamento consistente de features
- Versionamento
- Rastreabilidade
- Cache para inferência

Exemplo de uso:
```python
from feature_store.feature_store import FeatureStore

store = FeatureStore()

# Armazena features
store.store_features(
    feature_group="passenger_features",
    entity_id="passenger_123",
    features={
        "Pclass": 1,
        "Age": 30,
        "Sex": "male"
    }
)

# Recupera features
features = store.get_features(
    feature_group="passenger_features",
    entity_id="passenger_123"
)
```

## Treinamento de Modelos

O pipeline de treinamento suporta:
1. Validação automática de dados
2. Feature engineering configurável
3. Otimização de hiperparâmetros
4. Tracking de experimentos

Exemplo:
```python
from mlflow.train import run_experiment

# Treina modelo
run_experiment(
    experiment_name="titanic_experiment",
    data_path="data/train.csv",
    model_path="models/new_model.pkl",
    hyperparams={
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20]
    }
)
```

## Avaliação e Métricas

Métricas monitoradas:
- Acurácia
- Precisão
- Recall
- F1-Score
- ROC AUC
- Latência de predição
- Drift de features

## MLflow Tracking

O MLflow é usado para:
1. Tracking de experimentos
2. Versionamento de modelos
3. Gerenciamento de artefatos
4. Comparação de modelos

Exemplo:
```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.85)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

## Deployment de Modelos

Para promover um modelo para produção:

1. Registre o modelo:
```python
from mlflow.register_model import register_model

register_model(
    model_path="models/new_model.pkl",
    name="titanic_model",
    version="1.2.3"
)
```

2. Configure A/B testing:
```python
# Em config.py
AB_TESTING_CONFIG = {
    "enabled": True,
    "models": {
        "control": {"version": "1.2.2", "traffic_percentage": 90},
        "treatment": {"version": "1.2.3", "traffic_percentage": 10}
    }
}
```

## Monitoramento

Métricas disponíveis no Prometheus/Grafana:
- Taxa de predições
- Latência
- Acurácia em produção
- Drift de features
- Utilização de recursos

Alertas configurados para:
- Queda de performance
- Drift significativo
- Erros de predição
- Latência alta

## Melhores Práticas

1. **Experimentação**:
   - Use notebooks para prototipagem rápida
   - Valide dados antes do treinamento
   - Documente experimentos no MLflow

2. **Feature Engineering**:
   - Use o feature store para consistência
   - Versione transformações
   - Documente decisões

3. **Treinamento**:
   - Valide dados de entrada
   - Use cross-validation
   - Otimize hiperparâmetros
   - Registre experimentos

4. **Deployment**:
   - Teste modelos extensivamente
   - Use A/B testing
   - Monitore performance
   - Mantenha documentação atualizada

## Suporte

Para questões ou problemas:
1. Consulte a documentação em `/docs`
2. Verifique issues existentes
3. Abra uma nova issue com detalhes do problema

## Contribuindo

1. Fork o repositório
2. Crie uma branch para sua feature
3. Adicione testes
4. Atualize a documentação
5. Envie um pull request 