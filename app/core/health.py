from typing import Dict, Any, ClassVar
from pydantic import BaseModel

class HealthConfig(BaseModel):
    """Configurações para verificação de saúde."""
    # Limites de performance
    MODEL_HEALTH_THRESHOLD_MS: float = 500.0  # Tempo máximo de resposta do modelo
    MEMORY_THRESHOLD_MB: float = 2048.0  # Limite de uso de memória
    CPU_THRESHOLD_PERCENT: float = 90.0  # Limite de uso de CPU
    MIN_PREDICTIONS_PER_MINUTE: int = 5  # Taxa mínima de predições
    
    # Dados de teste para verificação
    TEST_INPUT: ClassVar[Dict[str, Any]] = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 29,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "S"
    }

    class Config:
        protected_namespaces = ()  # Desativa namespaces protegidos para resolver o warning do model_version

# Instância global de configuração
health_config = HealthConfig() 