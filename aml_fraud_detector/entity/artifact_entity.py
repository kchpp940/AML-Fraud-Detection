from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ProcessStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


PROCESS_STATUS_SUCCESS = ProcessStatus.SUCCESS.value
PROCESS_STATUS_ERROR = ProcessStatus.ERROR.value

REQUIRED_INPUT_FIELDS = [
    "from_bank",
    "account",
    "to_bank",
    "account_1",
    "amount_received",
    "receiving_currency",
    "payment_currency",
    "payment_format",
    "day",
]

FOUR_ARTIFACT_FILENAMES = [
    "model.pkl",
    "preprocessor.pkl",
    "feature_metadata.json",
    "model_metadata.json",
]


@dataclass
class RiskExplanation:
    fraud_probability: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    top_factors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelVersionInfo:
    model_version: int = 0
    model_name: str = ""
    training_time: str = ""
    selection_metric: str = ""
    best_metric_value: float = 0.0
    feature_schema_version: str = ""
    artifact_path: str = ""


@dataclass
class ValidationStatus:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checked_artifacts: List[str] = field(default_factory=list)


@dataclass
class PredictionResult:
    prediction: int = -1
    fraud_probability: float = 0.0
    legit_probability: float = 0.0
    class_label: str = ""
    process_status: ProcessStatus = ProcessStatus.SUCCESS
    error_reason: Optional[str] = None
    model_version: int = 0
    transaction_id: Optional[str] = None
    risk_explanation: RiskExplanation = field(default_factory=RiskExplanation)

    @property
    def risk_level(self) -> RiskLevel:
        return self.risk_explanation.risk_level

    @property
    def top_factors(self) -> List[Dict[str, Any]]:
        return self.risk_explanation.top_factors


@dataclass
class BatchPredictionResult:
    total_count: int = 0
    fraud_count: int = 0
    fraud_rate: float = 0.0
    predictions: List[PredictionResult] = field(default_factory=list)
    process_status: ProcessStatus = ProcessStatus.SUCCESS
    error_reason: Optional[str] = None


@dataclass
class UnifiedPredictionResponse:
    model_version: ModelVersionInfo = field(default_factory=ModelVersionInfo)
    validation: ValidationStatus = field(default_factory=ValidationStatus)
    single_prediction: Optional[PredictionResult] = None
    batch_prediction: Optional[BatchPredictionResult] = None
