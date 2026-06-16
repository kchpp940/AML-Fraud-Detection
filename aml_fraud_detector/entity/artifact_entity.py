from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class ProcessStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class RiskLevel(str, Enum):
    MINIMAL = "Minimal"
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
    risk_level: RiskLevel = RiskLevel.MINIMAL
    summary_text: str = ""
    top_contributors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ModelVersionInfo:
    model_version: int = 0
    best_model_name: str = ""
    training_time: str = ""
    selection_metric: str = ""
    best_metric_value: float = 0.0
    feature_contract_version: str = ""
    artifact_path: str = ""


@dataclass
class ValidationStatus:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    prediction: int = -1
    class_label: str = ""
    fraud_probability: float = 0.0
    legit_probability: float = 0.0
    model_version: int = 0
    transaction_id: Optional[str] = None
    process_status: ProcessStatus = ProcessStatus.SUCCESS
    error_reason: Optional[str] = None
    risk_explanation: RiskExplanation = field(default_factory=RiskExplanation)

    @property
    def risk_level(self) -> RiskLevel:
        return self.risk_explanation.risk_level

    @property
    def risk_summary(self) -> str:
        return self.risk_explanation.summary_text

    @property
    def top_contributors(self) -> List[Dict[str, Any]]:
        return self.risk_explanation.top_contributors


@dataclass
class BatchPredictionResult:
    total_count: int = 0
    fraud_count: int = 0
    legit_count: int = 0
    fraud_rate: float = 0.0
    process_status: ProcessStatus = ProcessStatus.SUCCESS
    error_reason: Optional[str] = None
    predictions: List[PredictionResult] = field(default_factory=list)


@dataclass
class UnifiedArtifactsView:
    model_version: ModelVersionInfo = field(default_factory=ModelVersionInfo)
    validation: ValidationStatus = field(default_factory=ValidationStatus)
