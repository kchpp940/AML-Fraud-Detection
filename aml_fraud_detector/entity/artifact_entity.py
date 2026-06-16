from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum

from aml_fraud_detector.exception import ErrorDetail, UnifiedErrorResponse


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
    error_detail: Optional[ErrorDetail] = None
    model_version: int = 0
    transaction_id: Optional[str] = None
    risk_explanation: RiskExplanation = field(default_factory=RiskExplanation)

    @property
    def risk_level(self) -> RiskLevel:
        return self.risk_explanation.risk_level

    @property
    def top_factors(self) -> List[Dict[str, Any]]:
        return self.risk_explanation.top_factors

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.error_detail:
            result["error_detail"] = self.error_detail.to_dict()
        return result

    def is_success(self) -> bool:
        return self.process_status == ProcessStatus.SUCCESS

    @classmethod
    def from_error(cls, error_detail: ErrorDetail, transaction_id: Optional[str] = None) -> "PredictionResult":
        return cls(
            process_status=ProcessStatus.ERROR,
            error_reason=error_detail.message,
            error_detail=error_detail,
            transaction_id=transaction_id,
        )


@dataclass
class BatchPredictionResult:
    total_count: int = 0
    fraud_count: int = 0
    fraud_rate: float = 0.0
    predictions: List[PredictionResult] = field(default_factory=list)
    process_status: ProcessStatus = ProcessStatus.SUCCESS
    error_reason: Optional[str] = None
    error_detail: Optional[ErrorDetail] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "total_count": self.total_count,
            "fraud_count": self.fraud_count,
            "fraud_rate": self.fraud_rate,
            "predictions": [p.to_dict() for p in self.predictions],
            "process_status": self.process_status.value,
            "error_reason": self.error_reason,
        }
        if self.error_detail:
            result["error_detail"] = self.error_detail.to_dict()
        return result

    def is_success(self) -> bool:
        return self.process_status == ProcessStatus.SUCCESS

    @classmethod
    def from_error(cls, error_detail: ErrorDetail) -> "BatchPredictionResult":
        return cls(
            process_status=ProcessStatus.ERROR,
            error_reason=error_detail.message,
            error_detail=error_detail,
        )


@dataclass
class TrainingPipelineResult:
    process_status: ProcessStatus = ProcessStatus.SUCCESS
    error_reason: Optional[str] = None
    error_detail: Optional[ErrorDetail] = None
    summary: Optional[Dict[str, Any]] = None
    artifacts: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "process_status": self.process_status.value,
            "error_reason": self.error_reason,
            "summary": self.summary,
            "artifacts": self.artifacts,
        }
        if self.error_detail:
            result["error_detail"] = self.error_detail.to_dict()
        return result

    def is_success(self) -> bool:
        return self.process_status == ProcessStatus.SUCCESS

    @classmethod
    def from_error(cls, error_detail: ErrorDetail) -> "TrainingPipelineResult":
        return cls(
            process_status=ProcessStatus.ERROR,
            error_reason=error_detail.message,
            error_detail=error_detail,
        )


@dataclass
class UnifiedPredictionResponse:
    model_version: ModelVersionInfo = field(default_factory=ModelVersionInfo)
    validation: ValidationStatus = field(default_factory=ValidationStatus)
    single_prediction: Optional[PredictionResult] = None
    batch_prediction: Optional[BatchPredictionResult] = None
    error: Optional[UnifiedErrorResponse] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "model_version": asdict(self.model_version),
            "validation": asdict(self.validation),
        }
        if self.single_prediction:
            result["single_prediction"] = self.single_prediction.to_dict()
        if self.batch_prediction:
            result["batch_prediction"] = self.batch_prediction.to_dict()
        if self.error:
            result["error"] = self.error.to_dict()
        return result

    def is_success(self) -> bool:
        if self.error:
            return False
        if self.single_prediction and not self.single_prediction.is_success():
            return False
        if self.batch_prediction and not self.batch_prediction.is_success():
            return False
        return True
