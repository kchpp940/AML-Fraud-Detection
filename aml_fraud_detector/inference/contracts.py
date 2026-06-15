import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import CustomerException


FRAUD_LABEL = 1
LEGIT_LABEL = 0

REQUIRED_INPUT_FIELDS: List[str] = [
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

PROCESS_STATUS_SUCCESS = "success"
PROCESS_STATUS_ERROR = "error"


@dataclass
class TransactionInput:
    from_bank: int
    account: str
    to_bank: int
    account_1: str
    amount_received: float
    receiving_currency: str
    payment_currency: str
    payment_format: str
    day: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_bank": self.from_bank,
            "account": self.account,
            "to_bank": self.to_bank,
            "account_1": self.account_1,
            "amount_received": self.amount_received,
            "receiving_currency": self.receiving_currency,
            "payment_currency": self.payment_currency,
            "payment_format": self.payment_format,
            "day": self.day,
        }

    def to_dataframe(self) -> pd.DataFrame:
        try:
            return pd.DataFrame([self.to_dict()])
        except Exception as e:
            raise CustomerException(e, sys)


@dataclass
class RiskExplanation:
    fraud_probability: float
    top_factors: List[Dict[str, Any]] = field(default_factory=list)
    risk_level: str = ""

    def to_flat_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "fraud_probability": self.fraud_probability,
            "risk_level": self.risk_level,
            "top_factors": self.top_factors,
        }
        return d


@dataclass
class PredictionResult:
    prediction: int
    fraud_probability: float
    legit_probability: float
    class_label: str
    process_status: str = PROCESS_STATUS_SUCCESS
    error_reason: Optional[str] = None
    risk_explanation: Optional[RiskExplanation] = None
    model_version: Optional[str] = None
    transaction_id: Optional[str] = None

    def to_flat_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "prediction": self.prediction,
            "fraud_probability": self.fraud_probability,
            "legit_probability": self.legit_probability,
            "class_label": self.class_label,
            "process_status": self.process_status,
            "error_reason": self.error_reason,
            "model_version": self.model_version,
            "transaction_id": self.transaction_id,
        }
        if self.risk_explanation is not None:
            d["risk_explanation"] = self.risk_explanation.to_flat_dict()
            d["top_factors"] = self.risk_explanation.top_factors
            d["risk_level"] = self.risk_explanation.risk_level
        else:
            d["risk_explanation"] = None
            d["top_factors"] = None
            d["risk_level"] = None
        return d


@dataclass
class BatchPredictionResult:
    results: List[PredictionResult]
    total_count: int = 0
    fraud_count: int = 0
    legit_count: int = 0
    fraud_rate: float = 0.0

    def to_flat_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            flat = r.to_flat_dict()
            row: Dict[str, Any] = {
                "prediction": flat["prediction"],
                "fraud_probability": flat["fraud_probability"],
                "legit_probability": flat["legit_probability"],
                "class_label": flat["class_label"],
                "process_status": flat["process_status"],
                "error_reason": flat["error_reason"],
                "model_version": flat["model_version"],
                "transaction_id": flat["transaction_id"],
                "risk_level": flat.get("risk_level"),
                "top_factors": flat.get("top_factors"),
            }
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class ModelArtifacts:
    model: Any
    preprocessor: Any
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    feature_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            msg = "Artifact validation failed:\n" + "\n".join(
                f"  - {e}" for e in self.errors
            )
            raise CustomerException(ValueError(msg), sys)
