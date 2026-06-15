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
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    top_contributors: List[Dict[str, Any]] = field(default_factory=list)
    summary_text: str = ""


@dataclass
class PredictionResult:
    prediction: int
    fraud_probability: float
    legit_probability: float
    class_label: str
    explanation: Optional[RiskExplanation] = None
    transaction_id: Optional[str] = None
    model_version: Optional[str] = None


@dataclass
class BatchPredictionResult:
    results: List[PredictionResult]
    total_count: int = 0
    fraud_count: int = 0
    legit_count: int = 0
    fraud_rate: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {
                "prediction": r.prediction,
                "fraud_probability": r.fraud_probability,
                "legit_probability": r.legit_probability,
                "class_label": r.class_label,
                "model_version": r.model_version,
            }
            if r.transaction_id is not None:
                row["transaction_id"] = r.transaction_id
            if r.explanation is not None:
                row["risk_level"] = r.explanation.risk_level
                row["summary_text"] = r.explanation.summary_text
            rows.append(row)
        return pd.DataFrame(rows)


@dataclass
class ModelArtifacts:
    model: Any
    preprocessor: Any
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    feature_metadata: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)


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
