import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.inference.contracts import (
    FRAUD_LABEL,
    LEGIT_LABEL,
    ModelArtifacts,
    RiskExplanation,
)


class RiskExplainer:
    RISK_LEVELS = [
        (0.75, "Critical"),
        (0.50, "High"),
        (0.25, "Medium"),
        (0.05, "Low"),
    ]

    def __init__(self, artifacts: Optional[ModelArtifacts] = None):
        self._feature_metadata: Dict[str, Any] = {}
        self._feature_labels: Dict[str, str] = {}
        self._initialized = False
        if artifacts is not None:
            self.bind(artifacts)

    def bind(self, artifacts: ModelArtifacts) -> None:
        self._feature_metadata = artifacts.feature_metadata or {}
        self._feature_labels = self._feature_metadata.get("feature_labels", {})
        self._initialized = True
        logging.info("RiskExplainer bound to artifacts")

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise CustomerException(
                RuntimeError("RiskExplainer is not bound to artifacts; call bind() first"),
                sys,
            )

    def _feature_display_name(self, raw_name: str) -> str:
        return self._feature_labels.get(raw_name, raw_name)

    def _risk_level(self, fraud_probability: float) -> str:
        for threshold, label in self.RISK_LEVELS:
            if fraud_probability >= threshold:
                return label
        return "Minimal"

    def explain_from_prob(
        self, fraud_probability: float, prediction: int
    ) -> RiskExplanation:
        self._ensure_initialized()
        try:
            is_fraud = int(prediction) == FRAUD_LABEL
            risk_level = self._risk_level(fraud_probability)
            top_factors: List[Dict[str, Any]] = []
            return RiskExplanation(
                fraud_probability=float(fraud_probability),
                top_factors=top_factors,
                risk_level=risk_level,
            )
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)

    def explain_from_dataframe(
        self,
        aligned_df: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> List[RiskExplanation]:
        self._ensure_initialized()
        try:
            preds = np.asarray(predictions).astype(int).flatten()
            proba = np.asarray(probabilities)
            n = len(preds)
            if proba.ndim != 2 or proba.shape[0] != n:
                raise CustomerException(
                    ValueError(
                        f"Shape mismatch: predictions={preds.shape}, proba={proba.shape}"
                    ),
                    sys,
                )
            explanations: List[RiskExplanation] = []
            for i in range(n):
                fraud_prob = float(proba[i, 1]) if proba.ndim == 2 else float(proba[i])
                explanations.append(
                    self.explain_from_prob(fraud_prob, int(preds[i]))
                )
            logging.info(f"RiskExplainer generated {n} explanation(s)")
            return explanations
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)
