import sys
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import (
    AMLException,
    ExplanationException,
    wrap_exception,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.logger import logging
from aml_fraud_detector.entity import RiskExplanation, RiskLevel


FEATURE_LABELS = {
    "amount_received": "交易金额",
    "account": "发起账户",
    "account_1": "接收账户",
    "payment_format": "支付方式",
    "day": "交易星期",
    "from_bank": "发起银行",
    "to_bank": "接收银行",
    "receiving_currency": "接收币种",
    "payment_currency": "支付币种",
}

EXPLANATION_SUPPORTED_MODELS = {
    "XGBClassifier", "XGBoostModel", "RandomForestClassifier",
    "GradientBoostingClassifier", "AdaBoostClassifier",
    "LogisticRegression", "DecisionTreeClassifier",
}


def _risk_level_from_prob(fraud_prob: float) -> RiskLevel:
    if fraud_prob >= 0.9:
        return RiskLevel.CRITICAL
    if fraud_prob >= 0.7:
        return RiskLevel.HIGH
    if fraud_prob >= 0.5:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _is_explainable(model: Any) -> bool:
    model_name = type(model).__name__
    if model_name in EXPLANATION_SUPPORTED_MODELS:
        return True
    if hasattr(model, "feature_importances_"):
        return True
    if hasattr(model, "coef_"):
        return True
    return False


class RiskExplainer:
    def __init__(
        self,
        model: Any = None,
        preprocessor: Any = None,
        feature_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_metadata = feature_metadata or {}
        self._labels = self.feature_metadata.get("feature_labels", FEATURE_LABELS)
        self._original_features = self.feature_metadata.get("original_features", list(FEATURE_LABELS.keys()))

    def explain(
        self,
        transformed_features: np.ndarray,
        fraud_prob: float,
        input_df: Optional[pd.DataFrame] = None,
        top_k: int = 3,
    ) -> RiskExplanation:
        try:
            factors = self._compute_top_factors(transformed_features, input_df, top_k=top_k)
            return RiskExplanation(
                fraud_probability=float(fraud_prob),
                risk_level=_risk_level_from_prob(float(fraud_prob)),
                top_factors=factors,
            )
        except AMLException:
            raise
        except Exception as e:
            raise ExplanationException(
                ErrorCode.EXPLANATION_FAILED,
                error_details=sys,
                detail=str(e),
            )

    def _compute_top_factors(
        self,
        transformed_features: np.ndarray,
        input_df: Optional[pd.DataFrame],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        model = self.model
        if model is None:
            raise ExplanationException(
                ErrorCode.EXPLANATION_NOT_SUPPORTED,
                error_details=sys,
            )

        model_name = type(model).__name__
        try:
            importances = None
            if hasattr(model, "feature_importances_"):
                importances = np.asarray(model.feature_importances_, dtype=float).ravel()
            elif hasattr(model, "coef_"):
                importances = np.abs(np.asarray(model.coef_, dtype=float).ravel())

            if importances is None:
                raise ExplanationException(
                    ErrorCode.EXPLANATION_MODEL_INCOMPATIBLE,
                    error_details=sys,
                    model_type=model_name,
                )

            if transformed_features is None:
                raise ExplanationException(
                    ErrorCode.EXPLANATION_FAILED,
                    error_details=sys,
                    detail="transformed_features is None",
                )

            row = np.asarray(transformed_features).ravel()
            if row.shape[0] != importances.shape[0]:
                min_len = min(row.shape[0], importances.shape[0])
                row = row[:min_len]
                importances = importances[:min_len]

            score = np.abs(row) * importances
            total = score.sum()
            if total <= 0 or not np.isfinite(total):
                score = importances
                total = score.sum()
                if total <= 0 or not np.isfinite(total):
                    return []

            top_idx = np.argsort(score)[-top_k:][::-1]

            factors: List[Dict[str, Any]] = []
            for idx in top_idx:
                feat_val = float(row[idx])
                cont = float(score[idx] / total)
                orig_label = self._feature_label_for_index(idx, input_df)
                direction = "push_fraud" if feat_val > 0 else "push_legit"
                factors.append({
                    "feature": orig_label,
                    "contribution": round(cont, 4),
                    "value": round(feat_val, 6),
                    "direction": direction,
                })
            return factors
        except ExplanationException:
            raise
        except Exception as e:
            raise wrap_exception(e, error_details=sys)

    def _feature_label_for_index(self, idx: int, input_df: Optional[pd.DataFrame]) -> str:
        if input_df is not None and idx < len(input_df.columns):
            col = input_df.columns.tolist()[idx]
            return self._labels.get(col, col)
        for orig in self._original_features:
            if str(idx) in orig:
                return self._labels.get(orig, orig)
        fallback = f"feature_{idx}"
        for orig in self._original_features:
            if idx < 5:
                return self._labels.get(orig, orig)
        return fallback
