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

    FRAUD_CLASS_INDEX = 1
    LEGIT_CLASS_INDEX = 0

    def __init__(self, artifacts: Optional[ModelArtifacts] = None):
        self._model: Any = None
        self._preprocessor: Any = None
        self._feature_metadata: Dict[str, Any] = {}
        self._feature_labels: Dict[str, str] = {}
        self._baseline_values: Dict[str, Any] = {}
        self._all_features: List[str] = []
        self._baseline_df: Optional[pd.DataFrame] = None
        self._baseline_pred_proba: Optional[np.ndarray] = None
        self._initialized = False
        if artifacts is not None:
            self.bind(artifacts)

    def bind(self, artifacts: ModelArtifacts) -> None:
        self._model = artifacts.model
        self._preprocessor = artifacts.preprocessor
        self._feature_metadata = artifacts.feature_metadata or {}
        self._feature_labels = self._feature_metadata.get("feature_labels", {})
        self._baseline_values = self._feature_metadata.get("baseline_values", {})
        self._all_features = list(self._feature_metadata.get("original_features", []))

        if self._model is None or self._preprocessor is None:
            raise CustomerException(
                RuntimeError("RiskExplainer requires both model and preprocessor to compute marginal contributions"),
                sys,
            )
        if not self._baseline_values:
            raise CustomerException(
                RuntimeError("feature_metadata.json missing 'baseline_values' section required for marginal contribution calculation"),
                sys,
            )

        baseline_row = {f: self._baseline_values[f] for f in self._all_features}
        self._baseline_df = pd.DataFrame([baseline_row])

        try:
            baseline_x = self._preprocessor.transform(self._baseline_df)
            if hasattr(baseline_x, "toarray"):
                baseline_x = baseline_x.toarray()
            self._baseline_pred_proba = self._model.predict_proba(baseline_x)[0]
        except Exception as e:
            raise CustomerException(
                RuntimeError(f"Failed to compute baseline prediction for RiskExplainer: {e}"),
                sys,
            )

        self._initialized = True
        logging.info(
            f"RiskExplainer bound: baseline_fraud_prob={self._baseline_pred_proba[self.FRAUD_CLASS_INDEX]:.4f}, "
            f"features={self._all_features}"
        )

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

    def _impact_label(self, abs_contribution: float) -> str:
        if abs_contribution >= 0.20:
            return "Critical"
        elif abs_contribution >= 0.10:
            return "High"
        elif abs_contribution >= 0.05:
            return "Medium"
        elif abs_contribution >= 0.01:
            return "Low"
        else:
            return "Minimal"

    def _predict_row(self, row_df: pd.DataFrame) -> np.ndarray:
        x = self._preprocessor.transform(row_df)
        if hasattr(x, "toarray"):
            x = x.toarray()
        return self._model.predict_proba(x)[0]

    def _compute_marginal_contributions(
        self, row: pd.Series
    ) -> List[Dict[str, Any]]:
        baseline_fraud_prob = float(self._baseline_pred_proba[self.FRAUD_CLASS_INDEX])
        contributions: List[Dict[str, Any]] = []

        for feature in self._all_features:
            single_change_row = self._baseline_df.copy()
            single_change_row.at[0, feature] = row[feature]

            try:
                pred_proba = self._predict_row(single_change_row)
                fraud_prob = float(pred_proba[self.FRAUD_CLASS_INDEX])
                marginal_contribution = fraud_prob - baseline_fraud_prob
            except Exception as e:
                logging.warning(f"Failed to compute marginal contribution for {feature}: {e}")
                marginal_contribution = 0.0
                fraud_prob = baseline_fraud_prob

            impact = self._impact_label(abs(marginal_contribution))
            contribution_pct = round(abs(marginal_contribution) * 100, 2)

            contributions.append(
                {
                    "feature": feature,
                    "display_name": self._feature_display_name(feature),
                    "value": row[feature],
                    "baseline_value": self._baseline_values[feature],
                    "fraud_prob_with_feature": fraud_prob,
                    "baseline_fraud_prob": baseline_fraud_prob,
                    "marginal_contribution": marginal_contribution,
                    "contribution_pct": contribution_pct,
                    "impact": impact,
                    "direction": "increasing" if marginal_contribution > 0 else "decreasing",
                }
            )

        contributions.sort(key=lambda x: abs(x["marginal_contribution"]), reverse=True)
        return contributions

    def explain_from_row(
        self, row: pd.Series, fraud_probability: float, prediction: int
    ) -> RiskExplanation:
        self._ensure_initialized()
        try:
            risk_level = self._risk_level(fraud_probability)
            all_contributions = self._compute_marginal_contributions(row)

            top_factors = []
            total_abs = sum(abs(c["marginal_contribution"]) for c in all_contributions)
            for c in all_contributions:
                if total_abs > 0:
                    relative_pct = round(abs(c["marginal_contribution"]) / total_abs * 100, 1)
                else:
                    relative_pct = 0.0
                top_factors.append(
                    {
                        "feature": c["feature"],
                        "display_name": c["display_name"],
                        "value": c["value"],
                        "baseline_value": c["baseline_value"],
                        "marginal_contribution": c["marginal_contribution"],
                        "contribution_pct": relative_pct,
                        "impact": c["impact"],
                        "direction": c["direction"],
                    }
                )

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
                row = aligned_df.iloc[i]
                fraud_prob = float(proba[i, 1])
                explanations.append(
                    self.explain_from_row(row, fraud_prob, int(preds[i]))
                )
            logging.info(f"RiskExplainer generated {n} explanation(s) using marginal contributions")
            return explanations
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)
