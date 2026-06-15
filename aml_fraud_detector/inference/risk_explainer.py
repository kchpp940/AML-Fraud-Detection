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
        self._training_stats: Dict[str, Any] = {}
        self._initialized = False
        if artifacts is not None:
            self.bind(artifacts)

    def bind(self, artifacts: ModelArtifacts) -> None:
        self._feature_metadata = artifacts.feature_metadata or {}
        self._feature_labels = self._feature_metadata.get("feature_labels", {})
        self._training_stats = self._feature_metadata.get("training_stats", {})
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

    def _build_top_factors(
        self, transaction: Dict[str, Any], fraud_prob: float
    ) -> List[Dict[str, Any]]:
        factors: List[Dict[str, Any]] = []
        try:
            amount = float(transaction.get("amount_received", 0.0))
            amount_stats = self._training_stats.get("amount_received", {})
            amount_q75 = float(amount_stats.get("q75", 0.0))
            amount_mean = float(amount_stats.get("mean", 0.0))

            if amount_q75 > 0 and amount > amount_q75 * 1.5:
                severity = min(1.0, amount / (amount_q75 * 1.5))
                factors.append(
                    {
                        "feature": "amount_received",
                        "display_name": self._feature_display_name("amount_received"),
                        "value": amount,
                        "impact": "High",
                        "contribution_pct": round(severity * 35, 1),
                    }
                )
            elif amount_mean > 0 and amount > amount_mean * 2:
                factors.append(
                    {
                        "feature": "amount_received",
                        "display_name": self._feature_display_name("amount_received"),
                        "value": amount,
                        "impact": "Medium",
                        "contribution_pct": 15.0,
                    }
                )

            pf = str(transaction.get("payment_format", ""))
            pf_stats = self._training_stats.get("payment_format", {})
            pf_top5 = pf_stats.get("top5", {})
            if pf in pf_top5:
                total_count = sum(pf_top5.values())
                pf_ratio = pf_top5[pf] / total_count if total_count > 0 else 0
                if pf in {"Wire", "ACH"}:
                    factors.append(
                        {
                            "feature": "payment_format",
                            "display_name": self._feature_display_name("payment_format"),
                            "value": pf,
                            "impact": "Medium",
                            "contribution_pct": round(pf_ratio * 12, 1),
                        }
                    )

            receiving = str(transaction.get("receiving_currency", ""))
            payment = str(transaction.get("payment_currency", ""))
            if receiving and payment and receiving != payment:
                factors.append(
                    {
                        "feature": "currency_mismatch",
                        "display_name": "币种不一致",
                        "value": f"{payment} -> {receiving}",
                        "impact": "Medium",
                        "contribution_pct": 10.0,
                    }
                )

            account = str(transaction.get("account", ""))
            account_stats = self._training_stats.get("account", {})
            account_top5 = account_stats.get("top5", {})
            if account and account in account_top5:
                n_unique = account_stats.get("n_unique", 1)
                if n_unique > 0:
                    freq_ratio = account_top5[account] / sum(account_top5.values()) if account_top5 else 0
                    factors.append(
                        {
                            "feature": "account",
                            "display_name": self._feature_display_name("account"),
                            "value": account,
                            "impact": "Low",
                            "contribution_pct": round(freq_ratio * 8, 1),
                        }
                    )

        except Exception as e:
            logging.warning(f"Error during top_factors extraction: {e}")

        factors.sort(key=lambda x: x.get("contribution_pct", 0.0), reverse=True)
        return factors[:5]

    def explain_from_row(
        self, row: pd.Series, fraud_probability: float, prediction: int
    ) -> RiskExplanation:
        self._ensure_initialized()
        try:
            tx = row.to_dict()
            risk_level = self._risk_level(fraud_probability)
            top_factors = self._build_top_factors(tx, fraud_probability)
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
            logging.info(f"RiskExplainer generated {n} explanation(s)")
            return explanations
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)
