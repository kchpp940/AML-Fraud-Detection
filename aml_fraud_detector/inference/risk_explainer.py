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
    PredictionResult,
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
        self._model: Any = None
        self._feature_metadata: Dict[str, Any] = {}
        self._feature_labels: Dict[str, str] = {}
        self._training_stats: Dict[str, Any] = {}
        self._numerical_features: List[str] = []
        self._categorical_features: List[str] = []
        self._initialized = False
        if artifacts is not None:
            self.bind(artifacts)

    def bind(self, artifacts: ModelArtifacts) -> None:
        self._model = artifacts.model
        self._feature_metadata = artifacts.feature_metadata or {}
        self._feature_labels = self._feature_metadata.get("feature_labels", {})
        self._training_stats = self._feature_metadata.get("training_stats", {})
        self._numerical_features = list(self._feature_metadata.get("numerical_features", []))
        self._categorical_features = list(self._feature_metadata.get("categorical_features", []))
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

    def _build_rule_based_contributors(
        self, transaction: Dict[str, Any], fraud_prob: float
    ) -> List[Dict[str, Any]]:
        contributors: List[Dict[str, Any]] = []
        try:
            amount = float(transaction.get("amount_received", 0.0))
            amount_stats = self._training_stats.get("amount_received", {})
            amount_q75 = float(amount_stats.get("q75", amount or 0.0))
            amount_mean = float(amount_stats.get("mean", 0.0))
            if amount_q75 > 0 and amount > amount_q75 * 1.5:
                severity = min(1.0, amount / (amount_q75 * 1.5))
                contributors.append(
                    {
                        "feature": "amount_received",
                        "display_name": self._feature_display_name("amount_received"),
                        "value": amount,
                        "impact": "High",
                        "contribution_pct": round(severity * 35, 1),
                        "reason": (
                            f"交易金额显著偏高 ({amount:.2f}, "
                            f"高于上四分位数的 1.5 倍 {(amount_q75 * 1.5):.2f})"
                        ),
                    }
                )
            elif amount_mean > 0 and amount > amount_mean * 2:
                contributors.append(
                    {
                        "feature": "amount_received",
                        "display_name": self._feature_display_name("amount_received"),
                        "value": amount,
                        "impact": "Medium",
                        "contribution_pct": 15.0,
                        "reason": "交易金额高于均值 2 倍以上",
                    }
                )

            pf = str(transaction.get("payment_format", ""))
            if pf in {"Wire", "ACH"}:
                contributors.append(
                    {
                        "feature": "payment_format",
                        "display_name": self._feature_display_name("payment_format"),
                        "value": pf,
                        "impact": "Medium",
                        "contribution_pct": 12.0,
                        "reason": f"{pf} 支付方式的欺诈交易占比相对较高",
                    }
                )

            receiving = str(transaction.get("receiving_currency", ""))
            payment = str(transaction.get("payment_currency", ""))
            if receiving and payment and receiving != payment:
                contributors.append(
                    {
                        "feature": "currency_mismatch",
                        "display_name": "币种不一致",
                        "value": f"{payment} -> {receiving}",
                        "impact": "Medium",
                        "contribution_pct": 10.0,
                        "reason": "付款币种与收款币种不同，需进一步核查",
                    }
                )
        except Exception as e:
            logging.warning(f"Error during rule-based contributor extraction: {e}")

        contributors.sort(key=lambda x: x.get("contribution_pct", 0.0), reverse=True)
        return contributors[:5]

    def explain_from_row(
        self, row: pd.Series, fraud_probability: float, prediction: int
    ) -> RiskExplanation:
        self._ensure_initialized()
        try:
            tx = row.to_dict()
            is_fraud = int(prediction) == FRAUD_LABEL
            risk_level = self._risk_level(fraud_probability)
            contributors = self._build_rule_based_contributors(tx, fraud_probability)
            if is_fraud:
                if fraud_probability >= 0.9:
                    verb = "极高度"
                elif fraud_probability >= 0.7:
                    verb = "高度"
                elif fraud_probability >= 0.5:
                    verb = "中度"
                else:
                    verb = "轻度"
                summary = (
                    f"该笔交易被判定为{verb}可疑欺诈（置信度 {fraud_probability * 100:.1f}%），"
                    f"风险等级：{risk_level}。"
                )
            else:
                if fraud_probability <= 0.05:
                    verb = "极低"
                elif fraud_probability <= 0.15:
                    verb = "较低"
                else:
                    verb = "中等"
                summary = (
                    f"该笔交易欺诈风险{verb}（欺诈概率 {fraud_probability * 100:.1f}%），"
                    f"风险等级：{risk_level}。"
                )
            if contributors:
                top_names = "、".join(c["display_name"] for c in contributors[:3])
                summary += f"主要影响因素：{top_names}。"
            return RiskExplanation(
                is_fraud=is_fraud,
                fraud_probability=float(fraud_probability),
                risk_level=risk_level,
                top_contributors=contributors,
                summary_text=summary,
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
