import sys
import os
import json
import uuid
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.entity import (
    ProcessStatus,
    RiskLevel,
    REQUIRED_INPUT_FIELDS,
    FOUR_ARTIFACT_FILENAMES,
    RiskExplanation,
    ModelVersionInfo,
    ValidationStatus,
    PredictionResult,
    BatchPredictionResult,
)


class RiskExplainer:
    @staticmethod
    def explain(fraud_probability: float) -> RiskExplanation:
        if fraud_probability >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif fraud_probability >= 0.6:
            risk_level = RiskLevel.HIGH
        elif fraud_probability >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        return RiskExplanation(
            fraud_probability=fraud_probability,
            risk_level=risk_level,
            top_factors=[],
        )


class ArtifactValidator:
    @staticmethod
    def validate(artifacts_dir: str = "artifacts") -> ValidationStatus:
        status = ValidationStatus(checked_artifacts=list(FOUR_ARTIFACT_FILENAMES))
        for fname in FOUR_ARTIFACT_FILENAMES:
            fpath = os.path.join(artifacts_dir, fname)
            if not os.path.exists(fpath):
                status.errors.append(f"Missing artifact: {fname}")
                status.is_valid = False
        return status


class PredictionPipeline:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self._model = None
        self._preprocessor = None
        self._model_metadata: Optional[Dict[str, Any]] = None
        self._feature_metadata: Optional[Dict[str, Any]] = None

    def _load_artifacts(self):
        if self._model is None:
            model_path = os.path.join(self.artifacts_dir, "model.pkl")
            preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")
            self._model = load_object(file_path=model_path)
            self._preprocessor = load_object(file_path=preprocessor_path)
        if self._model_metadata is None:
            mm_path = os.path.join(self.artifacts_dir, "model_metadata.json")
            if os.path.exists(mm_path):
                with open(mm_path, "r", encoding="utf-8") as f:
                    self._model_metadata = json.load(f)
        if self._feature_metadata is None:
            fm_path = os.path.join(self.artifacts_dir, "feature_metadata.json")
            if os.path.exists(fm_path):
                with open(fm_path, "r", encoding="utf-8") as f:
                    self._feature_metadata = json.load(f)

    def validate_artifacts(self) -> ValidationStatus:
        return ArtifactValidator.validate(self.artifacts_dir)

    def get_model_version_info(self) -> ModelVersionInfo:
        self._load_artifacts()
        info = ModelVersionInfo()
        if self._model_metadata:
            info.model_version = int(self._model_metadata.get("model_version", 0))
            info.model_name = str(self._model_metadata.get("best_model_name", ""))
            info.training_time = str(self._model_metadata.get("training_time", ""))
            info.selection_metric = str(self._model_metadata.get("selection_metric", ""))
            info.best_metric_value = float(self._model_metadata.get("best_metric_value", 0.0))
            info.feature_schema_version = str(self._model_metadata.get("feature_schema_version", ""))
            info.artifact_path = str(self._model_metadata.get("artifact_path", ""))
        return info

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        try:
            self._load_artifacts()
            data_scaled = self._preprocessor.transform(features)
            predictions = self._model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomerException(e, sys)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        try:
            self._load_artifacts()
            data_scaled = self._preprocessor.transform(features)
            predictions_prob = self._model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            raise CustomerException(e, sys)

    def _build_prediction_result(
        self,
        row_idx: int,
        pred_class: int,
        probs: np.ndarray,
        model_version: int,
        transaction_id: Optional[str] = None,
    ) -> PredictionResult:
        legit_prob = float(probs[0]) if len(probs) > 0 else 0.0
        fraud_prob = float(probs[1]) if len(probs) > 1 else 0.0
        class_label = "Fraud" if pred_class == 1 else "Not Fraud"
        explanation = RiskExplainer.explain(fraud_prob)
        return PredictionResult(
            prediction=int(pred_class),
            fraud_probability=fraud_prob,
            legit_probability=legit_prob,
            class_label=class_label,
            process_status=ProcessStatus.SUCCESS,
            error_reason=None,
            model_version=model_version,
            transaction_id=transaction_id,
            risk_explanation=explanation,
        )

    def _validate_input_dict(self, data: Dict[str, Any]) -> Optional[str]:
        for field in REQUIRED_INPUT_FIELDS:
            if field not in data:
                return f"Input dict missing required field: '{field}'"
        return None

    def predict_single(self, data: Dict[str, Any]) -> PredictionResult:
        try:
            err = self._validate_input_dict(data)
            if err is not None:
                return PredictionResult(
                    prediction=-1,
                    process_status=ProcessStatus.ERROR,
                    error_reason=err,
                )
            self._load_artifacts()
            df = pd.DataFrame([data])
            proba_result = self.predict_proba(df)
            pred_result = self.predict(df)
            mv = self.get_model_version_info().model_version
            tx_id = data.get("transaction_id") or str(uuid.uuid4())
            return self._build_prediction_result(0, int(pred_result[0]), proba_result[0], mv, tx_id)
        except Exception as e:
            return PredictionResult(
                prediction=-1,
                process_status=ProcessStatus.ERROR,
                error_reason=str(e),
            )

    def predict_batch(self, df: pd.DataFrame, explain: bool = True) -> BatchPredictionResult:
        try:
            batch = BatchPredictionResult()
            self._load_artifacts()
            mv = self.get_model_version_info().model_version
            proba_all = self.predict_proba(df)
            pred_all = self.predict(df)
            results: List[PredictionResult] = []
            for i in range(len(df)):
                tx_id = None
                if "transaction_id" in df.columns:
                    tx_id = str(df.iloc[i]["transaction_id"])
                pr = self._build_prediction_result(i, int(pred_all[i]), proba_all[i], mv, tx_id)
                results.append(pr)
            batch.total_count = len(results)
            batch.fraud_count = sum(1 for r in results if r.prediction == 1)
            batch.fraud_rate = (batch.fraud_count / batch.total_count) if batch.total_count > 0 else 0.0
            batch.predictions = results
            return batch
        except Exception as e:
            return BatchPredictionResult(
                process_status=ProcessStatus.ERROR,
                error_reason=str(e),
            )

    def to_flat_dataframe(self, batch: BatchPredictionResult) -> pd.DataFrame:
        rows = []
        for p in batch.predictions:
            rows.append({
                "prediction": p.prediction,
                "fraud_probability": p.fraud_probability,
                "legit_probability": p.legit_probability,
                "class_label": p.class_label,
                "process_status": p.process_status.value if isinstance(p.process_status, ProcessStatus) else p.process_status,
                "error_reason": p.error_reason,
                "model_version": p.model_version,
                "transaction_id": p.transaction_id,
                "risk_level": p.risk_level.value if isinstance(p.risk_level, RiskLevel) else p.risk_level,
                "top_factors": p.top_factors,
            })
        return pd.DataFrame(rows)


class CustomData:
    def __init__(self,
            from_bank: int,
            account: str,
            to_bank: int,
            account_1: str,
            amount_received: float,
            receiving_currency: str,
            payment_currency: str,
            payment_format: str,
            day: str):

        self.from_bank = from_bank
        self.account = account
        self.to_bank = to_bank
        self.account_1 = account_1
        self.amount_received = amount_received
        self.receiving_currency = receiving_currency
        self.payment_currency = payment_currency
        self.payment_format = payment_format
        self.day = day

    def get_data_as_DataFrame(self):
        try:
            custom_data_input_dict = {
                "from_bank": [self.from_bank],
                "account": [self.account],
                "to_bank": [self.to_bank],
                "account_1": [self.account_1],
                "amount_received": [self.amount_received],
                "receiving_currency": [self.receiving_currency],
                "payment_currency": [self.payment_currency],
                "payment_format": [self.payment_format],
                "day": [self.day]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomerException(e, sys)

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
