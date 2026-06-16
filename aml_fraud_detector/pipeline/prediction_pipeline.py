    
import sys
import os
import json
from typing import Dict, List, Optional, Any

import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.utils.feature_normalization import (
    normalize_column_names,
    derive_temporal_features,
    clean_categorical_fields,
    report_missing_fields,
    normalize_column_name,
)
from aml_fraud_detector.entity.artifact_entity import (
    ProcessStatus,
    PredictionResult,
    BatchPredictionResult,
    RiskLevel,
    RiskExplanation,
    REQUIRED_INPUT_FIELDS,
)


class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.feature_metadata_path = os.path.join("artifacts", "feature_metadata.json")
        self._model = None
        self._preprocessor = None
        self._expected_features: Optional[List[str]] = None
        self._numerical_features: Optional[List[str]] = None
        self._categorical_features: Optional[List[str]] = None
        self._model_version: int = 0
        self._load_feature_schema()

    def _load_feature_schema(self) -> None:
        try:
            if not os.path.exists(self.feature_metadata_path):
                logging.warning(
                    f"Feature metadata not found at {self.feature_metadata_path}, "
                    f"will use REQUIRED_INPUT_FIELDS fallback"
                )
                return

            with open(self.feature_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            self._expected_features = metadata.get("original_features")
            self._numerical_features = metadata.get("numerical_features")
            self._categorical_features = metadata.get("categorical_features")

            training_signature = metadata.get("training_signature", "")
            if training_signature:
                try:
                    self._model_version = int(training_signature[:8])
                except ValueError:
                    self._model_version = 0

            if self._expected_features:
                logging.info(
                    f"Loaded feature schema from training artifacts: "
                    f"{len(self._expected_features)} features "
                    f"({len(self._numerical_features or [])} num, "
                    f"{len(self._categorical_features or [])} cat)"
                )
            else:
                logging.warning(
                    "Feature metadata found but 'original_features' is empty"
                )
        except Exception as e:
            logging.warning(
                f"Failed to load feature metadata: {e}, "
                f"will use REQUIRED_INPUT_FIELDS fallback"
            )

    def _load_artifacts(self):
        if self._model is None:
            self._model = load_object(file_path=self.model_path)
        if self._preprocessor is None:
            self._preprocessor = load_object(file_path=self.preprocessor_path)

    def _validate_and_prepare(
        self, features: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        df = features.copy()

        df, _ = normalize_column_names(df)

        if "timestamp" in df.columns:
            df, _ = derive_temporal_features(df)

        df, _ = clean_categorical_fields(df)

        if self._expected_features:
            missing, _ = report_missing_fields(
                df, required_columns=self._expected_features
            )
            if missing:
                return None
            df = df[self._expected_features]

        return df

    def _get_missing_features(self, features: pd.DataFrame) -> List[str]:
        df = features.copy()
        df, _ = normalize_column_names(df)
        if "timestamp" in df.columns:
            df, _ = derive_temporal_features(df)
        expected = self._expected_features or REQUIRED_INPUT_FIELDS
        missing, _ = report_missing_fields(df, required_columns=expected)
        return missing

    def _classify_risk(self, fraud_prob: float) -> RiskLevel:
        if fraud_prob >= 0.8:
            return RiskLevel.CRITICAL
        elif fraud_prob >= 0.6:
            return RiskLevel.HIGH
        elif fraud_prob >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _build_top_factors(
        self, row: pd.Series, fraud_prob: float
    ) -> List[Dict[str, Any]]:
        factors = []
        if pd.notna(row.get("amount_received")) and float(row["amount_received"]) > 10000:
            factors.append({"factor": "amount_received", "weight": "high",
                           "explanation": f"大额交易: ${float(row['amount_received']):,.2f}"})
        if "receiving_currency" in row.index and "payment_currency" in row.index:
            if (row.get("receiving_currency") == row.get("payment_currency")
                    and row.get("receiving_currency") is not None):
                if fraud_prob > 0.5:
                    factors.append({"factor": "currency_pattern", "weight": "medium",
                                   "explanation": "同币种异常模式"})
        return factors

    def predict(self, features):
        try:
            if self._model is None or self._preprocessor is None:
                self._load_artifacts()

            normalized_features = self._validate_and_prepare(features)
            if normalized_features is None:
                missing = self._get_missing_features(features)
                raise CustomerException(
                    ValueError(
                        f"Prediction input is missing features that were present "
                        f"during training: {missing}. "
                        f"Expected (from feature_metadata.json): "
                        f"{self._expected_features or REQUIRED_INPUT_FIELDS}, "
                        f"Got: {list(features.columns)}"
                    ),
                    sys,
                )

            data_scaled = self._preprocessor.transform(normalized_features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            predictions = self._model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomerException(e, sys)
        
    def predict_proba(self, features):
        try:
            if self._model is None or self._preprocessor is None:
                self._load_artifacts()

            normalized_features = self._validate_and_prepare(features)
            if normalized_features is None:
                missing = self._get_missing_features(features)
                raise CustomerException(
                    ValueError(
                        f"Prediction input is missing features that were present "
                        f"during training: {missing}. "
                        f"Expected (from feature_metadata.json): "
                        f"{self._expected_features or REQUIRED_INPUT_FIELDS}, "
                        f"Got: {list(features.columns)}"
                    ),
                    sys,
                )

            data_scaled = self._preprocessor.transform(normalized_features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            predictions_prob = self._model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            raise CustomerException(e, sys)

    def predict_single(
        self,
        features: pd.DataFrame,
        transaction_id: Optional[str] = None,
    ) -> PredictionResult:
        try:
            if self._model is None or self._preprocessor is None:
                self._load_artifacts()

            missing = self._get_missing_features(features)
            if missing:
                return PredictionResult(
                    prediction=-1,
                    fraud_probability=0.0,
                    legit_probability=1.0,
                    class_label="Error",
                    process_status=ProcessStatus.ERROR,
                    error_reason=(
                        f"Missing required features: {missing}. "
                        f"Expected: {self._expected_features or REQUIRED_INPUT_FIELDS}"
                    ),
                    model_version=self._model_version,
                    transaction_id=transaction_id,
                )

            normalized_features = self._validate_and_prepare(features)
            if normalized_features is None:
                return PredictionResult(
                    prediction=-1,
                    fraud_probability=0.0,
                    legit_probability=1.0,
                    class_label="Error",
                    process_status=ProcessStatus.ERROR,
                    error_reason="Feature validation failed after normalization",
                    model_version=self._model_version,
                    transaction_id=transaction_id,
                )

            data_scaled = self._preprocessor.transform(normalized_features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            proba = self._model.predict_proba(data_scaled)
            pred = self._model.predict(data_scaled)

            fraud_prob = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            legit_prob = float(proba[0][0]) if proba.shape[1] > 1 else float(1.0 - fraud_prob)
            prediction = int(pred[0])

            risk_level = self._classify_risk(fraud_prob)
            top_factors = self._build_top_factors(
                normalized_features.iloc[0] if len(normalized_features) > 0 else pd.Series(),
                fraud_prob,
            )

            risk_explanation = RiskExplanation(
                fraud_probability=fraud_prob,
                risk_level=risk_level,
                top_factors=top_factors,
            )

            return PredictionResult(
                prediction=prediction,
                fraud_probability=fraud_prob,
                legit_probability=legit_prob,
                class_label="Fraud" if prediction == 1 else "Not Fraud",
                process_status=ProcessStatus.SUCCESS,
                error_reason=None,
                model_version=self._model_version,
                transaction_id=transaction_id,
                risk_explanation=risk_explanation,
            )

        except Exception as e:
            logging.error(f"predict_single failed: {e}", exc_info=True)
            return PredictionResult(
                prediction=-1,
                fraud_probability=0.0,
                legit_probability=1.0,
                class_label="Error",
                process_status=ProcessStatus.ERROR,
                error_reason=str(e),
                model_version=self._model_version,
                transaction_id=transaction_id,
            )

    def predict_batch(
        self,
        features: pd.DataFrame,
    ) -> BatchPredictionResult:
        try:
            if self._model is None or self._preprocessor is None:
                self._load_artifacts()

            expected = self._expected_features or REQUIRED_INPUT_FIELDS
            missing_cols = [c for c in expected if c not in features.columns]
            if missing_cols:
                return BatchPredictionResult(
                    total_count=len(features),
                    fraud_count=0,
                    fraud_rate=0.0,
                    predictions=[],
                    process_status=ProcessStatus.ERROR,
                    error_reason=(
                        f"Batch input missing required features: {missing_cols}. "
                        f"Expected: {expected}"
                    ),
                )

            normalized_features = self._validate_and_prepare(features)
            if normalized_features is None:
                missing = self._get_missing_features(features)
                return BatchPredictionResult(
                    total_count=len(features),
                    fraud_count=0,
                    fraud_rate=0.0,
                    predictions=[],
                    process_status=ProcessStatus.ERROR,
                    error_reason=(
                        f"Batch feature validation failed: missing {missing}"
                    ),
                )

            data_scaled = self._preprocessor.transform(normalized_features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            proba = self._model.predict_proba(data_scaled)
            preds = self._model.predict(data_scaled)

            results: List[PredictionResult] = []
            fraud_count = 0

            for i in range(len(features)):
                fraud_prob = float(proba[i][1]) if proba.shape[1] > 1 else float(proba[i][0])
                legit_prob = float(proba[i][0]) if proba.shape[1] > 1 else float(1.0 - fraud_prob)
                prediction = int(preds[i])

                if prediction == 1:
                    fraud_count += 1

                risk_level = self._classify_risk(fraud_prob)
                top_factors = self._build_top_factors(
                    normalized_features.iloc[i] if len(normalized_features) > i else pd.Series(),
                    fraud_prob,
                )

                risk_explanation = RiskExplanation(
                    fraud_probability=fraud_prob,
                    risk_level=risk_level,
                    top_factors=top_factors,
                )

                results.append(PredictionResult(
                    prediction=prediction,
                    fraud_probability=fraud_prob,
                    legit_probability=legit_prob,
                    class_label="Fraud" if prediction == 1 else "Not Fraud",
                    process_status=ProcessStatus.SUCCESS,
                    error_reason=None,
                    model_version=self._model_version,
                    transaction_id=None,
                    risk_explanation=risk_explanation,
                ))

            total = len(features)
            fraud_rate = fraud_count / total if total > 0 else 0.0

            return BatchPredictionResult(
                total_count=total,
                fraud_count=fraud_count,
                fraud_rate=fraud_rate,
                predictions=results,
                process_status=ProcessStatus.SUCCESS,
                error_reason=None,
            )

        except Exception as e:
            logging.error(f"predict_batch failed: {e}", exc_info=True)
            return BatchPredictionResult(
                total_count=len(features) if features is not None else 0,
                fraud_count=0,
                fraud_rate=0.0,
                predictions=[],
                process_status=ProcessStatus.ERROR,
                error_reason=str(e),
            )


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
        self.amount_received =  amount_received
        self.receiving_currency = receiving_currency
        self.payment_currency = payment_currency
        self.payment_format = payment_format
        self.day =  day
    
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
            df = pd.DataFrame(custom_data_input_dict)

            df, _ = clean_categorical_fields(df)

            return df

        except Exception as e:
            raise CustomerException(e, sys)
