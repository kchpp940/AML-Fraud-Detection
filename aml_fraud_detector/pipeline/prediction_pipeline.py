import sys
import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import asdict

from aml_fraud_detector.exception import (
    AMLException,
    InputValidationException,
    FeatureAlignmentException,
    PredictionException,
    ModelLoadingException,
    wrap_exception,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.entity import (
    REQUIRED_INPUT_FIELDS,
    PredictionResult,
    BatchPredictionResult,
    UnifiedPredictionResponse,
    ValidationStatus,
    RiskExplanation,
    RiskLevel,
    ProcessStatus,
    ModelVersionInfo,
)
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object


def _validate_input_field(field_name: str, value: Any, expected_type: type) -> None:
    if value is None:
        raise InputValidationException(
            ErrorCode.INPUT_MISSING_FIELD,
            error_details=sys,
            field=field_name,
        )
    if isinstance(value, str) and value.strip() == "":
        raise InputValidationException(
            ErrorCode.INPUT_EMPTY_VALUE,
            error_details=sys,
            field=field_name,
        )
    try:
        if expected_type == int and not isinstance(value, bool):
            int(value)
        elif expected_type == float and not isinstance(value, bool):
            float(value)
        elif expected_type == str:
            str(value)
    except (ValueError, TypeError):
        raise InputValidationException(
            ErrorCode.INPUT_INVALID_TYPE,
            error_details=sys,
            field=field_name,
            expected=expected_type.__name__,
            value=str(value),
        )


class PredictionPipeline:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.model_path = os.path.join(artifacts_dir, "model.pkl")
        self.preprocessor_path = os.path.join(artifacts_dir, "preprocessor.pkl")
        self.feature_metadata_path = os.path.join(artifacts_dir, "feature_metadata.json")
        self.model_metadata_path = os.path.join(artifacts_dir, "model_metadata.json")

        self._model = None
        self._preprocessor = None
        self._feature_metadata: Optional[Dict[str, Any]] = None
        self._model_metadata: Optional[Dict[str, Any]] = None

    def _load_artifacts(self) -> None:
        if self._model is None:
            try:
                self._model = load_object(file_path=self.model_path)
            except ModelLoadingException:
                raise
            except Exception as e:
                raise ModelLoadingException(
                    ErrorCode.MODEL_CORRUPTED,
                    error_details=sys,
                    path=self.model_path,
                    detail=str(e),
                )

        if self._preprocessor is None:
            try:
                self._preprocessor = load_object(file_path=self.preprocessor_path)
            except ModelLoadingException:
                raise
            except Exception as e:
                raise ModelLoadingException(
                    ErrorCode.PREPROCESSOR_CORRUPTED,
                    error_details=sys,
                    path=self.preprocessor_path,
                    detail=str(e),
                )

        if self._feature_metadata is None and os.path.exists(self.feature_metadata_path):
            try:
                import json
                with open(self.feature_metadata_path, "r") as f:
                    self._feature_metadata = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load feature metadata: {e}")

        if self._model_metadata is None and os.path.exists(self.model_metadata_path):
            try:
                import json
                with open(self.model_metadata_path, "r") as f:
                    self._model_metadata = json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load model metadata: {e}")

    def get_model_version_info(self) -> ModelVersionInfo:
        try:
            self._load_artifacts()
        except Exception:
            return ModelVersionInfo()

        info = ModelVersionInfo()
        if self._model_metadata:
            info.model_version = int(self._model_metadata.get("model_version", 0))
            info.model_name = str(self._model_metadata.get("model_name", ""))
            info.training_time = str(self._model_metadata.get("training_time", ""))
            info.selection_metric = str(self._model_metadata.get("selection_metric", ""))
            info.best_metric_value = float(self._model_metadata.get("best_metric_value", 0.0))
            info.feature_schema_version = str(self._model_metadata.get("feature_schema_version", ""))
        info.artifact_path = os.path.abspath(self.artifacts_dir)
        return info

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._feature_metadata and "expected_columns" in self._feature_metadata:
            expected_cols = self._feature_metadata["expected_columns"]
            actual_cols = df.columns.tolist()

            missing_cols = [c for c in expected_cols if c not in actual_cols]
            if missing_cols:
                raise FeatureAlignmentException(
                    ErrorCode.FEATURE_MISSING,
                    error_details=sys,
                    missing=", ".join(missing_cols),
                )

            extra_cols = [c for c in actual_cols if c not in expected_cols]
            if extra_cols:
                logging.warning(f"Unexpected columns in input: {extra_cols}")
                df = df.drop(columns=extra_cols)

            df = df[expected_cols]

        return df

    def predict(self, features: pd.DataFrame) -> PredictionResult:
        logging.info("Starting single prediction")
        try:
            self._load_artifacts()

            if len(features) != 1:
                raise PredictionException(
                    ErrorCode.PREDICTION_SHAPE_MISMATCH,
                    error_details=sys,
                    expected="1 row",
                    actual=f"{len(features)} rows",
                )

            features_aligned = self._align_features(features)

            try:
                data_scaled = self._preprocessor.transform(features_aligned)
                if hasattr(data_scaled, "toarray"):
                    data_scaled = data_scaled.toarray()
                else:
                    data_scaled = np.asarray(data_scaled)
            except Exception as e:
                raise FeatureAlignmentException(
                    ErrorCode.FEATURE_TRANSFORM_FAILED,
                    error_details=sys,
                    detail=str(e),
                )

            try:
                predictions = self._model.predict(data_scaled)
                predictions_proba = self._model.predict_proba(data_scaled)
            except Exception as e:
                raise PredictionException(
                    ErrorCode.PREDICTION_FAILED,
                    error_details=sys,
                    detail=str(e),
                )

            prediction = int(predictions[0])
            proba = predictions_proba[0]
            fraud_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
            legit_prob = float(proba[0]) if len(proba) > 1 else 1.0 - fraud_prob

            risk_level = RiskLevel.LOW
            if fraud_prob >= 0.9:
                risk_level = RiskLevel.CRITICAL
            elif fraud_prob >= 0.7:
                risk_level = RiskLevel.HIGH
            elif fraud_prob >= 0.5:
                risk_level = RiskLevel.MEDIUM

            risk_explanation = RiskExplanation(
                fraud_probability=fraud_prob,
                risk_level=risk_level,
                top_factors=[],
            )

            result = PredictionResult(
                prediction=prediction,
                fraud_probability=fraud_prob,
                legit_probability=legit_prob,
                class_label="Fraud" if prediction == 1 else "Not Fraud",
                process_status=ProcessStatus.SUCCESS,
                model_version=self.get_model_version_info().model_version,
                risk_explanation=risk_explanation,
            )
            logging.info(f"Prediction completed: {result.class_label}, fraud_prob={fraud_prob:.4f}")
            return result

        except AMLException as e:
            logging.error(f"Prediction failed: {e}")
            return PredictionResult.from_error(e.error_detail)
        except Exception as e:
            logging.error("Prediction failed with unexpected error", exc_info=True)
            wrapped = wrap_exception(e, error_details=sys)
            return PredictionResult.from_error(wrapped.error_detail)

    def predict_proba(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        logging.info("Starting predict_proba")
        try:
            self._load_artifacts()
            features_aligned = self._align_features(features)
            data_scaled = self._preprocessor.transform(features_aligned)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()
            else:
                data_scaled = np.asarray(data_scaled)
            return self._model.predict_proba(data_scaled)
        except Exception as e:
            logging.error(f"predict_proba failed: {e}")
            return None

    def predict_batch(self, features: pd.DataFrame) -> BatchPredictionResult:
        logging.info(f"Starting batch prediction for {len(features)} records")
        try:
            self._load_artifacts()

            if len(features) == 0:
                raise PredictionException(
                    ErrorCode.DATA_EMPTY,
                    error_details=sys,
                    rows=0,
                )

            features_aligned = self._align_features(features)

            try:
                data_scaled = self._preprocessor.transform(features_aligned)
                if hasattr(data_scaled, "toarray"):
                    data_scaled = data_scaled.toarray()
                else:
                    data_scaled = np.asarray(data_scaled)
            except Exception as e:
                raise FeatureAlignmentException(
                    ErrorCode.FEATURE_TRANSFORM_FAILED,
                    error_details=sys,
                    detail=str(e),
                )

            try:
                predictions = self._model.predict(data_scaled)
                predictions_proba = self._model.predict_proba(data_scaled)
            except Exception as e:
                raise PredictionException(
                    ErrorCode.BATCH_PREDICTION_FAILED,
                    error_details=sys,
                    detail=str(e),
                )

            results: List[PredictionResult] = []
            fraud_count = 0
            model_version = self.get_model_version_info().model_version

            for i in range(len(predictions)):
                prediction = int(predictions[i])
                proba = predictions_proba[i]
                fraud_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
                legit_prob = float(proba[0]) if len(proba) > 1 else 1.0 - fraud_prob

                if prediction == 1:
                    fraud_count += 1

                risk_level = RiskLevel.LOW
                if fraud_prob >= 0.9:
                    risk_level = RiskLevel.CRITICAL
                elif fraud_prob >= 0.7:
                    risk_level = RiskLevel.HIGH
                elif fraud_prob >= 0.5:
                    risk_level = RiskLevel.MEDIUM

                risk_explanation = RiskExplanation(
                    fraud_probability=fraud_prob,
                    risk_level=risk_level,
                    top_factors=[],
                )

                results.append(PredictionResult(
                    prediction=prediction,
                    fraud_probability=fraud_prob,
                    legit_probability=legit_prob,
                    class_label="Fraud" if prediction == 1 else "Not Fraud",
                    process_status=ProcessStatus.SUCCESS,
                    model_version=model_version,
                    transaction_id=str(i),
                    risk_explanation=risk_explanation,
                ))

            batch_result = BatchPredictionResult(
                total_count=len(results),
                fraud_count=fraud_count,
                fraud_rate=fraud_count / len(results) if results else 0.0,
                predictions=results,
                process_status=ProcessStatus.SUCCESS,
            )
            logging.info(
                f"Batch prediction completed: {batch_result.total_count} total, "
                f"{batch_result.fraud_count} fraud ({batch_result.fraud_rate:.2%})"
            )
            return batch_result

        except AMLException as e:
            logging.error(f"Batch prediction failed: {e}")
            return BatchPredictionResult.from_error(e.error_detail)
        except Exception as e:
            logging.error("Batch prediction failed with unexpected error", exc_info=True)
            wrapped = wrap_exception(e, error_details=sys)
            return BatchPredictionResult.from_error(wrapped.error_detail)


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

    def validate(self) -> ValidationStatus:
        status = ValidationStatus(is_valid=True)
        try:
            _validate_input_field("from_bank", self.from_bank, int)
        except InputValidationException as e:
            status.is_valid = False
            status.errors.append(e.message)

        try:
            _validate_input_field("account", self.account, str)
        except InputValidationException as e:
            status.is_valid = False
            status.errors.append(e.message)

        try:
            _validate_input_field("to_bank", self.to_bank, int)
        except InputValidationException as e:
            status.is_valid = False
            status.errors.append(e.message)

        try:
            _validate_input_field("account_1", self.account_1, str)
        except InputValidationException as e:
            status.is_valid = False
            status.errors.append(e.message)

        try:
            _validate_input_field("amount_received", self.amount_received, float)
            if isinstance(self.amount_received, (int, float)) and float(self.amount_received) < 0:
                status.is_valid = False
                status.errors.append(
                    f"字段 amount_received 不能为负数: {self.amount_received}"
                )
        except InputValidationException as e:
            status.is_valid = False
            status.errors.append(e.message)

        try:
            _validate_input_field("receiving_currency", self.receiving_currency, str)
        except InputValidationException as e:
            status.is_valid = False
            status.errors.append(e.message)

        try:
            _validate_input_field("payment_currency", self.payment_currency, str)
        except InputValidationException as e:
            status.is_valid = False
            status.errors.append(e.message)

        try:
            _validate_input_field("payment_format", self.payment_format, str)
        except InputValidationException as e:
            status.is_valid = False
            status.errors.append(e.message)

        try:
            _validate_input_field("day", self.day, str)
        except InputValidationException as e:
            status.is_valid = False
            status.errors.append(e.message)

        return status

    def get_data_as_DataFrame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "from_bank": [int(self.from_bank)],
                "account": [str(self.account)],
                "to_bank": [int(self.to_bank)],
                "account_1": [str(self.account_1)],
                "amount_received": [float(self.amount_received)],
                "receiving_currency": [str(self.receiving_currency)],
                "payment_currency": [str(self.payment_currency)],
                "payment_format": [str(self.payment_format)],
                "day": [str(self.day)]
            }
            df = pd.DataFrame(custom_data_input_dict)

            if "from_bank" in df.columns:
                df["from_bank"] = df["from_bank"].astype("object")
            if "to_bank" in df.columns:
                df["to_bank"] = df["to_bank"].astype("object")

            return df

        except Exception as e:
            raise InputValidationException(
                ErrorCode.INPUT_INVALID_FORMAT,
                error_details=sys,
                field="dataframe_creation",
                value=str(e),
            )
