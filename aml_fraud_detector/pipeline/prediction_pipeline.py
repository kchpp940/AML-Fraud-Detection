import sys
import os
import json
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import asdict

from aml_fraud_detector.exception import (
    AMLException,
    InputValidationException,
    FeatureAlignmentException,
    PredictionException,
    ModelLoadingException,
    MetadataValidationException,
    DataQualityException,
    ExplanationException,
    wrap_exception,
    create_error_response,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.utils.artifact_validator import (
    ArtifactValidator,
    ArtifactValidationResult,
)
from aml_fraud_detector.components.data_validation import (
    DataValidation,
    DataValidationResult,
)
from aml_fraud_detector.utils.risk_explainer import RiskExplainer
from aml_fraud_detector.entity import (
    PredictionResult,
    BatchPredictionResult,
    UnifiedPredictionResponse,
    ValidationStatus,
    RiskExplanation,
    RiskLevel,
    ProcessStatus,
    ModelVersionInfo,
    REQUIRED_INPUT_FIELDS,
)
from aml_fraud_detector.exception import ErrorDetail


FEATURE_SCHEMA_VERSION = "1.0"
MIN_MODEL_VERSION = 1


def _validate_input_field(field_name: str, value: Any, expected_type: type) -> None:
    if value is None:
        raise InputValidationException(
            ErrorCode.INPUT_MISSING_FIELD,
            error_details=sys,
            field=field_name,
        )
    if isinstance(value, str) and value.strip() == "" and expected_type == str:
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
        self.manifest_path = os.path.join(artifacts_dir, "artifact_manifest.json")

        self._model = None
        self._preprocessor = None
        self._feature_metadata: Optional[Dict[str, Any]] = None
        self._model_metadata: Optional[Dict[str, Any]] = None
        self._explainer: Optional[RiskExplainer] = None
        self._expected_columns: Optional[List[str]] = None

        self._artifact_validator = ArtifactValidator(artifacts_dir)
        self._last_validation: Optional[ValidationStatus] = None

    # ------------------------------------------------------------
    # Artifact & Version validation
    # ------------------------------------------------------------
    def validate_artifacts(
        self,
        verify_digests: bool = True,
        check_metadata_versions: bool = True,
    ) -> ValidationStatus:
        status = ValidationStatus(is_valid=True)
        status.checked_artifacts = []

        av_result: ArtifactValidationResult = self._artifact_validator.validate(
            verify_digests=verify_digests,
        )
        status.checked_artifacts = list(av_result.checked_artifacts)
        if not av_result.is_valid:
            status.is_valid = False
            status.errors.append(av_result.first_error_message())
            self._last_validation = status
            raise MetadataValidationException(
                av_result.first_error_code(),
                error_details=sys,
                detail=av_result.first_error_message(),
            )

        if check_metadata_versions:
            try:
                self._load_metadata_only()
            except AMLException as e:
                status.is_valid = False
                status.errors.append(e.message)
                self._last_validation = status
                raise

        self._last_validation = status
        return status

    def _load_metadata_only(self) -> None:
        if not os.path.exists(self.feature_metadata_path):
            raise MetadataValidationException(
                ErrorCode.METADATA_FILE_NOT_FOUND,
                error_details=sys,
                path=self.feature_metadata_path,
            )
        try:
            with open(self.feature_metadata_path, "r", encoding="utf-8") as f:
                fm = json.load(f)
        except Exception as e:
            raise MetadataValidationException(
                ErrorCode.METADATA_CORRUPTED,
                error_details=sys,
                path=self.feature_metadata_path,
                detail=str(e),
            )
        actual_schema = fm.get("contract_version", "0.0")
        if actual_schema != FEATURE_SCHEMA_VERSION:
            raise MetadataValidationException(
                ErrorCode.METADATA_VERSION_MISMATCH,
                error_details=sys,
                expected=FEATURE_SCHEMA_VERSION,
                actual=actual_schema,
            )
        self._feature_metadata = fm
        orig_features = list(fm.get("original_features", []))
        nr = fm.get("normalization_report", {})
        renamed_values = set(nr.get("renamed_columns", {}).values())
        target_col = nr.get("target_column")
        if orig_features:
            self._expected_columns = list(orig_features)
        else:
            self._expected_columns = [
                c for c in renamed_values if c != target_col
            ]

        if not os.path.exists(self.model_metadata_path):
            raise MetadataValidationException(
                ErrorCode.METADATA_FILE_NOT_FOUND,
                error_details=sys,
                path=self.model_metadata_path,
            )
        try:
            with open(self.model_metadata_path, "r", encoding="utf-8") as f:
                mm = json.load(f)
        except Exception as e:
            raise MetadataValidationException(
                ErrorCode.METADATA_CORRUPTED,
                error_details=sys,
                path=self.model_metadata_path,
                detail=str(e),
            )
        model_version = int(mm.get("model_version", 0))
        if model_version < MIN_MODEL_VERSION:
            raise MetadataValidationException(
                ErrorCode.METADATA_VERSION_MISMATCH,
                error_details=sys,
                expected=f">={MIN_MODEL_VERSION}",
                actual=str(model_version),
            )
        self._model_metadata = mm

    # ------------------------------------------------------------
    # Core loader
    # ------------------------------------------------------------
    def _load_artifacts(self) -> None:
        if self._model is None:
            self.validate_artifacts(verify_digests=True, check_metadata_versions=True)

        if self._model is None:
            self._model = load_object(file_path=self.model_path)
            logging.info(f"Model loaded from {self.model_path}")

        if self._preprocessor is None:
            self._preprocessor = load_object(file_path=self.preprocessor_path)
            logging.info(f"Preprocessor loaded from {self.preprocessor_path}")

        if self._explainer is None:
            self._explainer = RiskExplainer(
                model=self._model,
                preprocessor=self._preprocessor,
                feature_metadata=self._feature_metadata,
            )

    # ------------------------------------------------------------
    # Version info
    # ------------------------------------------------------------
    def get_model_version_info(self) -> ModelVersionInfo:
        info = ModelVersionInfo()
        try:
            self._load_metadata_only()
            if self._model_metadata:
                info.model_version = int(self._model_metadata.get("model_version", 0))
                info.model_name = str(self._model_metadata.get("best_model_name", ""))
                info.training_time = str(self._model_metadata.get("training_time", ""))
                info.selection_metric = str(self._model_metadata.get("selection_metric", ""))
                info.best_metric_value = float(self._model_metadata.get("best_metric_value", 0.0))
                if self._feature_metadata:
                    info.feature_schema_version = str(
                        self._feature_metadata.get("contract_version", "")
                    )
            info.artifact_path = os.path.abspath(self.artifacts_dir)
        except AMLException:
            pass
        return info

    # ------------------------------------------------------------
    # Feature alignment & data validation
    # ------------------------------------------------------------
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_cols = self._expected_columns or []
        if not expected_cols:
            return df

        target_col = None
        if self._feature_metadata:
            target_col = self._feature_metadata.get("normalization_report", {}).get("target_column")

        input_cols = [c for c in expected_cols if c != target_col]
        actual_cols = df.columns.tolist()

        missing = [c for c in input_cols if c not in actual_cols]
        if missing:
            raise FeatureAlignmentException(
                ErrorCode.FEATURE_MISSING,
                error_details=sys,
                missing=", ".join(missing),
            )

        extra = [c for c in actual_cols if c not in input_cols]
        if extra:
            logging.warning(f"Dropping unexpected input columns: {extra}")
            df = df.drop(columns=extra, errors="ignore")

        try:
            df = df[input_cols]
        except KeyError as e:
            raise FeatureAlignmentException(
                ErrorCode.FEATURE_MISMATCH,
                error_details=sys,
                expected=str(input_cols),
                actual=str(actual_cols),
            )
        return df

    def _run_data_validation(self, df: pd.DataFrame) -> Optional[DataValidationResult]:
        dv = DataValidation(
            critical_feature_columns=self._expected_columns
            or REQUIRED_INPUT_FIELDS,
            target_column=self._feature_metadata.get("normalization_report", {}).get("target_column")
            if self._feature_metadata else None,
        )
        return dv.validate(df)

    # ------------------------------------------------------------
    # Original-compatible API (returns arrays)
    # ------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        logging.info("predict(ndarray) - original API")
        try:
            self._load_artifacts()
            features_aligned = self._align_features(features)
            data_scaled = self._transform(features_aligned)
            predictions = self._model.predict(data_scaled)
            return np.asarray(predictions).astype(int).ravel()
        except AMLException as e:
            logging.error(f"predict() failed: {e}")
            raise
        except Exception as e:
            logging.error("predict() unexpected failure", exc_info=True)
            raise wrap_exception(e, error_details=sys)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        logging.info("predict_proba(ndarray) - original API")
        try:
            self._load_artifacts()
            features_aligned = self._align_features(features)
            data_scaled = self._transform(features_aligned)
            proba = self._model.predict_proba(data_scaled)
            return np.asarray(proba, dtype=float)
        except AMLException as e:
            logging.error(f"predict_proba() failed: {e}")
            raise
        except Exception as e:
            logging.error("predict_proba() unexpected failure", exc_info=True)
            raise wrap_exception(e, error_details=sys)

    def _transform(self, df: pd.DataFrame) -> np.ndarray:
        try:
            data_scaled = self._preprocessor.transform(df)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()
            return np.asarray(data_scaled, dtype=float)
        except AMLException:
            raise
        except Exception as e:
            raise FeatureAlignmentException(
                ErrorCode.FEATURE_TRANSFORM_FAILED,
                error_details=sys,
                detail=str(e),
            )

    # ------------------------------------------------------------
    # New detailed API (returns Unified objects)
    # ------------------------------------------------------------
    def predict_detailed(
        self,
        features: pd.DataFrame,
        transaction_id: Optional[str] = None,
    ) -> PredictionResult:
        logging.info(f"predict_detailed(1 row, tid={transaction_id})")
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

            dv_result = self._run_data_validation(features_aligned)
            if dv_result is not None and not dv_result.is_valid:
                raise DataQualityException(
                    dv_result.first_error_code(),
                    error_details=sys,
                    message=dv_result.first_error_message(),
                    detail=dv_result.first_error_message(),
                )

            data_scaled = self._transform(features_aligned)

            try:
                pred = self._model.predict(data_scaled)
                proba = self._model.predict_proba(data_scaled)
            except Exception as e:
                raise PredictionException(
                    ErrorCode.PREDICTION_FAILED,
                    error_details=sys,
                    detail=str(e),
                )

            prediction = int(pred.ravel()[0])
            proba_arr = np.asarray(proba, dtype=float).reshape(1, -1)
            proba_row = proba_arr[0]
            fraud_prob = float(proba_row[1]) if proba_arr.shape[1] > 1 else float(proba_row[0])
            legit_prob = float(proba_row[0]) if proba_arr.shape[1] > 1 else 1.0 - fraud_prob

            try:
                if self._explainer is not None:
                    explanation = self._explainer.explain(
                        transformed_features=data_scaled[0],
                        fraud_prob=fraud_prob,
                        input_df=features_aligned,
                    )
                else:
                    explanation = RiskExplanation(
                        fraud_probability=fraud_prob,
                        risk_level=RiskLevel.LOW,
                        top_factors=[],
                    )
            except ExplanationException as ee:
                logging.warning(f"Risk explanation skipped: {ee.message}")
                from aml_fraud_detector.utils.risk_explainer import _risk_level_from_prob
                explanation = RiskExplanation(
                    fraud_probability=fraud_prob,
                    risk_level=_risk_level_from_prob(fraud_prob),
                    top_factors=[],
                )

            mv_info = self.get_model_version_info()
            result = PredictionResult(
                prediction=prediction,
                fraud_probability=fraud_prob,
                legit_probability=legit_prob,
                class_label="Fraud" if prediction == 1 else "Not Fraud",
                process_status=ProcessStatus.SUCCESS,
                model_version=mv_info.model_version,
                transaction_id=transaction_id,
                risk_explanation=explanation,
            )
            logging.info(
                f"predict_detailed done: class={result.class_label}, "
                f"fraud_prob={fraud_prob:.4f}, risk={explanation.risk_level.value}"
            )
            return result

        except AMLException as e:
            logging.error(f"predict_detailed failed: {e}")
            return PredictionResult.from_error(e.error_detail, transaction_id=transaction_id)
        except Exception as e:
            logging.error("predict_detailed unexpected failure", exc_info=True)
            wrapped = wrap_exception(e, error_details=sys)
            return PredictionResult.from_error(wrapped.error_detail, transaction_id=transaction_id)

    def predict_batch(self, features: pd.DataFrame) -> BatchPredictionResult:
        logging.info(f"predict_batch: {len(features)} rows")
        try:
            self._load_artifacts()

            if len(features) == 0:
                raise DataQualityException(
                    ErrorCode.DATA_EMPTY,
                    error_details=sys,
                    rows=0,
                )

            features_aligned = self._align_features(features)

            dv_result = self._run_data_validation(features_aligned)
            if dv_result is not None and not dv_result.is_valid:
                raise DataQualityException(
                    dv_result.first_error_code(),
                    error_details=sys,
                    message=dv_result.first_error_message(),
                    detail=dv_result.first_error_message(),
                )

            data_scaled = self._transform(features_aligned)

            try:
                preds = self._model.predict(data_scaled)
                proba = self._model.predict_proba(data_scaled)
            except Exception as e:
                raise PredictionException(
                    ErrorCode.BATCH_PREDICTION_FAILED,
                    error_details=sys,
                    detail=str(e),
                )

            preds_arr = np.asarray(preds).astype(int).ravel()
            proba_arr = np.asarray(proba, dtype=float)
            model_version = self.get_model_version_info().model_version
            results: List[PredictionResult] = []
            fraud_count = 0

            for i in range(len(preds_arr)):
                prediction = int(preds_arr[i])
                row_proba = proba_arr[i]
                fraud_prob = float(row_proba[1]) if proba_arr.shape[1] > 1 else float(row_proba[0])
                legit_prob = float(row_proba[0]) if proba_arr.shape[1] > 1 else 1.0 - fraud_prob
                if prediction == 1:
                    fraud_count += 1
                from aml_fraud_detector.utils.risk_explainer import _risk_level_from_prob
                results.append(PredictionResult(
                    prediction=prediction,
                    fraud_probability=fraud_prob,
                    legit_probability=legit_prob,
                    class_label="Fraud" if prediction == 1 else "Not Fraud",
                    process_status=ProcessStatus.SUCCESS,
                    model_version=model_version,
                    transaction_id=str(i),
                    risk_explanation=RiskExplanation(
                        fraud_probability=fraud_prob,
                        risk_level=_risk_level_from_prob(fraud_prob),
                        top_factors=[],
                    ),
                ))

            br = BatchPredictionResult(
                total_count=len(results),
                fraud_count=fraud_count,
                fraud_rate=(fraud_count / len(results) if results else 0.0),
                overall_risk_level=_risk_level_from_prob(fraud_count / len(results) if results else 0.0),
                predictions=results,
                process_status=ProcessStatus.SUCCESS,
            )
            logging.info(
                f"predict_batch done: total={br.total_count}, "
                f"fraud={br.fraud_count}, rate={br.fraud_rate:.2%}"
            )
            return br

        except AMLException as e:
            logging.error(f"predict_batch failed: {e}")
            return BatchPredictionResult.from_error(e.error_detail)
        except Exception as e:
            logging.error("predict_batch unexpected failure", exc_info=True)
            wrapped = wrap_exception(e, error_details=sys)
            return BatchPredictionResult.from_error(wrapped.error_detail)


class CustomData:
    def __init__(self,
            from_bank, account, to_bank, account_1,
            amount_received, receiving_currency, payment_currency,
            payment_format, day):

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
        validators = [
            ("from_bank", self.from_bank, int),
            ("account", self.account, str),
            ("to_bank", self.to_bank, int),
            ("account_1", self.account_1, str),
            ("amount_received", self.amount_received, float),
            ("receiving_currency", self.receiving_currency, str),
            ("payment_currency", self.payment_currency, str),
            ("payment_format", self.payment_format, str),
            ("day", self.day, str),
        ]
        for name, value, etype in validators:
            try:
                _validate_input_field(name, value, etype)
            except InputValidationException as e:
                status.is_valid = False
                status.errors.append(e.message)

        try:
            amt_val = float(self.amount_received)
            if amt_val < 0:
                status.is_valid = False
                status.errors.append(
                    f"字段 amount_received 不能为负数: {self.amount_received}"
                )
        except (TypeError, ValueError):
            pass

        return status

    def get_data_as_DataFrame(self) -> pd.DataFrame:
        try:
            data = {
                "from_bank": [int(self.from_bank)],
                "account": [str(self.account)],
                "to_bank": [int(self.to_bank)],
                "account_1": [str(self.account_1)],
                "amount_received": [float(self.amount_received)],
                "receiving_currency": [str(self.receiving_currency)],
                "payment_currency": [str(self.payment_currency)],
                "payment_format": [str(self.payment_format)],
                "day": [str(self.day)],
            }
            df = pd.DataFrame(data)
            if "from_bank" in df.columns:
                df["from_bank"] = df["from_bank"].astype("object")
            if "to_bank" in df.columns:
                df["to_bank"] = df["to_bank"].astype("object")
            return df
        except AMLException:
            raise
        except Exception as e:
            raise InputValidationException(
                ErrorCode.INPUT_INVALID_FORMAT,
                error_details=sys,
                field="dataframe_creation",
                value=str(e),
            )
