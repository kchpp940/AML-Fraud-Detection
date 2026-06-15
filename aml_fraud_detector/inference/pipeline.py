import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.inference.contracts import (
    REQUIRED_INPUT_FIELDS,
    BatchPredictionResult,
    FRAUD_LABEL,
    LEGIT_LABEL,
    ModelArtifacts,
    PredictionResult,
    RiskExplanation,
    TransactionInput,
    ValidationReport,
)
from aml_fraud_detector.inference.artifact_validator import ArtifactValidator
from aml_fraud_detector.inference.model_loader import ModelLoader
from aml_fraud_detector.inference.schema_aligner import SchemaAligner
from aml_fraud_detector.inference.predictor import Predictor
from aml_fraud_detector.inference.batch_assembler import BatchAssembler
from aml_fraud_detector.inference.risk_explainer import RiskExplainer


class PredictionPipeline:
    """
    Unified orchestration entry for all fraud detection predictions.

    Responsibilities intentionally kept narrow:
      1. Lazily boot / cache the internal service graph (Validator + Loader + Aligner + Predictor + Assembler + Explainer).
      2. Convert varied caller input (dict / TransactionInput / DataFrame) into the normalized aligned DataFrame.
      3. Invoke services in the correct order and return typed results.

    All concrete logic (artifact loading, schema alignment, preprocessing, inference, result shaping, explanation)
    is delegated to the dedicated helper classes under aml_fraud_detector.inference.

    Flask / Streamlit UI layers MUST always route through PredictionPipeline and MUST NOT directly instantiate
    ModelLoader / SchemaAligner / Predictor or manipulate model inputs, preprocessor outputs, or explanation shapes.
    """

    def __init__(self, artifacts_dir: Optional[str] = None, validate: bool = True):
        self._artifacts_dir = artifacts_dir or os.path.join("artifacts")
        self._validate_on_init = validate
        self._model_loader: Optional[ModelLoader] = None
        self._artifact_validator: Optional[ArtifactValidator] = None
        self._schema_aligner: Optional[SchemaAligner] = None
        self._predictor: Optional[Predictor] = None
        self._batch_assembler: BatchAssembler = BatchAssembler()
        self._risk_explainer: Optional[RiskExplainer] = None
        self._artifacts: Optional[ModelArtifacts] = None
        self._ready = False
        logging.info(
            f"PredictionPipeline created (artifacts_dir={self._artifacts_dir}, validate={validate})"
        )

    def _ensure_services_ready(self, force_reload: bool = False) -> ModelArtifacts:
        if self._ready and self._artifacts is not None and not force_reload:
            return self._artifacts
        logging.info("PredictionPipeline: initializing inference services")
        self._artifact_validator = ArtifactValidator(artifacts_dir=self._artifacts_dir)
        self._model_loader = ModelLoader(
            artifacts_dir=self._artifacts_dir, validate=self._validate_on_init
        )
        artifacts = self._model_loader.load(force_reload=force_reload)
        self._schema_aligner = SchemaAligner(artifacts)
        self._predictor = Predictor(artifacts)
        self._risk_explainer = RiskExplainer(artifacts)
        self._artifacts = artifacts
        self._ready = True
        logging.info(
            "PredictionPipeline: all services ready, "
            f"model_version={self._predictor.model_version}"
        )
        return artifacts

    def validate_artifacts(self, strict: bool = True) -> ValidationReport:
        validator = self._artifact_validator or ArtifactValidator(
            artifacts_dir=self._artifacts_dir
        )
        return validator.validate(strict=strict)

    def _normalize_input_to_dataframe(
        self,
        data: Union[Dict[str, Any], TransactionInput, pd.DataFrame, List[Dict[str, Any]]],
    ) -> pd.DataFrame:
        if isinstance(data, TransactionInput):
            return data.to_dataframe()
        if isinstance(data, dict):
            try:
                for field in REQUIRED_INPUT_FIELDS:
                    if field not in data:
                        raise ValueError(
                            f"Input dict missing required field: '{field}'"
                        )
                return pd.DataFrame([data])
            except ValueError as ve:
                raise CustomerException(ve, sys)
        if isinstance(data, list):
            try:
                if not data:
                    raise ValueError("Input list is empty")
                return pd.DataFrame(data)
            except ValueError as ve:
                raise CustomerException(ve, sys)
        if isinstance(data, pd.DataFrame):
            try:
                if data.empty:
                    raise ValueError("Input DataFrame is empty")
                return data.copy()
            except ValueError as ve:
                raise CustomerException(ve, sys)
        try:
            raise TypeError(
                f"Unsupported input type: {type(data).__name__}. "
                f"Expected TransactionInput, dict, list[dict], or DataFrame."
            )
        except TypeError as te:
            raise CustomerException(te, sys)

    def predict_single(
        self,
        data: Union[Dict[str, Any], TransactionInput, pd.DataFrame],
        explain: bool = True,
        transaction_id: Optional[str] = None,
    ) -> PredictionResult:
        try:
            artifacts = self._ensure_services_ready()
            raw_df = self._normalize_input_to_dataframe(data)
            if len(raw_df) != 1:
                try:
                    raise ValueError(
                        f"predict_single expects exactly 1 row, got {len(raw_df)}. "
                        f"Use predict_batch for multiple transactions."
                    )
                except ValueError as ve:
                    raise CustomerException(ve, sys)
            aligned_df = self._schema_aligner.align_dataframe(raw_df)
            predictions, probabilities = self._predictor.predict_with_proba(aligned_df)
            explanation: Optional[RiskExplanation] = None
            if explain:
                explanation = self._risk_explainer.explain_from_prob(
                    float(probabilities[0, 1]),
                    int(predictions[0]),
                )
            result = self._batch_assembler.assemble_single(
                prediction=int(predictions[0]),
                proba_row=probabilities[0],
                model_version=self._predictor.model_version,
                transaction_id=transaction_id,
                explanation=explanation,
            )
            logging.info(
                f"PredictionPipeline.predict_single: label={result.class_label}, "
                f"fraud_prob={result.fraud_probability:.4f}, "
                f"status={result.process_status}"
            )
            return result
        except CustomerException as e:
            logging.error(f"PredictionPipeline.predict_single failed: {e.error_message}")
            version = (
                self._predictor.model_version
                if self._predictor is not None
                else "unknown"
            )
            return self._batch_assembler.assemble_single_error(
                error_reason=str(e.error_message),
                model_version=version,
                transaction_id=transaction_id,
            )
        except Exception as e:
            logging.error(f"PredictionPipeline.predict_single failed: {e}")
            version = (
                self._predictor.model_version
                if self._predictor is not None
                else "unknown"
            )
            return self._batch_assembler.assemble_single_error(
                error_reason=str(e),
                model_version=version,
                transaction_id=transaction_id,
            )

    def predict_batch(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], List[TransactionInput]],
        explain: bool = False,
        transaction_ids: Optional[List[str]] = None,
    ) -> BatchPredictionResult:
        try:
            artifacts = self._ensure_services_ready()
            if isinstance(data, list) and data and isinstance(data[0], TransactionInput):
                raw_df = pd.DataFrame([t.to_dict() for t in data])
            else:
                raw_df = self._normalize_input_to_dataframe(data)
            aligned_df = self._schema_aligner.align_dataframe(raw_df)
            predictions, probabilities = self._predictor.predict_with_proba(aligned_df)
            explanations: Optional[List[Optional[RiskExplanation]]] = None
            if explain:
                explanations = self._risk_explainer.explain_from_dataframe(
                    aligned_df, predictions, probabilities
                )
            batch = self._batch_assembler.assemble_batch(
                predictions=predictions,
                probabilities=probabilities,
                model_version=self._predictor.model_version,
                transaction_ids=transaction_ids,
                explanations=explanations,
            )
            logging.info(
                f"PredictionPipeline.predict_batch: n={batch.total_count}, "
                f"fraud_rate={batch.fraud_rate:.4f}"
            )
            return batch
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Backward-compatible shim for the legacy PredictionPipeline.predict(features) API.
        Prefer predict_single / predict_batch for new code.
        """
        try:
            batch = self.predict_batch(features, explain=False)
            return np.array([r.prediction for r in batch.results], dtype=int)
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """
        Backward-compatible shim for the legacy PredictionPipeline.predict_proba(features) API.
        Prefer predict_single / predict_batch for new code.
        """
        try:
            artifacts = self._ensure_services_ready()
            raw_df = features.copy() if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
            aligned_df = self._schema_aligner.align_dataframe(raw_df)
            return self._predictor.predict_proba(aligned_df)
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)

    @property
    def model_version(self) -> str:
        self._ensure_services_ready()
        return self._predictor.model_version

    @property
    def model_name(self) -> str:
        self._ensure_services_ready()
        return self._predictor.model_name

    def is_ready(self) -> bool:
        return self._ready

    def reload(self) -> None:
        logging.info("PredictionPipeline: reloading artifacts and services")
        ModelLoader.reset_instance()
        self._ready = False
        self._artifacts = None
        self._ensure_services_ready(force_reload=True)

    def last_validation_report(self) -> Optional[ValidationReport]:
        if self._model_loader is None:
            return None
        return self._model_loader.last_validation_report()


class CustomData(TransactionInput):
    """
    Backward-compatible alias of TransactionInput for callers that import
    the old `CustomData` class directly from prediction_pipeline.py.
    """

    def get_data_as_DataFrame(self) -> pd.DataFrame:
        return self.to_dataframe()
