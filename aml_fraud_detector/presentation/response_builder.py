from dataclasses import asdict
from typing import Dict, List, Any, Optional

import pandas as pd

from aml_fraud_detector.entity import (
    ProcessStatus,
    RiskLevel,
    PredictionResult,
    BatchPredictionResult,
    ModelVersionInfo,
    ValidationStatus,
    UnifiedPredictionResponse,
)
from aml_fraud_detector.pipeline.prediction_pipeline import PredictionPipeline


class ResponseBuilder:
    def __init__(self, pipeline: PredictionPipeline):
        self._pipeline = pipeline

    def build_single_response(
        self,
        data: Dict[str, Any],
    ) -> UnifiedPredictionResponse:
        validation = self._pipeline.validate_artifacts()
        model_version = self._pipeline.get_model_version_info()
        result = self._pipeline.predict_single(data)
        return UnifiedPredictionResponse(
            model_version=model_version,
            validation=validation,
            single_prediction=result,
            batch_prediction=None,
        )

    def build_batch_response(
        self,
        df: pd.DataFrame,
    ) -> UnifiedPredictionResponse:
        validation = self._pipeline.validate_artifacts()
        model_version = self._pipeline.get_model_version_info()
        batch = self._pipeline.predict_batch(df)
        return UnifiedPredictionResponse(
            model_version=model_version,
            validation=validation,
            single_prediction=None,
            batch_prediction=batch,
        )

    def build_validation_response(self) -> UnifiedPredictionResponse:
        validation = self._pipeline.validate_artifacts()
        model_version = self._pipeline.get_model_version_info()
        return UnifiedPredictionResponse(
            model_version=model_version,
            validation=validation,
            single_prediction=None,
            batch_prediction=None,
        )

    @staticmethod
    def to_dict(response: UnifiedPredictionResponse) -> Dict[str, Any]:
        return asdict(response)

    @staticmethod
    def to_flat_dataframe(response: UnifiedPredictionResponse) -> pd.DataFrame:
        if response.batch_prediction is None:
            return pd.DataFrame()
        return _flatten_batch(response.batch_prediction)

    @staticmethod
    def extract_display_fields(response: UnifiedPredictionResponse) -> Dict[str, Any]:
        v = response.validation
        m = response.model_version
        fields: Dict[str, Any] = {
            "model_version_number": m.model_version,
            "model_name": m.model_name,
            "training_time": m.training_time,
            "selection_metric": m.selection_metric,
            "best_metric_value": m.best_metric_value,
            "validation_valid": v.is_valid,
            "validation_errors": v.errors,
            "validation_warnings": v.warnings,
            "has_alerts": bool(v.errors or v.warnings),
        }
        if response.single_prediction is not None:
            sp = response.single_prediction
            re = sp.risk_explanation
            fields.update({
                "is_error": sp.process_status == ProcessStatus.ERROR,
                "error_reason": sp.error_reason,
                "prediction_label": sp.class_label,
                "prediction_code": sp.prediction,
                "fraud_probability": sp.fraud_probability,
                "legit_probability": sp.legit_probability,
                "risk_level": re.risk_level.value if isinstance(re.risk_level, RiskLevel) else re.risk_level,
                "risk_factors": re.top_factors,
                "transaction_id": sp.transaction_id,
            })
        if response.batch_prediction is not None:
            bp = response.batch_prediction
            fields.update({
                "batch_total": bp.total_count,
                "batch_fraud_count": bp.fraud_count,
                "batch_fraud_rate": bp.fraud_rate,
                "batch_is_error": bp.process_status == ProcessStatus.ERROR,
                "batch_error_reason": bp.error_reason,
            })
        return fields


def _flatten_batch(batch: BatchPredictionResult) -> pd.DataFrame:
    rows = []
    for p in batch.predictions:
        re = p.risk_explanation
        rows.append({
            "prediction": p.prediction,
            "fraud_probability": p.fraud_probability,
            "legit_probability": p.legit_probability,
            "class_label": p.class_label,
            "process_status": p.process_status.value if isinstance(p.process_status, ProcessStatus) else p.process_status,
            "error_reason": p.error_reason,
            "model_version": p.model_version,
            "transaction_id": p.transaction_id,
            "risk_level": re.risk_level.value if isinstance(re.risk_level, RiskLevel) else re.risk_level,
            "top_factors": re.top_factors,
        })
    return pd.DataFrame(rows)
