from __future__ import annotations

import sys
import uuid
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from aml_fraud_detector.constants import (
    ErrorCode,
    ErrorCategory,
    ERROR_CATEGORY_DISPLAY,
    ERROR_SEVERITY,
    HTTP_STATUS_CODES,
    PROCESS_STATUS_SUCCESS,
    PROCESS_STATUS_ERROR,
)
from aml_fraud_detector.exception import (
    AMLException,
    ErrorDetail,
    UnifiedErrorResponse,
    wrap_exception,
    create_error_response,
)
from aml_fraud_detector.entity.artifact_entity import (
    PredictionResult,
    BatchPredictionResult,
    ValidationStatus,
    ModelVersionInfo,
    ProcessStatus,
    RiskLevel,
)


ERROR_FIELDNAMES = [
    "error_code",
    "error_category",
    "error_category_display",
    "error_message",
    "error_field",
    "error_value",
    "error_context",
    "error_trace_id",
    "error_severity",
]

MODEL_VERSION_FIELDNAMES = [
    "model_version",
    "model_name",
    "training_time",
    "selection_metric",
    "best_metric_value",
    "feature_contract_version",
]

VALIDATION_FIELDNAMES = [
    "validation_valid",
    "validation_errors",
    "validation_warnings",
    "validation_details",
    "has_alerts",
]

SINGLE_FIELDNAMES = [
    "is_error",
    "error_reason",
    "prediction_code",
    "prediction_label",
    "fraud_probability",
    "legit_probability",
    "risk_level",
    "risk_summary",
    "risk_contributors",
    "transaction_id",
    "process_status",
]

BATCH_FIELDNAMES = [
    "batch_total",
    "batch_fraud",
    "batch_legit",
    "batch_fraud_rate",
    "batch_is_error",
    "batch_error_reason",
]

BATCH_DF_COLUMNS = [
    "transaction_id",
    "prediction_code",
    "prediction_label",
    "fraud_probability",
    "legit_probability",
    "risk_level",
    "risk_summary",
    "top_factor_1",
    "top_factor_2",
    "top_factor_3",
    "process_status",
]


@dataclass
class UnifiedViewModel:
    model_version: str = ""
    model_name: str = ""
    training_time: str = ""
    selection_metric: str = ""
    best_metric_value: str = ""
    feature_contract_version: str = ""

    validation_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    validation_details: List[Dict[str, Any]] = field(default_factory=list)
    has_alerts: bool = False

    is_error: bool = False
    error_reason: str = ""
    prediction_code: int = 0
    prediction_label: str = ""
    fraud_probability: float = 0.0
    legit_probability: float = 0.0
    risk_level: str = "LOW"
    risk_summary: str = ""
    risk_contributors: List[Dict[str, Any]] = field(default_factory=list)
    transaction_id: str = ""
    process_status: str = "success"

    batch_total: int = 0
    batch_fraud: int = 0
    batch_legit: int = 0
    batch_fraud_rate: str = "0.00%"
    batch_is_error: bool = False
    batch_error_reason: str = ""

    error_code: str = ""
    error_category: str = ""
    error_category_display: str = ""
    error_message: str = ""
    error_field: str = ""
    error_value: str = ""
    error_context: Dict[str, str] = field(default_factory=dict)
    error_trace_id: str = ""
    error_severity: str = "info"

    _batch_results: List[PredictionResult] = field(default_factory=list)
    _raw_error: Optional[UnifiedErrorResponse] = None

    def has_error(self) -> bool:
        return bool(self.error_code) or self.is_error or self.batch_is_error


class ResponseBuilder:
    def __init__(self):
        self._last_trace_id: str = ""

    def _new_trace_id(self) -> str:
        tid = uuid.uuid4().hex[:16]
        self._last_trace_id = tid
        return tid

    def _fill_model_fields(
        self, vm: UnifiedViewModel, mvi: Optional[ModelVersionInfo]
    ) -> None:
        if mvi is None:
            return
        vm.model_version = mvi.model_version or ""
        vm.model_name = mvi.model_name or ""
        vm.training_time = mvi.training_time or ""
        vm.selection_metric = mvi.selection_metric or ""
        vm.best_metric_value = mvi.best_metric_value or ""
        vm.feature_contract_version = mvi.feature_contract_version or ""

    def _fill_validation_fields(
        self, vm: UnifiedViewModel, vs: Optional[ValidationStatus]
    ) -> None:
        if vs is None:
            return
        vm.validation_valid = vs.is_valid
        vm.validation_errors = list(vs.errors)
        vm.validation_warnings = list(vs.warnings)
        vm.validation_details = list(vs.details)
        vm.has_alerts = (len(vs.errors) + len(vs.warnings)) > 0

    def _fill_error_fields(
        self,
        vm: UnifiedViewModel,
        error_detail: Optional[ErrorDetail],
        trace_id: str,
    ) -> None:
        if error_detail is None:
            return
        vm.error_code = error_detail.error_code.value if hasattr(error_detail.error_code, "value") else str(error_detail.error_code)
        vm.error_category = error_detail.error_category.value if hasattr(error_detail.error_category, "value") else str(error_detail.error_category)
        vm.error_category_display = ERROR_CATEGORY_DISPLAY.get(error_detail.error_category, vm.error_category)
        vm.error_message = error_detail.message or ""
        vm.error_field = error_detail.field_name or ""
        vm.error_value = str(error_detail.field_value) if error_detail.field_value is not None else ""
        vm.error_context = dict(error_detail.context) if error_detail.context else {}
        vm.error_trace_id = trace_id or self._last_trace_id
        vm.error_severity = ERROR_SEVERITY.get(error_detail.error_code, "info")

    def build_validation_only(
        self,
        model_version_info: Optional[ModelVersionInfo] = None,
        validation_status: Optional[ValidationStatus] = None,
        trace_id: Optional[str] = None,
    ) -> UnifiedViewModel:
        tid = trace_id or self._new_trace_id()
        vm = UnifiedViewModel()
        self._fill_model_fields(vm, model_version_info)
        self._fill_validation_fields(vm, validation_status)
        vm.process_status = ProcessStatus.SUCCESS.value
        return vm

    def build_single(
        self,
        prediction_result: Optional[PredictionResult] = None,
        model_version_info: Optional[ModelVersionInfo] = None,
        validation_status: Optional[ValidationStatus] = None,
        trace_id: Optional[str] = None,
    ) -> UnifiedViewModel:
        tid = trace_id or self._new_trace_id()
        vm = UnifiedViewModel()
        self._fill_model_fields(vm, model_version_info)
        self._fill_validation_fields(vm, validation_status)

        if prediction_result is None:
            vm.is_error = True
            vm.error_reason = "预测结果为空"
            vm.process_status = ProcessStatus.ERROR.value
            return vm

        vm.transaction_id = prediction_result.transaction_id or ""
        vm.process_status = prediction_result.process_status.value

        if not prediction_result.is_success():
            vm.is_error = True
            vm.error_reason = prediction_result.error_reason or ""
            self._fill_error_fields(vm, prediction_result.error_detail, tid)
            return vm

        vm.prediction_code = int(prediction_result.class_label.value) if hasattr(prediction_result.class_label, "value") else int(prediction_result.class_label)
        vm.prediction_label = prediction_result.class_label.name if hasattr(prediction_result.class_label, "name") else str(prediction_result.class_label)
        vm.fraud_probability = float(prediction_result.fraud_probability)
        vm.legit_probability = float(1.0 - prediction_result.fraud_probability)
        vm.risk_level = prediction_result.risk_level.value if hasattr(prediction_result.risk_level, "value") else str(prediction_result.risk_level)
        vm.risk_summary = prediction_result.risk_summary or ""
        vm.risk_contributors = [dict(x) for x in (prediction_result.top_factors or [])]
        return vm

    def build_batch(
        self,
        batch_result: Optional[BatchPredictionResult] = None,
        model_version_info: Optional[ModelVersionInfo] = None,
        validation_status: Optional[ValidationStatus] = None,
        trace_id: Optional[str] = None,
    ) -> UnifiedViewModel:
        tid = trace_id or self._new_trace_id()
        vm = UnifiedViewModel()
        self._fill_model_fields(vm, model_version_info)
        self._fill_validation_fields(vm, validation_status)

        if batch_result is None:
            vm.batch_is_error = True
            vm.batch_error_reason = "批量预测结果为空"
            vm.process_status = ProcessStatus.ERROR.value
            return vm

        vm.process_status = batch_result.process_status.value
        vm._batch_results = list(batch_result.predictions)

        if not batch_result.is_success():
            vm.batch_is_error = True
            vm.batch_error_reason = batch_result.error_reason or ""
            vm.is_error = True
            vm.error_reason = batch_result.error_reason or ""
            self._fill_error_fields(vm, batch_result.error_detail, tid)
            return vm

        vm.batch_total = int(batch_result.total_count)
        vm.batch_fraud = int(batch_result.fraud_count)
        vm.batch_legit = int(batch_result.total_count - batch_result.fraud_count)
        vm.batch_fraud_rate = f"{float(batch_result.fraud_rate) * 100:.2f}%"
        vm.risk_level = batch_result.overall_risk_level.value if hasattr(batch_result.overall_risk_level, "value") else str(batch_result.overall_risk_level)
        return vm

    def flatten_for_display(self, vm: UnifiedViewModel) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for name in MODEL_VERSION_FIELDNAMES:
            result[name] = getattr(vm, name)
        for name in VALIDATION_FIELDNAMES:
            result[name] = getattr(vm, name)
        for name in SINGLE_FIELDNAMES:
            result[name] = getattr(vm, name)
        for name in BATCH_FIELDNAMES:
            result[name] = getattr(vm, name)
        for name in ERROR_FIELDNAMES:
            result[name] = getattr(vm, name)
        result["has_error"] = vm.has_error()
        return result

    def batch_to_dataframe(self, vm: UnifiedViewModel) -> pd.DataFrame:
        rows = []
        for pr in vm._batch_results:
            factors = list(pr.top_factors or [])
            row = {
                "transaction_id": pr.transaction_id or "",
                "prediction_code": int(pr.class_label.value) if hasattr(pr.class_label, "value") else int(pr.class_label),
                "prediction_label": pr.class_label.name if hasattr(pr.class_label, "name") else str(pr.class_label),
                "fraud_probability": float(pr.fraud_probability),
                "legit_probability": float(1.0 - pr.fraud_probability),
                "risk_level": pr.risk_level.value if hasattr(pr.risk_level, "value") else str(pr.risk_level),
                "risk_summary": pr.risk_summary or "",
                "top_factor_1": factors[0].get("feature", "") if len(factors) > 0 else "",
                "top_factor_2": factors[1].get("feature", "") if len(factors) > 1 else "",
                "top_factor_3": factors[2].get("feature", "") if len(factors) > 2 else "",
                "process_status": pr.process_status.value if hasattr(pr.process_status, "value") else str(pr.process_status),
            }
            rows.append(row)
        df = pd.DataFrame(rows, columns=BATCH_DF_COLUMNS)
        return df

    def to_dict(self, vm: UnifiedViewModel) -> Dict[str, Any]:
        flat = self.flatten_for_display(vm)
        flat["_raw_error"] = vm._raw_error
        flat["_batch_result_count"] = len(vm._batch_results)
        return flat
