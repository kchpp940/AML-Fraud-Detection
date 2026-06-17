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
    HTTP_STATUS_CODES,
)
from aml_fraud_detector.entity.artifact_entity import (
    PROCESS_STATUS_SUCCESS,
    PROCESS_STATUS_ERROR,
)

ERROR_CATEGORY_DISPLAY = {
    ErrorCategory.DATA_QUALITY: "数据质量",
    ErrorCategory.INPUT_VALIDATION: "输入校验",
    ErrorCategory.FEATURE_ALIGNMENT: "特征对齐",
    ErrorCategory.MODEL_LOADING: "模型加载",
    ErrorCategory.METADATA_VALIDATION: "元数据校验",
    ErrorCategory.PREDICTION_ERROR: "预测执行",
    ErrorCategory.EXPLANATION_ERROR: "解释生成",
    ErrorCategory.CONFIG_ERROR: "配置错误",
    ErrorCategory.PIPELINE_ERROR: "管道执行",
    ErrorCategory.INTERNAL_ERROR: "内部错误",
}

ERROR_SEVERITY = {
    ErrorCode.DATA_SOURCE_NOT_FOUND: "critical",
    ErrorCode.DATA_EMPTY: "critical",
    ErrorCode.DATA_MISSING_COLUMNS: "critical",
    ErrorCode.DATA_INVALID_DTYPE: "warning",
    ErrorCode.DATA_OUT_OF_RANGE: "warning",
    ErrorCode.DATA_CORRUPTED: "critical",
    ErrorCode.DATA_SAMPLING_ERROR: "error",
    ErrorCode.INPUT_MISSING_FIELD: "error",
    ErrorCode.INPUT_INVALID_TYPE: "error",
    ErrorCode.INPUT_INVALID_FORMAT: "error",
    ErrorCode.INPUT_OUT_OF_RANGE: "warning",
    ErrorCode.INPUT_EMPTY_VALUE: "error",
    ErrorCode.FEATURE_MISMATCH: "critical",
    ErrorCode.FEATURE_MISSING: "critical",
    ErrorCode.FEATURE_UNEXPECTED: "warning",
    ErrorCode.FEATURE_TRANSFORM_FAILED: "error",
    ErrorCode.FEATURE_ENCODING_FAILED: "error",
    ErrorCode.MODEL_FILE_NOT_FOUND: "critical",
    ErrorCode.MODEL_CORRUPTED: "critical",
    ErrorCode.MODEL_INCOMPATIBLE: "critical",
    ErrorCode.MODEL_NOT_TRAINED: "critical",
    ErrorCode.PREPROCESSOR_FILE_NOT_FOUND: "critical",
    ErrorCode.PREPROCESSOR_CORRUPTED: "critical",
    ErrorCode.METADATA_FILE_NOT_FOUND: "warning",
    ErrorCode.METADATA_CORRUPTED: "error",
    ErrorCode.METADATA_SCHEMA_MISMATCH: "error",
    ErrorCode.METADATA_VERSION_MISMATCH: "warning",
    ErrorCode.MANIFEST_INTEGRITY_FAILED: "critical",
    ErrorCode.ARTIFACT_MISSING: "critical",
    ErrorCode.PREDICTION_FAILED: "error",
    ErrorCode.PREDICTION_SHAPE_MISMATCH: "error",
    ErrorCode.BATCH_PREDICTION_FAILED: "error",
    ErrorCode.EXPLANATION_NOT_SUPPORTED: "info",
    ErrorCode.EXPLANATION_FAILED: "warning",
    ErrorCode.EXPLANATION_MODEL_INCOMPATIBLE: "warning",
    ErrorCode.CONFIG_FILE_NOT_FOUND: "error",
    ErrorCode.CONFIG_PARSE_FAILED: "error",
    ErrorCode.CONFIG_INVALID_VALUE: "warning",
    ErrorCode.CONFIG_MISSING_REQUIRED: "error",
    ErrorCode.PIPELINE_STEP_FAILED: "error",
    ErrorCode.PIPELINE_INVALID_STATE: "error",
    ErrorCode.PIPELINE_DEPENDENCY_MISSING: "error",
    ErrorCode.INTERNAL_UNEXPECTED: "critical",
}
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
from aml_fraud_detector.presentation.display_builders import (
    BATCH_DF_COLUMNS as _BATCH_DF_COLUMNS,
    batch_predictions_to_dataframe,
)


BATCH_DF_COLUMNS = _BATCH_DF_COLUMNS

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
        vm.feature_contract_version = mvi.feature_schema_version or ""

    def _fill_validation_fields(
        self, vm: UnifiedViewModel, vs: Optional[ValidationStatus]
    ) -> None:
        if vs is None:
            return
        vm.validation_valid = vs.is_valid
        vm.validation_errors = list(vs.errors)
        vm.validation_warnings = list(vs.warnings)
        vm.validation_details = [{"artifact": a} for a in vs.checked_artifacts]
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
        vm.error_field = error_detail.field or ""
        vm.error_value = str(error_detail.value) if error_detail.value is not None else ""
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

        vm.prediction_code = int(prediction_result.prediction)
        vm.prediction_label = str(prediction_result.class_label)
        vm.fraud_probability = float(prediction_result.fraud_probability)
        vm.legit_probability = float(prediction_result.legit_probability)
        vm.risk_level = prediction_result.risk_level.value if hasattr(prediction_result.risk_level, "value") else str(prediction_result.risk_level)
        vm.risk_summary = ""
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
        if batch_result.fraud_rate >= 0.7:
            vm.risk_level = RiskLevel.CRITICAL.value
        elif batch_result.fraud_rate >= 0.5:
            vm.risk_level = RiskLevel.HIGH.value
        elif batch_result.fraud_rate >= 0.3:
            vm.risk_level = RiskLevel.MEDIUM.value
        else:
            vm.risk_level = RiskLevel.LOW.value
        return vm

    def build_error(
        self,
        error: AMLException,
        model_version_info: Optional[ModelVersionInfo] = None,
        validation_status: Optional[ValidationStatus] = None,
        trace_id: Optional[str] = None,
    ) -> UnifiedViewModel:
        tid = trace_id or self._new_trace_id()
        vm = UnifiedViewModel()
        self._fill_model_fields(vm, model_version_info)
        self._fill_validation_fields(vm, validation_status)
        vm.is_error = True
        vm.error_reason = error.message or ""
        vm.process_status = ProcessStatus.ERROR.value
        self._fill_error_fields(vm, error.error_detail, tid)
        return vm

    def build_training(
        self,
        training_result: Optional[Any] = None,
        error: Optional[AMLException] = None,
        trace_id: Optional[str] = None,
    ) -> UnifiedViewModel:
        tid = trace_id or self._new_trace_id()
        vm = UnifiedViewModel()
        if error is not None:
            vm.is_error = True
            vm.error_reason = error.message or ""
            vm.process_status = ProcessStatus.ERROR.value
            self._fill_error_fields(vm, error.error_detail, tid)
            return vm
        if training_result is None:
            vm.is_error = True
            vm.error_reason = "训练结果为空"
            vm.process_status = ProcessStatus.ERROR.value
            return vm
        if not training_result.is_success():
            vm.is_error = True
            vm.error_reason = training_result.error_reason or ""
            vm.process_status = ProcessStatus.ERROR.value
            if training_result.error_detail:
                self._fill_error_fields(vm, training_result.error_detail, tid)
            return vm
        vm.process_status = ProcessStatus.SUCCESS.value
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
                "prediction_code": int(pr.prediction),
                "prediction_label": str(pr.class_label),
                "fraud_probability": float(pr.fraud_probability),
                "legit_probability": float(pr.legit_probability),
                "risk_level": pr.risk_level.value if hasattr(pr.risk_level, "value") else str(pr.risk_level),
                "risk_summary": "",
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
