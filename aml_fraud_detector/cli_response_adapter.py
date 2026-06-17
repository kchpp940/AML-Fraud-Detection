from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from aml_fraud_detector.entity import (
    BatchPredictionResult,
    ModelVersionInfo,
    PredictionResult,
    ProcessStatus,
    RiskLevel,
    TrainingPipelineResult,
    ValidationStatus,
)
from aml_fraud_detector.exception import AMLException


MODEL_VERSION_KEYS = [
    "model_version",
    "model_name",
    "training_time",
    "selection_metric",
    "best_metric_value",
    "feature_contract_version",
]

VALIDATION_KEYS = [
    "validation_valid",
    "validation_errors",
    "validation_warnings",
    "validation_details",
    "has_alerts",
]

SINGLE_KEYS = [
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

BATCH_KEYS = [
    "batch_total",
    "batch_fraud",
    "batch_legit",
    "batch_fraud_rate",
    "batch_is_error",
    "batch_error_reason",
]

ERROR_KEYS = [
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

_CATEGORY_DISPLAY = {
    "data_quality": "数据质量",
    "input_validation": "输入校验",
    "feature_alignment": "特征对齐",
    "model_loading": "模型加载",
    "metadata_validation": "元数据校验",
    "prediction_error": "预测执行",
    "explanation_error": "解释生成",
    "config_error": "配置错误",
    "pipeline_error": "管道执行",
    "internal_error": "内部错误",
}


def _empty_envelope() -> Dict[str, Any]:
    return {
        "model_version": "",
        "model_name": "",
        "training_time": "",
        "selection_metric": "",
        "best_metric_value": "",
        "feature_contract_version": "",
        "validation_valid": True,
        "validation_errors": [],
        "validation_warnings": [],
        "validation_details": [],
        "has_alerts": False,
        "is_error": False,
        "error_reason": "",
        "prediction_code": 0,
        "prediction_label": "",
        "fraud_probability": 0.0,
        "legit_probability": 0.0,
        "risk_level": "",
        "risk_summary": "",
        "risk_contributors": [],
        "transaction_id": "",
        "process_status": "success",
        "batch_total": 0,
        "batch_fraud": 0,
        "batch_legit": 0,
        "batch_fraud_rate": "0.00%",
        "batch_is_error": False,
        "batch_error_reason": "",
        "error_code": "",
        "error_category": "",
        "error_category_display": "",
        "error_message": "",
        "error_field": "",
        "error_value": "",
        "error_context": {},
        "error_trace_id": "",
        "error_severity": "",
    }


def _fill_model(e: Dict[str, Any], mvi: Optional[ModelVersionInfo]) -> None:
    if mvi is None:
        return
    e["model_version"] = str(mvi.model_version or "")
    e["model_name"] = mvi.model_name or ""
    e["training_time"] = mvi.training_time or ""
    e["selection_metric"] = mvi.selection_metric or ""
    e["best_metric_value"] = str(mvi.best_metric_value or "")
    e["feature_contract_version"] = mvi.feature_schema_version or ""


def _fill_validation(e: Dict[str, Any], vs: Optional[ValidationStatus]) -> None:
    if vs is None:
        return
    e["validation_valid"] = vs.is_valid
    e["validation_errors"] = list(vs.errors)
    e["validation_warnings"] = list(vs.warnings)
    e["validation_details"] = [{"artifact": a} for a in vs.checked_artifacts]
    e["has_alerts"] = bool(vs.errors or vs.warnings)


def _fill_error(e: Dict[str, Any], exc: AMLException, trace_id: str) -> None:
    ed = exc.error_detail
    e["is_error"] = True
    e["error_reason"] = exc.message or ""
    e["process_status"] = ProcessStatus.ERROR.value
    e["error_code"] = ed.error_code.value if hasattr(ed.error_code, "value") else str(ed.error_code)
    cat_val = ed.error_category.value if hasattr(ed.error_category, "value") else str(ed.error_category)
    e["error_category"] = cat_val
    e["error_category_display"] = _CATEGORY_DISPLAY.get(cat_val, cat_val)
    e["error_message"] = ed.message or ""
    e["error_field"] = ed.field or ""
    e["error_value"] = str(ed.value) if ed.value is not None else ""
    e["error_context"] = dict(ed.context) if ed.context else {}
    e["error_trace_id"] = trace_id
    e["error_severity"] = _severity(ed.error_code)


def _severity(code) -> str:
    val = code.value if hasattr(code, "value") else str(code)
    if val.startswith("E1"):
        return "critical"
    if val.startswith("E2"):
        return "error"
    if val.startswith("E3"):
        return "critical"
    if val.startswith("E4"):
        return "critical"
    if val.startswith("E5"):
        return "error"
    if val.startswith("E6"):
        return "error"
    if val.startswith("E7"):
        return "warning"
    if val.startswith("E8"):
        return "error"
    if val.startswith("E9"):
        return "error"
    return "critical"


def adapt_validation(
    mvi: Optional[ModelVersionInfo] = None,
    vs: Optional[ValidationStatus] = None,
    trace_id: str = "",
) -> Dict[str, Any]:
    e = _empty_envelope()
    _fill_model(e, mvi)
    _fill_validation(e, vs)
    e["process_status"] = ProcessStatus.SUCCESS.value
    return e


def adapt_error(
    exc: AMLException,
    mvi: Optional[ModelVersionInfo] = None,
    vs: Optional[ValidationStatus] = None,
    trace_id: str = "",
) -> Dict[str, Any]:
    e = _empty_envelope()
    _fill_model(e, mvi)
    _fill_validation(e, vs)
    _fill_error(e, exc, trace_id)
    return e


def adapt_training(
    result: TrainingPipelineResult,
    trace_id: str = "",
) -> Dict[str, Any]:
    e = _empty_envelope()
    e["process_status"] = result.process_status.value
    if not result.is_success():
        e["is_error"] = True
        e["error_reason"] = result.error_reason or ""
        if result.error_detail:
            _fill_error_from_detail(e, result.error_detail, trace_id)
    return e


def adapt_single(
    result: PredictionResult,
    mvi: Optional[ModelVersionInfo] = None,
    vs: Optional[ValidationStatus] = None,
    trace_id: str = "",
) -> Dict[str, Any]:
    e = _empty_envelope()
    _fill_model(e, mvi)
    _fill_validation(e, vs)
    e["process_status"] = result.process_status.value
    if not result.is_success():
        e["is_error"] = True
        e["error_reason"] = result.error_reason or ""
        if result.error_detail:
            _fill_error_from_detail(e, result.error_detail, trace_id)
        return e
    e["prediction_code"] = result.prediction
    e["prediction_label"] = str(result.class_label)
    e["fraud_probability"] = result.fraud_probability
    e["legit_probability"] = result.legit_probability
    e["risk_level"] = result.risk_level.value if hasattr(result.risk_level, "value") else str(result.risk_level)
    e["risk_contributors"] = list(result.top_factors or [])
    e["transaction_id"] = result.transaction_id or ""
    return e


def adapt_batch(
    result: BatchPredictionResult,
    mvi: Optional[ModelVersionInfo] = None,
    vs: Optional[ValidationStatus] = None,
    trace_id: str = "",
) -> Dict[str, Any]:
    e = _empty_envelope()
    _fill_model(e, mvi)
    _fill_validation(e, vs)
    e["process_status"] = result.process_status.value
    if not result.is_success():
        e["is_error"] = True
        e["batch_is_error"] = True
        e["batch_error_reason"] = result.error_reason or ""
        e["error_reason"] = result.error_reason or ""
        if result.error_detail:
            _fill_error_from_detail(e, result.error_detail, trace_id)
        return e
    e["batch_total"] = result.total_count
    e["batch_fraud"] = result.fraud_count
    e["batch_legit"] = result.total_count - result.fraud_count
    e["batch_fraud_rate"] = f"{result.fraud_rate * 100:.2f}%"
    if result.fraud_rate >= 0.7:
        e["risk_level"] = RiskLevel.CRITICAL.value
    elif result.fraud_rate >= 0.5:
        e["risk_level"] = RiskLevel.HIGH.value
    elif result.fraud_rate >= 0.3:
        e["risk_level"] = RiskLevel.MEDIUM.value
    else:
        e["risk_level"] = RiskLevel.LOW.value
    return e


def batch_to_dataframe(result: BatchPredictionResult) -> pd.DataFrame:
    rows = []
    for pr in result.predictions:
        factors = list(pr.top_factors or [])
        rows.append({
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
        })
    return pd.DataFrame(rows, columns=BATCH_DF_COLUMNS)


def _fill_error_from_detail(e: Dict[str, Any], ed, trace_id: str) -> None:
    e["error_code"] = ed.error_code.value if hasattr(ed.error_code, "value") else str(ed.error_code)
    cat_val = ed.error_category.value if hasattr(ed.error_category, "value") else str(ed.error_category)
    e["error_category"] = cat_val
    e["error_category_display"] = _CATEGORY_DISPLAY.get(cat_val, cat_val)
    e["error_message"] = ed.message or ""
    e["error_field"] = ed.field or ""
    e["error_value"] = str(ed.value) if ed.value is not None else ""
    e["error_context"] = dict(ed.context) if ed.context else {}
    e["error_trace_id"] = trace_id
    e["error_severity"] = _severity(ed.error_code)
