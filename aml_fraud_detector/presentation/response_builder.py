from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from aml_fraud_detector.entity import (
    ProcessStatus,
    RiskLevel,
    PredictionResult,
    BatchPredictionResult,
    ModelVersionInfo,
    ValidationStatus,
)
from aml_fraud_detector.pipeline.prediction_pipeline import PredictionPipeline


# ---------------------------------------------------------------------------
# 展示专用的静态辅助（纯函数，不做业务判断，仅做 display label/class 的映射）
# ---------------------------------------------------------------------------

RISK_BADGE_CLASS: Dict[str, str] = {
    "Minimal": "risk-Minimal",
    "Low": "risk-Low",
    "Medium": "risk-Medium",
    "High": "risk-High",
    "Critical": "risk-Critical",
}

PREDICTION_BADGE_CLASS: Dict[int, str] = {
    0: "prediction-legit",
    1: "prediction-fraud",
}


def _risk_label(level: Any) -> str:
    if isinstance(level, RiskLevel):
        return level.value
    return str(level)


def _status_label(status: Any) -> str:
    if isinstance(status, ProcessStatus):
        if status == ProcessStatus.SUCCESS:
            return "OK"
        return "ERROR"
    s = str(status).lower()
    if s in ("success", "ok", "true"):
        return "OK"
    return "ERROR"


def _status_class(status: Any) -> str:
    return "status-ok" if _status_label(status) == "OK" else "status-error"


def _prediction_badge_label(pred: int, default_label: str = "") -> str:
    if default_label:
        return default_label
    return "Fraud" if int(pred) == 1 else "Not Fraud"


def _prediction_badge_class(pred: int) -> str:
    return PREDICTION_BADGE_CLASS.get(int(pred), "prediction-unknown")


def _risk_badge_class(level: Any) -> str:
    return RISK_BADGE_CLASS.get(_risk_label(level), "risk-Unknown")


def _contributors_to_display(contributors: List[Dict[str, Any]]) -> str:
    """把 contributor 列表转成页面直接渲染的纯文本（每行一条）。"""
    if not contributors:
        return ""
    lines = []
    for c in contributors:
        try:
            val = c.get("value", "")
            contrib = c.get("contribution", 0)
            feature = c.get("feature", "")
            lines.append(f"• {feature}={val}  →  {contrib:.5f}")
        except Exception:
            continue
    return "\n".join(lines)


class ResponseBuilder:
    """
    展示层适配器：消费 PredictionPipeline 产出的真实结果对象
    （PredictionResult / BatchPredictionResult / ValidationStatus /
    ModelVersionInfo），组装成 Flask / Streamlit 共用的稳定 display
    字典与 DataFrame。

    注意：本类不做任何业务逻辑（不做 hash 校验、不写风险解释、
    不做 inference），只负责字段提取与结构重组。
    """

    def __init__(self, pipeline: Optional[PredictionPipeline] = None):
        self._pipeline = pipeline if pipeline is not None else PredictionPipeline()

    # ------------------------------------------------------------------
    # 构建 view model：统一调用 pipeline，组装结果
    # ------------------------------------------------------------------
    def build_single(self, data: dict) -> "UnifiedViewModel":
        model_info = self._pipeline.get_model_version_info()
        validation = self._pipeline.validate_artifacts()
        prediction = self._pipeline.predict_single(data)
        return UnifiedViewModel(
            model=model_info,
            validation=validation,
            single=prediction,
        )

    def build_batch(self, df: pd.DataFrame) -> "UnifiedViewModel":
        model_info = self._pipeline.get_model_version_info()
        validation = self._pipeline.validate_artifacts()
        batch = self._pipeline.predict_batch(df)
        return UnifiedViewModel(
            model=model_info,
            validation=validation,
            batch=batch,
        )

    def build_validation_only(self) -> "UnifiedViewModel":
        model_info = self._pipeline.get_model_version_info()
        validation = self._pipeline.validate_artifacts()
        return UnifiedViewModel(
            model=model_info,
            validation=validation,
        )

    # ------------------------------------------------------------------
    # 扁平展示字典：Flask / Streamlit 直接消费
    # ------------------------------------------------------------------
    @staticmethod
    def flatten_for_display(response) -> Dict[str, Any]:
        """
        把任意一种 response（single / batch / validation-only）
        展平成一个稳定的 display 字典，里面**包含所有展示专用的
        预计算字段（label / class / banner HTML 片段）**。
        页面模板和 Streamlit 渲染函数只做读取，不再判断任何
        process_status / risk_explanation / prediction_code。
        """
        out: Dict[str, Any] = {}

        model = getattr(response, "model", None)
        if isinstance(model, ModelVersionInfo):
            out.update({
                "model_version": model.model_version,
                "model_name": model.best_model_name,
                "training_time": model.training_time,
                "selection_metric": model.selection_metric,
                "best_metric_value": model.best_metric_value,
                "feature_contract_version": model.feature_contract_version,
                "artifact_path": model.artifact_path,
            })

        v = getattr(response, "validation", None)
        if isinstance(v, ValidationStatus):
            out.update({
                "validation_valid": v.is_valid,
                "validation_errors": list(v.errors),
                "validation_warnings": list(v.warnings),
                "validation_details": dict(v.details),
                "has_alerts": bool(v.errors or v.warnings),
            })

        s = getattr(response, "single", None)
        if isinstance(s, PredictionResult):
            re = s.risk_explanation
            risk_label = _risk_label(re.risk_level)
            is_error = s.process_status == ProcessStatus.ERROR

            out.update({
                # 原始字段（保留，供需要原始值的消费者使用）
                "is_error": is_error,
                "error_reason": s.error_reason,
                "prediction_code": s.prediction,
                "prediction_label": s.class_label,
                "fraud_probability": s.fraud_probability,
                "legit_probability": s.legit_probability,
                "risk_level": risk_label,
                "risk_summary": re.summary_text,
                "risk_contributors": list(re.top_contributors),
                "transaction_id": s.transaction_id,
                "process_status": s.process_status.value if isinstance(s.process_status, ProcessStatus) else s.process_status,
                "model_version_number": s.model_version,

                # ================================================================
                # 展示专用字段（页面无需再做判断，直接渲染）
                # ================================================================

                # 预测结果 badge
                "single_prediction_badge_label": _prediction_badge_label(s.prediction, s.class_label),
                "single_prediction_badge_class": _prediction_badge_class(s.prediction),
                # Streamlit 专用：应该用 st.success 还是 st.error 风格展示
                "single_prediction_streamlit_style": (
                    "error" if s.prediction == 1 else "success"
                ) if not is_error else "",

                # 风险等级 badge
                "single_risk_badge_label": risk_label,
                "single_risk_badge_class": _risk_badge_class(re.risk_level),

                # 风险 contributors 纯文本展示
                "single_risk_contributors_display": _contributors_to_display(re.top_contributors),

                # 错误 banner（页面直接渲染即可，无需判断是否有错误）
                "single_error_banner": s.error_reason if is_error else "",
            })

        b = getattr(response, "batch", None)
        if isinstance(b, BatchPredictionResult):
            batch_is_error = b.process_status == ProcessStatus.ERROR
            out.update({
                # 原始字段
                "batch_total": b.total_count,
                "batch_fraud": b.fraud_count,
                "batch_legit": b.legit_count,
                "batch_fraud_rate": b.fraud_rate,
                "batch_is_error": batch_is_error,
                "batch_error_reason": b.error_reason,

                # ================================================================
                # 展示专用字段
                # ================================================================
                "batch_summary_label_total": f"{b.total_count}",
                "batch_summary_label_legit": f"{b.legit_count}",
                "batch_summary_label_fraud": f"{b.fraud_count}",
                "batch_summary_label_fraud_rate": f"{b.fraud_rate * 100:.2f}%",
                "batch_summary_card_fraud_class": (
                    "metric-card metric-card-danger"
                    if b.fraud_count > 0 else "metric-card metric-card-safe"
                ),
                "batch_row_count_label": f"Input rows: {b.total_count}",
                "batch_error_banner": b.error_reason if batch_is_error else "",
            })

        return out

    # ------------------------------------------------------------------
    # 批量明细表（每行含展示专用列：status / prediction / risk 的 label + class）
    # ------------------------------------------------------------------
    @staticmethod
    def batch_to_display_dataframe(response) -> pd.DataFrame:
        """
        给 Flask/Streamlit 批量详情页直接消费的 DataFrame，
        所有状态判断、颜色 class、错误展示都已预生成。
        页面层遍历行后只输出字段，不再做任何判断。
        """
        b = getattr(response, "batch", None)
        if not isinstance(b, BatchPredictionResult):
            return pd.DataFrame(columns=BATCH_DISPLAY_COLUMNS)

        records: List[Dict[str, Any]] = []
        for idx, p in enumerate(b.predictions, start=1):
            risk_lbl = _risk_label(p.risk_explanation.risk_level)
            is_error_row = p.process_status == ProcessStatus.ERROR
            records.append({
                "row_index": idx,

                # 基础标识列
                "transaction_id": p.transaction_id or "",

                # 预测列（展示专用，含 label + class）
                "prediction_code": p.prediction,
                "prediction_label": p.class_label,
                "prediction_badge_label": _prediction_badge_label(p.prediction, p.class_label),
                "prediction_badge_class": _prediction_badge_class(p.prediction),

                # 概率列
                "fraud_probability": p.fraud_probability,
                "fraud_probability_display": f"{p.fraud_probability:.4f}",
                "legit_probability": p.legit_probability,
                "legit_probability_display": f"{p.legit_probability:.4f}",

                # 风险等级列（展示专用）
                "risk_level_label": risk_lbl,
                "risk_level_class": _risk_badge_class(p.risk_explanation.risk_level),
                "risk_summary_display": p.risk_explanation.summary_text,
                "risk_contributors_display": _contributors_to_display(p.risk_explanation.top_contributors),
                "has_contributors": bool(p.risk_explanation.top_contributors),

                # 处理状态列（展示专用）
                "status_label": _status_label(p.process_status),
                "status_class": _status_class(p.process_status),
                "is_error_row": is_error_row,
                "error_message_display": p.error_reason if is_error_row else "",

                # Fraud 行高亮 class（页面直接用在 <tr class=...>）
                "row_tr_class": "fraud-row" if p.prediction == 1 else "",
            })
        df = pd.DataFrame(records)
        # 保证列顺序固定（即使空表也有这些列）
        return df.reindex(columns=BATCH_DISPLAY_COLUMNS)

    @staticmethod
    def batch_to_dataframe(response) -> pd.DataFrame:
        """保留旧名称（与 PredictionPipeline 一致），内部直接调用 display DataFrame。"""
        return ResponseBuilder.batch_to_display_dataframe(response)

    # ------------------------------------------------------------------
    # 原始 dict 序列化
    # ------------------------------------------------------------------
    @staticmethod
    def to_dict(response) -> Dict[str, Any]:
        return asdict(response)


# ---------------------------------------------------------------------------
# 批量展示列的固定顺序（保证页面/Streamlit/测试 都用同一份契约）
# ---------------------------------------------------------------------------
BATCH_DISPLAY_COLUMNS: List[str] = [
    "row_index",
    "transaction_id",
    "prediction_code",
    "prediction_label",
    "prediction_badge_label",
    "prediction_badge_class",
    "fraud_probability",
    "fraud_probability_display",
    "legit_probability",
    "legit_probability_display",
    "risk_level_label",
    "risk_level_class",
    "risk_summary_display",
    "risk_contributors_display",
    "has_contributors",
    "status_label",
    "status_class",
    "is_error_row",
    "error_message_display",
    "row_tr_class",
]


# ----------------------------------------------------------------------
# 统一视图模型：Flask / Streamlit 共用的稳定结构
# ----------------------------------------------------------------------
from dataclasses import dataclass, field  # noqa: E402


@dataclass
class UnifiedViewModel:
    model: ModelVersionInfo = field(default_factory=ModelVersionInfo)
    validation: ValidationStatus = field(default_factory=ValidationStatus)
    single: Optional[PredictionResult] = None
    batch: Optional[BatchPredictionResult] = None
