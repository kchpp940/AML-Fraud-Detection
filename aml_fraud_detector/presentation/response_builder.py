from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

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
        展平成一个稳定的 display 字典。

        所有字段判断（如 process_status → is_error、
        risk_explanation → risk_level/risk_summary/risk_contributors、
        校验告警聚合 → has_alerts）都在这一步完成，
        页面模板只做读取，不再重复判断。
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
            out.update({
                "is_error": s.process_status == ProcessStatus.ERROR,
                "error_reason": s.error_reason,
                "prediction_code": s.prediction,
                "prediction_label": s.class_label,
                "fraud_probability": s.fraud_probability,
                "legit_probability": s.legit_probability,
                "risk_level": re.risk_level.value if isinstance(re.risk_level, RiskLevel) else re.risk_level,
                "risk_summary": re.summary_text,
                "risk_contributors": list(re.top_contributors),
                "transaction_id": s.transaction_id,
                "process_status": s.process_status.value if isinstance(s.process_status, ProcessStatus) else s.process_status,
                "model_version_number": s.model_version,
            })

        b = getattr(response, "batch", None)
        if isinstance(b, BatchPredictionResult):
            out.update({
                "batch_total": b.total_count,
                "batch_fraud": b.fraud_count,
                "batch_legit": b.legit_count,
                "batch_fraud_rate": b.fraud_rate,
                "batch_is_error": b.process_status == ProcessStatus.ERROR,
                "batch_error_reason": b.error_reason,
            })

        return out

    # ------------------------------------------------------------------
    # 批量明细表：Streamlit / CSV 导出用
    # ------------------------------------------------------------------
    @staticmethod
    def batch_to_dataframe(response) -> pd.DataFrame:
        b = getattr(response, "batch", None)
        if not isinstance(b, BatchPredictionResult):
            return pd.DataFrame()
        return PredictionPipeline.batch_to_dataframe(b)

    # ------------------------------------------------------------------
    # 原始 dict 序列化
    # ------------------------------------------------------------------
    @staticmethod
    def to_dict(response) -> Dict[str, Any]:
        return asdict(response)


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
