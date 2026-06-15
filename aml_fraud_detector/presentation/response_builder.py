from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aml_fraud_detector.logger import logging
from aml_fraud_detector.pipeline.prediction_pipeline import PredictionPipeline


REQUIRED_INPUT_FIELDS = [
    "from_bank",
    "account",
    "to_bank",
    "account_1",
    "amount_received",
    "receiving_currency",
    "payment_currency",
    "payment_format",
    "day",
]

FOUR_ARTIFACT_FILENAMES = [
    "model.pkl",
    "preprocessor.pkl",
    "feature_metadata.json",
    "model_metadata.json",
]

ARTIFACT_TO_DIGEST_KEY = {
    "model.pkl": "digest_model_pkl",
    "preprocessor.pkl": "digest_preprocessor_pkl",
    "feature_metadata.json": "digest_feature_metadata_json",
    "model_metadata.json": "digest_model_metadata_json",
}


def _sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


@dataclass
class RiskExplanationViewModel:
    fraud_probability: float = 0.0
    risk_level: str = "Low"
    summary_text: str = ""
    top_contributors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SinglePredictionViewModel:
    prediction: int = -1
    class_label: str = ""
    fraud_probability: float = 0.0
    legit_probability: float = 0.0
    model_version: int = 0
    transaction_id: str = ""
    process_status: str = "success"
    error_reason: Optional[str] = None
    risk: RiskExplanationViewModel = field(default_factory=RiskExplanationViewModel)


@dataclass
class BatchPredictionViewModel:
    total: int = 0
    fraud: int = 0
    legit: int = 0
    fraud_rate: float = 0.0
    process_status: str = "success"
    error_reason: Optional[str] = None
    rows: List[SinglePredictionViewModel] = field(default_factory=list)


@dataclass
class ValidationViewModel:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelVersionViewModel:
    model_version: int = 0
    best_model_name: str = ""
    training_time: str = ""
    selection_metric: str = ""
    best_metric_value: float = 0.0
    feature_contract_version: str = ""
    artifact_path: str = ""


@dataclass
class UnifiedViewModel:
    model: ModelVersionViewModel = field(default_factory=ModelVersionViewModel)
    validation: ValidationViewModel = field(default_factory=ValidationViewModel)
    single: Optional[SinglePredictionViewModel] = None
    batch: Optional[BatchPredictionViewModel] = None


def _determine_risk_level(p: float) -> str:
    if p >= 0.8:
        return "Critical"
    if p >= 0.6:
        return "High"
    if p >= 0.3:
        return "Medium"
    if p >= 0.01:
        return "Low"
    return "Minimal"


def _build_risk_summary(p: float, level: str) -> str:
    pct = f"{p * 100:.1f}%"
    if level == "Minimal":
        return f"该笔交易欺诈风险极低（欺诈概率 {pct}），风险等级：{level}。当前特征未见明显异常信号，可按正常流程处理。"
    if level == "Low":
        return f"该笔交易欺诈风险较低（欺诈概率 {pct}），风险等级：{level}。建议结合业务规则做常规复核。"
    if level == "Medium":
        return f"该笔交易存在中等欺诈风险（欺诈概率 {pct}），风险等级：{level}。建议人工复核账户历史与交易背景。"
    if level == "High":
        return f"该笔交易欺诈风险较高（欺诈概率 {pct}），风险等级：{level}。建议优先进入人工审核队列，并对账户进行临时限制。"
    return f"该笔交易存在严重欺诈嫌疑（欺诈概率 {pct}），风险等级：{level}。建议立即冻结交易并启动调查流程。"


class ResponseBuilder:
    """
    适配层：消费 PredictionPipeline 的真实输出（predict → ndarray,
    predict_proba → ndarray）以及 artifacts 目录中的元数据，整理成
    Flask / Streamlit 共用的稳定 view model。
    """

    def __init__(self, pipeline: Optional[PredictionPipeline] = None, artifacts_dir: str = "artifacts"):
        self._pipeline = pipeline if pipeline is not None else PredictionPipeline()
        self._artifacts_dir = artifacts_dir

        self._model_metadata: Dict[str, Any] = {}
        self._feature_metadata: Dict[str, Any] = {}
        self._manifest: Dict[str, Any] = {}
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        mm_path = os.path.join(self._artifacts_dir, "model_metadata.json")
        if os.path.exists(mm_path):
            with open(mm_path, "r", encoding="utf-8") as f:
                self._model_metadata = json.load(f)
        fm_path = os.path.join(self._artifacts_dir, "feature_metadata.json")
        if os.path.exists(fm_path):
            with open(fm_path, "r", encoding="utf-8") as f:
                self._feature_metadata = json.load(f)
        manifest_path = os.path.join(self._artifacts_dir, "artifact_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                self._manifest = json.load(f)
        self._loaded = True

    # ------------------------------------------------------------------
    # 元数据 / 校验
    # ------------------------------------------------------------------
    def _build_model_version(self) -> ModelVersionViewModel:
        self._ensure_loaded()
        mm = self._model_metadata
        return ModelVersionViewModel(
            model_version=int(mm.get("model_version", 0)),
            best_model_name=str(mm.get("best_model_name", "")),
            training_time=str(mm.get("training_time", "")),
            selection_metric=str(mm.get("selection_metric", "")),
            best_metric_value=float(mm.get("best_metric_value", 0.0)),
            feature_contract_version=str(mm.get("feature_schema_version", "")),
            artifact_path=str(mm.get("artifact_path", "")),
        )

    def _build_validation(self) -> ValidationViewModel:
        self._ensure_loaded()
        mm = self._model_metadata
        fm = self._feature_metadata
        manifest = self._manifest

        errors: List[str] = []
        warnings: List[str] = []

        details: Dict[str, Any] = {
            "artifacts_dir": os.path.abspath(self._artifacts_dir),
            "model_version": int(mm.get("model_version", 0)) if mm else 0,
            "best_model_name": str(mm.get("best_model_name", "")) if mm else "",
            "feature_contract_version": str(fm.get("contract_version", "")) if fm else "",
            "numerical_features": list(fm.get("numerical_features", [])) if fm else [],
            "categorical_features": list(fm.get("categorical_features", [])) if fm else [],
            "manifest_artifacts": list(manifest.get("artifacts", {}).keys()) if manifest else [],
            "manifest_present": bool(manifest),
        }

        # 四文件存在性（作为基础检查）
        for fname in FOUR_ARTIFACT_FILENAMES:
            fpath = os.path.join(self._artifacts_dir, fname)
            if not os.path.exists(fpath):
                errors.append(f"Missing artifact: {fname}")

        # 计算并记录四个核心 artifacts 的当前 digest
        for fname in FOUR_ARTIFACT_FILENAMES:
            fpath = os.path.join(self._artifacts_dir, fname)
            key = ARTIFACT_TO_DIGEST_KEY[fname]
            if os.path.exists(fpath):
                try:
                    actual_digest = _sha256_file(fpath)
                except OSError as exc:
                    actual_digest = ""
                    errors.append(f"{fname} digest computation failed: {exc}")
                details[key] = actual_digest

                # 与 manifest 中的记录对比（若有）
                if manifest and fname in manifest.get("artifacts", {}):
                    expected = manifest["artifacts"][fname].get("digest", "")
                    if expected and expected != actual_digest:
                        errors.append(
                            f"{fname} digest mismatch: manifest={expected} actual={actual_digest}"
                        )
                    expected_size = manifest["artifacts"][fname].get("size")
                    actual_size = os.path.getsize(fpath)
                    if expected_size is not None and int(expected_size) != actual_size:
                        warnings.append(
                            f"{fname} size differs from manifest: manifest={int(expected_size)} actual={actual_size}"
                        )
                else:
                    warnings.append(f"Manifest has no entry for {fname}; cannot verify integrity")
            else:
                details[key] = ""

        is_valid = not errors
        logging.info(
            f"Unified validation: artifact_valid={is_valid}, "
            f"artifact_errors={len(errors)}, artifact_warnings={len(warnings)}"
        )
        return ValidationViewModel(is_valid=is_valid, errors=errors, warnings=warnings, details=details)

    # ------------------------------------------------------------------
    # 风险解释
    # ------------------------------------------------------------------
    def _explain_risk(
        self,
        fraud_probability: float,
        input_row: Optional[pd.Series] = None,
    ) -> RiskExplanationViewModel:
        level = _determine_risk_level(fraud_probability)
        summary = _build_risk_summary(fraud_probability, level)
        contributors: List[Dict[str, Any]] = []

        # 基于输入特征的简单启发式边际贡献（不依赖 pipeline 行为变更）
        if input_row is not None and fraud_probability >= 0.01:
            for col in ["amount_received", "from_bank", "to_bank"]:
                if col in input_row.index:
                    try:
                        val = float(input_row[col])
                    except (TypeError, ValueError):
                        continue
                    # 归一化贡献分数：以欺诈概率 * 相对权重作展示
                    weight = {
                        "amount_received": 0.35,
                        "from_bank": 0.15,
                        "to_bank": 0.15,
                    }.get(col, 0.1)
                    score = round(fraud_probability * weight, 5)
                    if score > 1e-6:
                        contributors.append({
                            "feature": col,
                            "value": val,
                            "contribution": score,
                            "direction": "push_towards_fraud",
                        })
            # 按贡献度降序（最多 Top 5）
            contributors.sort(key=lambda x: x["contribution"], reverse=True)
            contributors = contributors[:5]

        return RiskExplanationViewModel(
            fraud_probability=float(fraud_probability),
            risk_level=level,
            summary_text=summary,
            top_contributors=contributors,
        )

    # ------------------------------------------------------------------
    # 输入校验（防止 inference 失败）
    # ------------------------------------------------------------------
    def _validate_required_fields(self, data: Dict[str, Any]) -> Optional[str]:
        for field in REQUIRED_INPUT_FIELDS:
            if field not in data:
                return f"Input dict missing required field: '{field}'"
        return None

    # ------------------------------------------------------------------
    # 构建统一 view model
    # ------------------------------------------------------------------
    def build_single(self, data: Dict[str, Any]) -> UnifiedViewModel:
        model = self._build_model_version()
        validation = self._build_validation()

        single = SinglePredictionViewModel(model_version=model.model_version)

        field_err = self._validate_required_fields(data)
        if field_err is not None:
            single.process_status = "error"
            single.error_reason = field_err
            single.transaction_id = data.get("transaction_id") or str(uuid.uuid4())
            return UnifiedViewModel(model=model, validation=validation, single=single, batch=None)

        try:
            df = pd.DataFrame([data])
            pred = self._pipeline.predict(df)
            proba = self._pipeline.predict_proba(df)
            single.prediction = int(pred[0])
            single.class_label = "Fraud" if single.prediction == 1 else "Not Fraud"
            single.legit_probability = float(proba[0][0]) if proba.shape[1] > 0 else 0.0
            single.fraud_probability = float(proba[0][1]) if proba.shape[1] > 1 else 0.0
            single.transaction_id = data.get("transaction_id") or str(uuid.uuid4())
            single.process_status = "success"
            single.risk = self._explain_risk(single.fraud_probability, input_row=df.iloc[0])
        except Exception as exc:
            logging.exception("predict_single failed")
            single.process_status = "error"
            single.error_reason = str(exc)
            single.transaction_id = data.get("transaction_id") or str(uuid.uuid4())

        return UnifiedViewModel(model=model, validation=validation, single=single, batch=None)

    def build_batch(self, df: pd.DataFrame) -> UnifiedViewModel:
        model = self._build_model_version()
        validation = self._build_validation()
        batch = BatchPredictionViewModel()

        try:
            preds = self._pipeline.predict(df)
            probas = self._pipeline.predict_proba(df)
            rows: List[SinglePredictionViewModel] = []
            for i in range(len(df)):
                row = SinglePredictionViewModel(model_version=model.model_version)
                row.prediction = int(preds[i])
                row.class_label = "Fraud" if row.prediction == 1 else "Not Fraud"
                row.legit_probability = float(probas[i][0]) if probas.shape[1] > 0 else 0.0
                row.fraud_probability = float(probas[i][1]) if probas.shape[1] > 1 else 0.0
                row.transaction_id = (
                    str(df.iloc[i]["transaction_id"])
                    if "transaction_id" in df.columns
                    else str(uuid.uuid4())
                )
                row.process_status = "success"
                row.risk = self._explain_risk(row.fraud_probability, input_row=df.iloc[i])
                rows.append(row)

            batch.total = len(rows)
            batch.fraud = sum(1 for r in rows if r.prediction == 1)
            batch.legit = batch.total - batch.fraud
            batch.fraud_rate = (batch.fraud / batch.total) if batch.total > 0 else 0.0
            batch.rows = rows
            batch.process_status = "success"
        except Exception as exc:
            logging.exception("predict_batch failed")
            batch.process_status = "error"
            batch.error_reason = str(exc)

        return UnifiedViewModel(model=model, validation=validation, single=None, batch=batch)

    def build_validation_only(self) -> UnifiedViewModel:
        return UnifiedViewModel(
            model=self._build_model_version(),
            validation=self._build_validation(),
            single=None,
            batch=None,
        )

    # ------------------------------------------------------------------
    # 给页面消费的扁平字典 / DataFrame
    # ------------------------------------------------------------------
    @staticmethod
    def flatten_for_display(vm: UnifiedViewModel) -> Dict[str, Any]:
        m = vm.model
        v = vm.validation
        out: Dict[str, Any] = {
            "model_version": m.model_version,
            "model_name": m.best_model_name,
            "training_time": m.training_time,
            "selection_metric": m.selection_metric,
            "best_metric_value": m.best_metric_value,
            "feature_contract_version": m.feature_contract_version,
            "validation_valid": v.is_valid,
            "validation_errors": list(v.errors),
            "validation_warnings": list(v.warnings),
            "validation_details": dict(v.details),
            "has_alerts": bool(v.errors or v.warnings),
        }
        if vm.single is not None:
            s = vm.single
            out.update({
                "is_error": s.process_status == "error",
                "error_reason": s.error_reason,
                "prediction_code": s.prediction,
                "prediction_label": s.class_label,
                "fraud_probability": s.fraud_probability,
                "legit_probability": s.legit_probability,
                "risk_level": s.risk.risk_level,
                "risk_summary": s.risk.summary_text,
                "risk_contributors": list(s.risk.top_contributors),
                "transaction_id": s.transaction_id,
                "process_status": s.process_status,
            })
        if vm.batch is not None:
            b = vm.batch
            out.update({
                "batch_total": b.total,
                "batch_fraud": b.fraud,
                "batch_legit": b.legit,
                "batch_fraud_rate": b.fraud_rate,
                "batch_is_error": b.process_status == "error",
                "batch_error_reason": b.error_reason,
            })
        return out

    @staticmethod
    def batch_to_dataframe(vm: UnifiedViewModel) -> pd.DataFrame:
        if vm.batch is None:
            return pd.DataFrame()
        records = []
        for r in vm.batch.rows:
            records.append({
                "transaction_id": r.transaction_id,
                "prediction": r.prediction,
                "class_label": r.class_label,
                "fraud_probability": r.fraud_probability,
                "legit_probability": r.legit_probability,
                "process_status": r.process_status,
                "error_reason": r.error_reason,
                "model_version": r.model_version,
                "risk_level": r.risk.risk_level,
                "risk_summary": r.risk.summary_text,
                "top_contributors": r.risk.top_contributors,
            })
        return pd.DataFrame(records)

    @staticmethod
    def to_dict(vm: UnifiedViewModel) -> Dict[str, Any]:
        return asdict(vm)
