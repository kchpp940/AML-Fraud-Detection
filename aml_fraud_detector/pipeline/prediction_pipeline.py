import sys
import os
import json
import uuid
import hashlib
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.entity import (
    ProcessStatus,
    RiskLevel,
    REQUIRED_INPUT_FIELDS,
    FOUR_ARTIFACT_FILENAMES,
    RiskExplanation,
    ModelVersionInfo,
    ValidationStatus,
    PredictionResult,
    BatchPredictionResult,
)


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


def _determine_risk_level(p: float) -> RiskLevel:
    if p >= 0.8:
        return RiskLevel.CRITICAL
    if p >= 0.6:
        return RiskLevel.HIGH
    if p >= 0.3:
        return RiskLevel.MEDIUM
    if p >= 0.01:
        return RiskLevel.LOW
    return RiskLevel.MINIMAL


def _build_risk_summary(p: float, level: RiskLevel) -> str:
    pct = f"{p * 100:.1f}%"
    label = level.value
    if level == RiskLevel.MINIMAL:
        return f"该笔交易欺诈风险极低（欺诈概率 {pct}），风险等级：{label}。当前特征未见明显异常信号，可按正常流程处理。"
    if level == RiskLevel.LOW:
        return f"该笔交易欺诈风险较低（欺诈概率 {pct}），风险等级：{label}。建议结合业务规则做常规复核。"
    if level == RiskLevel.MEDIUM:
        return f"该笔交易存在中等欺诈风险（欺诈概率 {pct}），风险等级：{label}。建议人工复核账户历史与交易背景。"
    if level == RiskLevel.HIGH:
        return f"该笔交易欺诈风险较高（欺诈概率 {pct}），风险等级：{label}。建议优先进入人工审核队列，并对账户进行临时限制。"
    return f"该笔交易存在严重欺诈嫌疑（欺诈概率 {pct}），风险等级：{label}。建议立即冻结交易并启动调查流程。"


class RiskExplainer:
    """基于欺诈概率与输入特征的边际贡献解释器。"""

    def __init__(self, feature_metadata: Optional[Dict[str, Any]] = None):
        self._feature_metadata = feature_metadata or {}

    def explain(
        self,
        fraud_probability: float,
        input_row: Optional[pd.Series] = None,
    ) -> RiskExplanation:
        level = _determine_risk_level(fraud_probability)
        summary = _build_risk_summary(fraud_probability, level)
        contributors: List[Dict[str, Any]] = []

        if input_row is not None and fraud_probability >= 0.01:
            for col in ["amount_received", "from_bank", "to_bank"]:
                if col in input_row.index:
                    try:
                        val = float(input_row[col])
                    except (TypeError, ValueError):
                        continue
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
            contributors.sort(key=lambda x: x["contribution"], reverse=True)
            contributors = contributors[:5]

        return RiskExplanation(
            fraud_probability=float(fraud_probability),
            risk_level=level,
            summary_text=summary,
            top_contributors=contributors,
        )


class ArtifactValidator:
    """基于 manifest + SHA-256 digest 的 artifact 完整性校验器。"""

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir

    def validate(self) -> ValidationStatus:
        errors: List[str] = []
        warnings: List[str] = []

        manifest_path = os.path.join(self.artifacts_dir, "artifact_manifest.json")
        manifest: Dict[str, Any] = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        mm = self._load_json("model_metadata.json")
        fm = self._load_json("feature_metadata.json")

        details: Dict[str, Any] = {
            "artifacts_dir": os.path.abspath(self.artifacts_dir),
            "model_version": int(mm.get("model_version", 0)) if mm else 0,
            "best_model_name": str(mm.get("best_model_name", "")) if mm else "",
            "feature_contract_version": str(fm.get("contract_version", "")) if fm else "",
            "numerical_features": list(fm.get("numerical_features", [])) if fm else [],
            "categorical_features": list(fm.get("categorical_features", [])) if fm else [],
            "manifest_artifacts": list(manifest.get("artifacts", {}).keys()) if manifest else [],
            "manifest_present": bool(manifest),
        }

        for fname in FOUR_ARTIFACT_FILENAMES:
            fpath = os.path.join(self.artifacts_dir, fname)
            key = ARTIFACT_TO_DIGEST_KEY[fname]
            if not os.path.exists(fpath):
                errors.append(f"Missing artifact: {fname}")
                details[key] = ""
                continue

            try:
                actual_digest = _sha256_file(fpath)
            except OSError as exc:
                errors.append(f"{fname} digest computation failed: {exc}")
                details[key] = ""
                continue

            details[key] = actual_digest

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

        is_valid = not errors
        logging.info(
            f"Artifact validation: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}"
        )
        return ValidationStatus(is_valid=is_valid, errors=errors, warnings=warnings, details=details)

    def _load_json(self, name: str) -> Dict[str, Any]:
        path = os.path.join(self.artifacts_dir, name)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class PredictionPipeline:
    def __init__(self, artifacts_dir: str = "artifacts", validate: bool = True):
        self.artifacts_dir = artifacts_dir
        self._model = None
        self._preprocessor = None
        self._model_metadata: Optional[Dict[str, Any]] = None
        self._feature_metadata: Optional[Dict[str, Any]] = None
        self._explainer: Optional[RiskExplainer] = None
        self._validator: Optional[ArtifactValidator] = None
        self._loaded = False

        if validate:
            self.validate_artifacts()
            logging.info(
                f"PredictionPipeline created (artifacts_dir={artifacts_dir}, validate={validate})"
            )

    # ---------- artifact loading ----------
    def _ensure_loaded(self):
        if self._loaded:
            return
        model_path = os.path.join(self.artifacts_dir, "model.pkl")
        preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")
        self._model = load_object(file_path=model_path)
        self._preprocessor = load_object(file_path=preprocessor_path)

        mm_path = os.path.join(self.artifacts_dir, "model_metadata.json")
        if os.path.exists(mm_path):
            with open(mm_path, "r", encoding="utf-8") as f:
                self._model_metadata = json.load(f)

        fm_path = os.path.join(self.artifacts_dir, "feature_metadata.json")
        if os.path.exists(fm_path):
            with open(fm_path, "r", encoding="utf-8") as f:
                self._feature_metadata = json.load(f)

        self._explainer = RiskExplainer(self._feature_metadata)
        self._validator = ArtifactValidator(self.artifacts_dir)
        self._loaded = True

    # ---------- validation ----------
    def validate_artifacts(self) -> ValidationStatus:
        self._ensure_loaded()
        return self._validator.validate()

    def get_model_version_info(self) -> ModelVersionInfo:
        self._ensure_loaded()
        mm = self._model_metadata or {}
        return ModelVersionInfo(
            model_version=int(mm.get("model_version", 0)),
            best_model_name=str(mm.get("best_model_name", "")),
            training_time=str(mm.get("training_time", "")),
            selection_metric=str(mm.get("selection_metric", "")),
            best_metric_value=float(mm.get("best_metric_value", 0.0)),
            feature_contract_version=str(mm.get("feature_schema_version", "")),
            artifact_path=str(mm.get("artifact_path", "")),
        )

    # ---------- core inference (legacy API) ----------
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        根据 feature_metadata.json 把输入 DataFrame 的分类列转成 object str，
        避免 preprocessor 的 categorical encoder 因 dtype 不匹配（如 int64→isnan）报错。
        注意：本方法不改变语义，只保证 pipeline 的鲁棒性。
        """
        fm = self._feature_metadata or {}
        categorical_cols = fm.get("categorical_features", [])
        if not categorical_cols:
            return features
        out = features.copy()
        for col in categorical_cols:
            if col in out.columns:
                out[col] = out[col].astype(object).where(out[col].notna(), None).astype(str)
        return out

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        try:
            self._ensure_loaded()
            normed = self._normalize_features(features)
            data_scaled = self._preprocessor.transform(normed)
            predictions = self._model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomerException(e, sys)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        try:
            self._ensure_loaded()
            normed = self._normalize_features(features)
            data_scaled = self._preprocessor.transform(normed)
            predictions_prob = self._model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            raise CustomerException(e, sys)

    # ---------- helper ----------
    def _build_prediction_result(
        self,
        row_idx: int,
        pred_class: int,
        probs: np.ndarray,
        model_version: int,
        input_row: Optional[pd.Series] = None,
        transaction_id: Optional[str] = None,
    ) -> PredictionResult:
        legit_prob = float(probs[0]) if len(probs) > 0 else 0.0
        fraud_prob = float(probs[1]) if len(probs) > 1 else 0.0
        class_label = "Fraud" if pred_class == 1 else "Not Fraud"
        explanation = self._explainer.explain(fraud_prob, input_row=input_row)
        return PredictionResult(
            prediction=int(pred_class),
            class_label=class_label,
            fraud_probability=fraud_prob,
            legit_probability=legit_prob,
            model_version=model_version,
            transaction_id=transaction_id,
            process_status=ProcessStatus.SUCCESS,
            error_reason=None,
            risk_explanation=explanation,
        )

    def _validate_required_fields(self, data: Dict[str, Any]) -> Optional[str]:
        for field in REQUIRED_INPUT_FIELDS:
            if field not in data:
                return f"Input dict missing required field: '{field}'"
        return None

    # ---------- single prediction ----------
    def predict_single(self, data: Dict[str, Any]) -> PredictionResult:
        self._ensure_loaded()

        field_err = self._validate_required_fields(data)
        if field_err is not None:
            return PredictionResult(
                prediction=-1,
                process_status=ProcessStatus.ERROR,
                error_reason=field_err,
            )

        try:
            df = pd.DataFrame([data])
            preds = self.predict(df)
            probas = self.predict_proba(df)
            mv = self.get_model_version_info().model_version
            tx_id = data.get("transaction_id") or str(uuid.uuid4())
            return self._build_prediction_result(
                0, int(preds[0]), probas[0], mv,
                input_row=df.iloc[0], transaction_id=tx_id,
            )
        except Exception as e:
            logging.exception("predict_single failed")
            return PredictionResult(
                prediction=-1,
                process_status=ProcessStatus.ERROR,
                error_reason=str(e),
            )

    # ---------- batch prediction ----------
    def predict_batch(self, df: pd.DataFrame) -> BatchPredictionResult:
        self._ensure_loaded()
        batch = BatchPredictionResult()

        try:
            preds = self.predict(df)
            probas = self.predict_proba(df)
            mv = self.get_model_version_info().model_version
            results: List[PredictionResult] = []

            for i in range(len(df)):
                tx_id = None
                if "transaction_id" in df.columns:
                    tx_id = str(df.iloc[i]["transaction_id"])
                pr = self._build_prediction_result(
                    i, int(preds[i]), probas[i], mv,
                    input_row=df.iloc[i], transaction_id=tx_id,
                )
                results.append(pr)

            batch.total_count = len(results)
            batch.fraud_count = sum(1 for r in results if r.prediction == 1)
            batch.legit_count = batch.total_count - batch.fraud_count
            batch.fraud_rate = (batch.fraud_count / batch.total_count) if batch.total_count > 0 else 0.0
            batch.predictions = results
            return batch
        except Exception as e:
            logging.exception("predict_batch failed")
            return BatchPredictionResult(
                process_status=ProcessStatus.ERROR,
                error_reason=str(e),
            )

    # ---------- flat output ----------
    @staticmethod
    def batch_to_dataframe(batch: BatchPredictionResult) -> pd.DataFrame:
        records = []
        for p in batch.predictions:
            records.append({
                "transaction_id": p.transaction_id,
                "prediction": p.prediction,
                "class_label": p.class_label,
                "fraud_probability": p.fraud_probability,
                "legit_probability": p.legit_probability,
                "process_status": p.process_status.value,
                "error_reason": p.error_reason,
                "model_version": p.model_version,
                "risk_level": p.risk_level.value,
                "risk_summary": p.risk_summary,
                "top_contributors": p.top_contributors,
            })
        return pd.DataFrame(records)


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

    def get_data_as_DataFrame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "from_bank": [self.from_bank],
                "account": [self.account],
                "to_bank": [self.to_bank],
                "account_1": [self.account_1],
                "amount_received": [self.amount_received],
                "receiving_currency": [self.receiving_currency],
                "payment_currency": [self.payment_currency],
                "payment_format": [self.payment_format],
                "day": [self.day]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomerException(e, sys)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_bank": self.from_bank,
            "account": self.account,
            "to_bank": self.to_bank,
            "account_1": self.account_1,
            "amount_received": self.amount_received,
            "receiving_currency": self.receiving_currency,
            "payment_currency": self.payment_currency,
            "payment_format": self.payment_format,
            "day": self.day,
        }
