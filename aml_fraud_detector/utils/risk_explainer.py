import sys
import json
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


FEATURE_LABELS: Dict[str, str] = {
    "amount_received": "交易金额",
    "account": "发起账户",
    "account_1": "接收账户",
    "payment_format": "支付方式",
    "day": "交易星期",
    "from_bank": "发起银行",
    "to_bank": "接收银行",
    "receiving_currency": "接收币种",
    "payment_currency": "支付币种",
}

FACTOR_TEMPLATES: Dict[str, Dict[str, str]] = {
    "amount_received": {
        "high_risk": "交易金额为 {value}，金额偏高，大额交易具有更高的洗钱风险",
        "low_risk": "交易金额为 {value}，金额处于正常范围",
    },
    "payment_format": {
        "high_risk": "支付方式为 {value}，此类支付方式可能具有较高匿名性，增加洗钱风险",
        "low_risk": "支付方式为 {value}，此类支付方式风险相对较低",
    },
    "account": {
        "high_risk": "发起账户 {value} 的交易模式异常，需关注",
        "low_risk": "发起账户 {value} 的交易模式正常",
    },
    "account_1": {
        "high_risk": "接收账户 {value} 的交易模式异常，需关注",
        "low_risk": "接收账户 {value} 的交易模式正常",
    },
    "day": {
        "high_risk": "交易发生在{value}，异常时段交易需警惕",
        "low_risk": "交易发生在{value}，属正常交易日",
    },
    "from_bank": {
        "high_risk": "发起银行 {value} 相关交易风险较高",
        "low_risk": "发起银行 {value} 相关交易风险较低",
    },
    "to_bank": {
        "high_risk": "接收银行 {value} 相关交易风险较高",
        "low_risk": "接收银行 {value} 相关交易风险较低",
    },
    "receiving_currency": {
        "high_risk": "接收币种为 {value}，该币种相关交易需关注",
        "low_risk": "接收币种为 {value}，该币种风险较低",
    },
    "payment_currency": {
        "high_risk": "支付币种为 {value}，该币种相关交易需关注",
        "low_risk": "支付币种为 {value}，该币种风险较低",
    },
}

DEFAULT_LABEL = "未知字段"
DEFAULT_HIGH = "{label} 取值 {value}，增加了风险",
DEFAULT_LOW = "{label} 取值 {value}，风险较低",

INFERENCE_CONTRACT_VERSION = "1.0"


def _format_value(val: Any) -> str:
    if isinstance(val, float):
        if abs(val) >= 1:
            return f"{val:,.2f}"
        return f"{val:.4f}"
    return str(val)


def _describe_factor(feature: str, value: Any, direction: str) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    str_val = _format_value(value)
    templates = FACTOR_TEMPLATES.get(feature)
    if templates:
        return templates[direction].format(value=str_val, label=label)
    tpl = DEFAULT_HIGH[0] if direction == "high_risk" else DEFAULT_LOW[0]
    return tpl.format(value=str_val, label=label)


def generate_training_signature(
    feature_columns: List[str],
    numerical_features: List[str],
    categorical_features: List[str],
    baseline_values: Dict[str, Any],
) -> str:
    sig_components: List[str] = []
    sig_components.append("v1")
    sig_components.append("|".join(sorted(feature_columns)))
    sig_components.append("|".join(sorted(numerical_features)))
    sig_components.append("|".join(sorted(categorical_features)))
    for key in sorted(baseline_values.keys()):
        sig_components.append(f"{key}={baseline_values[key]}")
    raw = "||".join(sig_components).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def validate_training_signature(
    metadata_signature: str,
    summary_signature: str,
) -> Tuple[bool, str]:
    if not metadata_signature or not summary_signature:
        return False, "训练签名缺失，无法验证产物一致性"
    if metadata_signature != summary_signature:
        return (
            False,
            f"训练签名不一致：metadata={metadata_signature[:8]}...  summary={summary_signature[:8]}...",
        )
    return True, "训练签名验证通过"


def extract_feature_metadata(
    preprocessor,
    numerical_features: List[str],
    categorical_features: List[str],
    train_df: pd.DataFrame,
) -> Dict[str, Any]:
    try:
        metadata: Dict[str, Any] = {
            "contract_version": INFERENCE_CONTRACT_VERSION,
            "generated_at": datetime.now().isoformat(),
            "original_features": list(numerical_features + categorical_features),
            "numerical_features": list(numerical_features),
            "categorical_features": list(categorical_features),
            "encoding_info": {},
            "feature_labels": {},
            "baseline_values": {},
            "training_stats": {},
        }

        for feat in numerical_features:
            metadata["encoding_info"][feat] = {"type": "numerical"}
            if feat in train_df.columns:
                series = pd.to_numeric(train_df[feat], errors="coerce")
                metadata["baseline_values"][feat] = float(series.median())
                metadata["training_stats"][feat] = {
                    "median": float(series.median()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                    "q25": float(series.quantile(0.25)),
                    "q75": float(series.quantile(0.75)),
                }
            metadata["feature_labels"][feat] = FEATURE_LABELS.get(feat, feat)

        cat_transformer = None
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "transformers_"):
                cat_transformer = transformer
                break

        if cat_transformer is not None:
            for inner_name, inner_tf, inner_cols in cat_transformer.transformers_:
                if inner_name == "remainder":
                    continue
                if isinstance(inner_tf, OneHotEncoder):
                    for feat, cats in zip(inner_cols, inner_tf.categories_):
                        metadata["encoding_info"][feat] = {
                            "type": "onehot",
                            "categories": [str(c) for c in cats],
                        }
                elif inner_tf.__class__.__name__ == "CountEncoder":
                    for feat in inner_cols:
                        metadata["encoding_info"][feat] = {"type": "frequency"}

        for feat in categorical_features:
            metadata["feature_labels"][feat] = FEATURE_LABELS.get(feat, feat)
            if feat in train_df.columns:
                mode_val = train_df[feat].mode(dropna=True)
                baseline = str(mode_val.iloc[0]) if len(mode_val) > 0 else ""
                metadata["baseline_values"][feat] = baseline
                vc = train_df[feat].value_counts()
                metadata["training_stats"][feat] = {
                    "mode": baseline,
                    "n_unique": int(vc.shape[0]),
                    "top5": {str(k): int(v) for k, v in vc.head(5).items()},
                }

        metadata["training_signature"] = generate_training_signature(
            feature_columns=metadata["original_features"],
            numerical_features=metadata["numerical_features"],
            categorical_features=metadata["categorical_features"],
            baseline_values=metadata["baseline_values"],
        )
        logging.info(
            f"Generated training signature: {metadata['training_signature']}"
        )

        return metadata
    except Exception as e:
        raise CustomerException(e, sys)


def save_feature_metadata(metadata: Dict[str, Any], file_path: str) -> str:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        logging.info(f"Feature metadata saved to: {file_path}")
        return os.path.abspath(file_path)
    except Exception as e:
        raise CustomerException(e, sys)


def load_feature_metadata(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise CustomerException(e, sys)


def load_training_summary(file_path: str) -> Optional[Dict[str, Any]]:
    try:
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _explain_single_row(
    row_df: pd.DataFrame,
    model,
    preprocessor,
    feature_metadata: Dict[str, Any],
    top_n: int,
    preprocessed_X: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if preprocessed_X is None:
        X = preprocessor.transform(row_df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        else:
            X = np.asarray(X)
    else:
        X = preprocessed_X

    base_proba = float(model.predict_proba(X)[0, 1])
    prediction = int(model.predict(X)[0])

    original_features = feature_metadata.get("original_features", [])
    baseline_values = feature_metadata.get("baseline_values", {})

    contributions: List[Dict[str, Any]] = []
    for feat in original_features:
        if feat not in baseline_values:
            continue
        modified_df = row_df.copy()
        modified_df[feat] = baseline_values[feat]
        try:
            X_mod = preprocessor.transform(modified_df)
            if hasattr(X_mod, "toarray"):
                X_mod = X_mod.toarray()
            else:
                X_mod = np.asarray(X_mod)
            mod_proba = float(model.predict_proba(X_mod)[0, 1])
        except Exception:
            mod_proba = base_proba

        contribution = base_proba - mod_proba
        raw_value = row_df[feat].iloc[0]
        direction = "high_risk" if contribution > 0 else "low_risk"

        contributions.append({
            "feature": feat,
            "label": feature_metadata.get("feature_labels", {}).get(feat, feat),
            "value": _format_value(raw_value),
            "contribution": round(contribution, 6),
            "direction": direction,
            "description": _describe_factor(feat, raw_value, direction),
        })

    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    total_abs = sum(abs(c["contribution"]) for c in contributions)
    for c in contributions:
        if total_abs > 0:
            c["contribution_pct"] = round(abs(c["contribution"]) / total_abs * 100, 1)
        else:
            c["contribution_pct"] = 0.0

    return {
        "prediction": prediction,
        "prediction_label": "欺诈交易" if prediction == 1 else "正常交易",
        "fraud_probability": round(base_proba, 4),
        "risk_explanation": {
            "top_factors": contributions[:top_n],
        },
    }


def explain_risk(
    features_df: pd.DataFrame,
    model,
    preprocessor,
    feature_metadata: Dict[str, Any],
    top_n: int = 5,
) -> Dict[str, Any]:
    try:
        X_full = preprocessor.transform(features_df)
        if hasattr(X_full, "toarray"):
            X_full = X_full.toarray()
        else:
            X_full = np.asarray(X_full)

        n_rows = features_df.shape[0]
        base_probas = model.predict_proba(X_full)[:, 1]
        predictions = model.predict(X_full)

        original_features = feature_metadata.get("original_features", [])
        baseline_values = feature_metadata.get("baseline_values", {})

        rows: List[Dict[str, Any]] = []
        fraud_count = 0
        normal_count = 0

        for i in range(n_rows):
            row_df = features_df.iloc[[i]].copy()
            base_proba = float(base_probas[i])
            prediction = int(predictions[i])

            if prediction == 1:
                fraud_count += 1
            else:
                normal_count += 1

            try:
                contributions: List[Dict[str, Any]] = []
                for feat in original_features:
                    if feat not in baseline_values:
                        continue
                    modified_df = row_df.copy()
                    modified_df[feat] = baseline_values[feat]
                    try:
                        X_mod = preprocessor.transform(modified_df)
                        if hasattr(X_mod, "toarray"):
                            X_mod = X_mod.toarray()
                        else:
                            X_mod = np.asarray(X_mod)
                        mod_proba = float(model.predict_proba(X_mod)[0, 1])
                    except Exception:
                        mod_proba = base_proba

                    contribution = base_proba - mod_proba
                    raw_value = row_df[feat].iloc[0]
                    direction = "high_risk" if contribution > 0 else "low_risk"

                    contributions.append({
                        "feature": feat,
                        "label": feature_metadata.get("feature_labels", {}).get(feat, feat),
                        "value": _format_value(raw_value),
                        "contribution": round(contribution, 6),
                        "direction": direction,
                        "description": _describe_factor(feat, raw_value, direction),
                    })

                contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
                total_abs = sum(abs(c["contribution"]) for c in contributions)
                for c in contributions:
                    if total_abs > 0:
                        c["contribution_pct"] = round(abs(c["contribution"]) / total_abs * 100, 1)
                    else:
                        c["contribution_pct"] = 0.0

                row_dict: Dict[str, Any] = {
                    "row_index": i,
                    "process_status": "success",
                    "prediction": prediction,
                    "prediction_label": "欺诈交易" if prediction == 1 else "正常交易",
                    "fraud_probability": round(base_proba, 4),
                    "error_reason": None,
                    "risk_explanation": {
                        "top_factors": contributions[:top_n],
                    },
                }
                for col in features_df.columns:
                    row_dict[col] = features_df[col].iloc[i]

                rows.append(row_dict)

            except Exception as row_err:
                row_dict = {
                    "row_index": i,
                    "process_status": "failed",
                    "prediction": None,
                    "prediction_label": None,
                    "fraud_probability": None,
                    "error_reason": f"解释生成失败: {str(row_err)}",
                    "risk_explanation": {
                        "top_factors": [],
                    },
                }
                for col in features_df.columns:
                    row_dict[col] = features_df[col].iloc[i]
                rows.append(row_dict)

        output: Dict[str, Any] = {
            "contract_version": feature_metadata.get("contract_version", "1.0"),
            "training_signature": feature_metadata.get("training_signature", ""),
            "is_batch": n_rows > 1,
            "count": n_rows,
            "fraud_count": fraud_count,
            "normal_count": normal_count,
            "fraud_rate": round(fraud_count / n_rows * 100, 2) if n_rows > 0 else 0.0,
            "rows": rows,
        }

        if n_rows == 1:
            output["row"] = rows[0]

        return output
    except Exception as e:
        raise CustomerException(e, sys)
