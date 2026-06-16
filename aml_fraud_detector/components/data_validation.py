import os
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from aml_fraud_detector.exception import (
    AMLException,
    DataQualityException,
    wrap_exception,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.logger import logging


DEFAULT_CRITICAL_FEATURE_COLUMNS = [
    "account",
    "account_1",
    "amount_received",
    "payment_format",
]

DEFAULT_CATEGORICAL_COLUMNS = [
    "account",
    "account_1",
    "payment_format",
    "day",
]

DEFAULT_AMOUNT_COLUMNS = [
    "amount_received",
    "amount_paid",
]

DEFAULT_TIMESTAMP_COLUMNS = [
    "timestamp",
]

MISSING_CRITICAL_RATIO = 0.3
MISSING_WARNING_RATIO = 0.1
DUP_WARNING_RATIO = 0.1


@dataclass
class DataValidationArtifact:
    report_path: str
    is_valid: bool = True
    issues: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DataValidationResult:
    is_valid: bool = True
    issues: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_values_report: List[Dict[str, Any]] = field(default_factory=list)
    duplicates_report: Dict[str, Any] = field(default_factory=dict)
    abnormal_amounts: List[Dict[str, Any]] = field(default_factory=list)
    unknown_categories: List[Dict[str, Any]] = field(default_factory=list)
    target_distribution: Optional[Dict[str, Any]] = None
    error_code: Optional[ErrorCode] = None
    error_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.error_code:
            result["error_code"] = self.error_code.value
        return result

    def first_error_code(self) -> ErrorCode:
        if self.error_code:
            return self.error_code
        if self.issues:
            top = self.issues[0]
            issue_type = top.get("type", "")
            if issue_type == "missing_values":
                return ErrorCode.DATA_MISSING_COLUMNS
            elif issue_type == "invalid_numeric":
                return ErrorCode.DATA_INVALID_DTYPE
            elif issue_type == "unknown_category":
                return ErrorCode.DATA_INVALID_DTYPE
            elif issue_type == "target_distribution":
                return ErrorCode.DATA_CORRUPTED
            severity = top.get("severity", "warning")
            if severity == "critical":
                return ErrorCode.DATA_MISSING_COLUMNS
            return ErrorCode.DATA_QUALITY if hasattr(ErrorCode, "DATA_QUALITY") else ErrorCode.DATA_CORRUPTED
        return ErrorCode.INTERNAL_UNEXPECTED

    def first_error_message(self) -> str:
        if self.error_reason:
            return self.error_reason
        if self.issues:
            top = self.issues[0]
            return f"{top.get('column', 'data')}: {top.get('message', '数据校验失败')}"
        return "数据校验失败"


class DataValidation:
    def __init__(
        self,
        critical_feature_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        categorical_whitelist: Optional[Dict[str, List[str]]] = None,
        amount_columns: Optional[List[str]] = None,
        timestamp_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ):
        self.critical_feature_columns = critical_feature_columns or DEFAULT_CRITICAL_FEATURE_COLUMNS
        self.categorical_columns = categorical_columns or DEFAULT_CATEGORICAL_COLUMNS
        self.categorical_whitelist = categorical_whitelist or {}
        self.amount_columns = amount_columns or DEFAULT_AMOUNT_COLUMNS
        self.timestamp_columns = timestamp_columns or DEFAULT_TIMESTAMP_COLUMNS
        self.target_column = target_column
        self.thresholds = thresholds or {
            "missing_value_critical_ratio": MISSING_CRITICAL_RATIO,
            "missing_value_warning_ratio": MISSING_WARNING_RATIO,
            "unknown_category_critical_ratio": 0.05,
            "unknown_category_warning_ratio": 0.01,
            "duplicate_row_warning_ratio": DUP_WARNING_RATIO,
            "target_missing_critical_ratio": 0.0,
            "target_expected_classes": 2,
            "target_imbalance_critical_ratio": 0.01,
            "target_imbalance_warning_ratio": 0.05,
        }

    def validate(self, df: pd.DataFrame) -> DataValidationResult:
        result = DataValidationResult()

        if df is None or len(df) == 0:
            result.is_valid = False
            result.error_code = ErrorCode.DATA_EMPTY
            result.error_reason = f"数据集为空，共 {len(df) if df is not None else 0} 行"
            return result

        self._check_missing_values(df, result)
        self._check_duplicates(df, result)
        self._check_abnormal_amounts(df, result)
        self._check_unknown_categories(df, result)
        self._check_target_distribution(df, result)

        result.is_valid = len(result.issues) == 0
        if not result.is_valid:
            result.error_code = result.first_error_code()
            result.error_reason = result.first_error_message()
            logging.warning(
                f"Data validation failed: {len(result.issues)} critical issue(s), "
                f"first: {result.error_reason}"
            )
        else:
            logging.info(
                f"Data validation passed: {len(df)} rows, "
                f"{len(result.warnings)} warning(s)"
            )
        return result

    def _check_missing_columns(self, df: pd.DataFrame, result: DataValidationResult) -> None:
        actual = set(df.columns.tolist())
        missing_critical = [c for c in self.critical_feature_columns if c not in actual]
        if missing_critical:
            issue = {
                "type": "missing_critical_columns",
                "column": ",".join(missing_critical),
                "message": f"缺少必需列: {', '.join(missing_critical)}",
                "severity": "critical",
            }
            result.issues.append(issue)
            result.error_code = ErrorCode.DATA_MISSING_COLUMNS
            result.error_reason = issue["message"]

    def _check_missing_values(self, df: pd.DataFrame, result: DataValidationResult) -> None:
        cr = self.thresholds.get("missing_value_critical_ratio", MISSING_CRITICAL_RATIO)
        wr = self.thresholds.get("missing_value_warning_ratio", MISSING_WARNING_RATIO)
        for col in df.columns:
            total = len(df)
            missing_count = int(df[col].isnull().sum())
            ratio = missing_count / total if total else 0.0
            entry = {
                "column": col,
                "missing_count": missing_count,
                "missing_ratio": ratio,
                "severity": "none",
                "threshold_critical": cr,
                "threshold_warning": wr,
            }
            if col in self.critical_feature_columns and ratio > cr:
                entry["severity"] = "critical"
                result.issues.append({
                    "type": "missing_values",
                    "column": col,
                    "message": f"列 {col} 缺失率 {ratio:.2%} 超过阈值 {cr:.0%}",
                    "severity": "critical",
                })
            elif ratio > wr:
                entry["severity"] = "warning"
                result.warnings.append(f"列 {col} 缺失率 {ratio:.2%} 超过警告阈值 {wr:.0%}")
            result.missing_values_report.append(entry)

    def _check_duplicates(self, df: pd.DataFrame, result: DataValidationResult) -> None:
        total = len(df)
        dup_count = int(df.duplicated().sum())
        ratio = dup_count / total if total else 0.0
        dr = self.thresholds.get("duplicate_row_warning_ratio", DUP_WARNING_RATIO)
        entry = {
            "duplicate_row_count": dup_count,
            "duplicate_row_ratio": ratio,
            "severity": "none",
            "threshold_warning": dr,
        }
        if ratio > dr:
            entry["severity"] = "warning"
            result.warnings.append(f"重复行比例 {ratio:.2%} 超过警告阈值 {dr:.0%}")
        result.duplicates_report = entry

    def _check_abnormal_amounts(self, df: pd.DataFrame, result: DataValidationResult) -> None:
        for col in self.amount_columns:
            if col not in df.columns:
                continue
            original_series = df[col]
            numeric_series = pd.to_numeric(original_series, errors="coerce")
            invalid_count = int(numeric_series.isna().sum() - original_series.isna().sum())
            invalid_ratio = invalid_count / len(original_series) if len(original_series) else 0.0
            neg_count = int((numeric_series < 0).sum())
            neg_ratio = neg_count / len(numeric_series) if len(numeric_series) else 0.0
            entry = {
                "column": col,
                "invalid_count": invalid_count,
                "invalid_ratio": invalid_ratio,
                "negative_count": neg_count,
                "negative_ratio": neg_ratio,
                "severity": "none",
            }
            if invalid_count > 0:
                entry["severity"] = "critical"
                result.issues.append({
                    "type": "invalid_numeric",
                    "column": col,
                    "message": f"列 {col} 存在 {invalid_count} 个非数值数据，无法转换为数字",
                    "severity": "critical",
                })
            elif neg_count > 0:
                entry["severity"] = "warning"
                result.warnings.append(f"列 {col} 存在 {neg_count} 个负值")
            result.abnormal_amounts.append(entry)

    def _check_unknown_categories(self, df: pd.DataFrame, result: DataValidationResult) -> None:
        cr = self.thresholds.get("unknown_category_critical_ratio", 0.05)
        wr = self.thresholds.get("unknown_category_warning_ratio", 0.01)
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            if col not in self.categorical_whitelist:
                continue
            whitelist = set(self.categorical_whitelist[col])
            actual = df[col].fillna("__MISSING__").astype(str).tolist()
            total = len(actual)
            unknown_count = sum(1 for v in actual if v not in whitelist)
            ratio = unknown_count / total if total else 0.0
            entry = {
                "column": col,
                "unknown_count": unknown_count,
                "unknown_ratio": ratio,
                "severity": "none",
            }
            if ratio > cr:
                entry["severity"] = "critical"
                result.issues.append({
                    "type": "unknown_category",
                    "column": col,
                    "message": f"列 {col} 未知类别比例 {ratio:.2%} 超过阈值 {cr:.0%}",
                    "severity": "critical",
                })
            elif ratio > wr:
                entry["severity"] = "warning"
                result.warnings.append(
                    f"列 {col} 未知类别比例 {ratio:.2%} 超过警告阈值 {wr:.0%}"
                )
            result.unknown_categories.append(entry)

    def _check_target_distribution(self, df: pd.DataFrame, result: DataValidationResult) -> None:
        if not self.target_column or self.target_column not in df.columns:
            return
        target = df[self.target_column].dropna()
        if target.empty:
            result.issues.append({
                "type": "target_missing",
                "column": self.target_column,
                "message": f"目标列 {self.target_column} 全部为空",
                "severity": "critical",
            })
            return
        counts = target.value_counts().to_dict()
        counts = {str(k): int(v) for k, v in counts.items()}
        total = len(target)
        expected_classes = self.thresholds.get("target_expected_classes", 2)
        if len(counts) != expected_classes:
            result.warnings.append(
                f"目标列类别数 {len(counts)} 与期望 {expected_classes} 不一致"
            )
        min_class = min(counts.values()) if counts else 0
        imbalance = min_class / total if total else 0.0
        cr = self.thresholds.get("target_imbalance_critical_ratio", 0.01)
        wr = self.thresholds.get("target_imbalance_warning_ratio", 0.05)
        if imbalance < cr:
            result.warnings.append(f"类别极度不平衡: 最小类占比 {imbalance:.2%}")
        elif imbalance < wr:
            result.warnings.append(f"类别不平衡: 最小类占比 {imbalance:.2%}")
        result.target_distribution = counts


def save_data_validation_report(
    report_path: str,
    validation_result: DataValidationResult,
    df: pd.DataFrame,
) -> str:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = {
        "dataset_shape": list(df.shape) if df is not None else [0, 0],
        "missing_values": validation_result.missing_values_report,
        "critical_missing_columns": [
            i["column"] for i in validation_result.issues
            if i.get("type") == "missing_critical_columns"
        ],
        "duplicates": validation_result.duplicates_report,
        "abnormal_amounts": validation_result.abnormal_amounts,
        "unknown_categories": validation_result.unknown_categories,
        "time_parsing": [],
        "target_distribution": validation_result.target_distribution,
        "risk_items": validation_result.issues,
        "warnings": validation_result.warnings,
        "generated_at": _now_iso(),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logging.info(f"Data validation report written to {report_path}")
    return report_path


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat()
