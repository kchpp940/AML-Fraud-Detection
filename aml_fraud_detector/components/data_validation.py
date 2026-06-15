import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aml_fraud_detector.artifact_registry import ArtifactRegistry
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


@dataclass
class DataValidationConfig:
    critical_feature_columns: List[str] = field(default_factory=list)
    categorical_check_columns: List[str] = field(default_factory=list)
    categorical_whitelist: Dict[str, List[str]] = field(default_factory=dict)
    amount_columns: List[str] = field(default_factory=list)
    timestamp_columns: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataValidationArtifact:
    is_valid: bool
    report_path: str
    feature_metadata_path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


_DEFAULT_THRESHOLDS = {
    "missing_value_critical_ratio": 0.3,
    "missing_value_warning_ratio": 0.1,
    "unknown_category_critical_ratio": 0.05,
    "unknown_category_warning_ratio": 0.01,
    "duplicate_row_warning_ratio": 0.1,
    "target_missing_critical_ratio": 0.0,
    "target_expected_classes": 2,
    "target_imbalance_critical_ratio": 0.01,
    "target_imbalance_warning_ratio": 0.05,
}


class DataValidation:
    def __init__(
        self,
        registry: ArtifactRegistry,
        target_column: str = "is_laundering",
        config: Optional[DataValidationConfig] = None,
    ):
        self.registry = registry
        self.target_column = target_column
        self.config = config or DataValidationConfig(thresholds=dict(_DEFAULT_THRESHOLDS))
        for k, v in _DEFAULT_THRESHOLDS.items():
            self.config.thresholds.setdefault(k, v)
        logging.info(
            f"DataValidation initialized: target={self.target_column}, "
            f"report_out={self.registry.relative_path('data_quality_report_json')}, "
            f"critical_features={self.config.critical_feature_columns}"
        )

    def _check_missing_values(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        results = []
        crit = self.config.thresholds["missing_value_critical_ratio"]
        warn = self.config.thresholds["missing_value_warning_ratio"]
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            missing_ratio = float(missing_count / len(df)) if len(df) > 0 else 0.0
            severity = "NONE"
            if missing_ratio >= crit:
                severity = "CRITICAL"
            elif missing_ratio >= warn:
                severity = "WARNING"
            results.append({
                "column": col,
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio, 6),
                "severity": severity,
                "threshold_critical": crit,
                "threshold_warning": warn,
            })
        return results

    def _check_critical_missing(self, missing_results: List[Dict[str, Any]]) -> List[str]:
        return [
            r["column"] for r in missing_results
            if r["severity"] == "CRITICAL" and r["column"] in self.config.critical_feature_columns
        ]

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        dup_count = int(df.duplicated().sum())
        dup_ratio = float(dup_count / len(df)) if len(df) > 0 else 0.0
        warn = self.config.thresholds["duplicate_row_warning_ratio"]
        severity = "WARNING" if dup_ratio >= warn else "NONE"
        return {
            "duplicate_row_count": dup_count,
            "duplicate_row_ratio": round(dup_ratio, 6),
            "severity": severity,
            "threshold_warning": warn,
        }

    def _check_abnormal_amounts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        results = []
        for col in self.config.amount_columns:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            results.append({
                "column": col,
                "negative_count": int((series < 0).sum()),
                "zero_count": int((series == 0).sum()),
                "outlier_count": int(((series < lower) | (series > upper)).sum()),
                "outlier_lower": round(lower, 4),
                "outlier_upper": round(upper, 4),
            })
        return results

    def _check_unknown_categories(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        results = []
        for col in self.config.categorical_check_columns:
            if col not in df.columns or col not in self.config.categorical_whitelist:
                continue
            allowed = set(self.config.categorical_whitelist[col])
            actual = set(df[col].dropna().unique())
            unknown = actual - allowed
            if unknown:
                ratio = len(unknown) / len(actual) if len(actual) > 0 else 0.0
                results.append({
                    "column": col,
                    "unknown_categories": sorted(list(unknown)),
                    "unknown_ratio": round(ratio, 6),
                })
        return results

    def _check_time_parsing(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        results = []
        for col in self.config.timestamp_columns:
            if col not in df.columns:
                continue
            total = len(df)
            parsed = pd.to_datetime(df[col], errors="coerce")
            fail_count = int(parsed.isnull().sum() - df[col].isnull().sum())
            fail_count = max(fail_count, 0)
            results.append({
                "column": col,
                "total_count": total,
                "parse_fail_count": fail_count,
                "parse_fail_ratio": round(fail_count / total, 6) if total > 0 else 0.0,
            })
        return results

    def _check_target_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.target_column not in df.columns:
            return {}
        series = df[self.target_column]
        missing_count = int(series.isnull().sum())
        missing_ratio = float(missing_count / len(df)) if len(df) > 0 else 0.0
        unique_classes = int(series.nunique())
        expected = int(self.config.thresholds["target_expected_classes"])
        value_counts = series.value_counts().to_dict()
        value_counts = {str(k): int(v) for k, v in value_counts.items()}
        total = len(df)
        positive_ratio = 0.0
        imbalance_ratio = 0.0
        if 1 in value_counts or "1" in value_counts:
            pos_count = value_counts.get(1, value_counts.get("1", 0))
            positive_ratio = float(pos_count / total) if total > 0 else 0.0
            imbalance_ratio = positive_ratio

        issues = []
        imb_crit = self.config.thresholds["target_imbalance_critical_ratio"]
        imb_warn = self.config.thresholds["target_imbalance_warning_ratio"]
        overall = "NONE"
        if missing_ratio > 0:
            sev = "CRITICAL" if missing_ratio > self.config.thresholds["target_missing_critical_ratio"] else "WARNING"
            issues.append({
                "code": "target_missing_values",
                "message": f"Target column has {missing_count} missing values ({missing_ratio:.2%})",
                "severity": sev,
                "ratio": round(missing_ratio, 4),
            })
            overall = sev
        if unique_classes != expected:
            issues.append({
                "code": "target_class_mismatch",
                "message": f"Expected {expected} classes, found {unique_classes}",
                "severity": "WARNING",
            })
            if overall != "CRITICAL":
                overall = "WARNING"
        if imbalance_ratio > 0 and imbalance_ratio < imb_warn:
            if imbalance_ratio < imb_crit:
                issues.append({
                    "code": "target_severe_imbalance",
                    "message": f"Target column is severely imbalanced: minority ratio = {imbalance_ratio:.4%}, critical threshold = {imb_crit:.2%}",
                    "severity": "CRITICAL",
                    "ratio": round(imbalance_ratio, 4),
                    "threshold": imb_crit,
                })
                overall = "CRITICAL"
            else:
                issues.append({
                    "code": "target_mild_imbalance",
                    "message": f"Target column is mildly imbalanced: minority ratio = {imbalance_ratio:.4%}, warning threshold = {imb_warn:.2%}",
                    "severity": "WARNING",
                    "ratio": round(imbalance_ratio, 4),
                    "threshold": imb_warn,
                })
                if overall != "CRITICAL":
                    overall = "WARNING"

        return {
            "column": self.target_column,
            "missing_count": missing_count,
            "missing_ratio": round(missing_ratio, 6),
            "missing_threshold": self.config.thresholds["target_missing_critical_ratio"],
            "unique_classes": unique_classes,
            "expected_classes": expected,
            "value_counts": value_counts,
            "total_count": total,
            "positive_ratio": round(positive_ratio, 4),
            "imbalance_ratio": round(imbalance_ratio, 4),
            "imbalance_threshold_critical": imb_crit,
            "imbalance_threshold_warning": imb_warn,
            "issues": issues,
            "overall_severity": overall,
        }

    def _build_risk_items(self, missing_results, dup_result, target_dist) -> List[Dict[str, Any]]:
        items = []
        for r in missing_results:
            if r["severity"] in ("CRITICAL", "WARNING"):
                items.append({
                    "level": r["severity"],
                    "category": f"missing__{r['column']}",
                    "message": f"Column {r['column']} has {r['missing_count']} missing values ({r['missing_ratio']:.2%})",
                })
        if dup_result.get("severity") in ("CRITICAL", "WARNING"):
            items.append({
                "level": dup_result["severity"],
                "category": "duplicates",
                "message": f"Duplicate rows: {dup_result['duplicate_row_count']} ({dup_result['duplicate_row_ratio']:.2%})",
            })
        if target_dist and target_dist.get("issues"):
            for issue in target_dist["issues"]:
                items.append({
                    "level": issue["severity"],
                    "category": f"target__{issue['code']}",
                    "message": issue["message"],
                })
        return items

    def _build_feature_metadata(
        self,
        df: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str],
        encoding_info: Dict[str, Any],
        feature_labels: Dict[str, str],
    ) -> Dict[str, Any]:
        original_features = numerical_features + categorical_features
        metadata: Dict[str, Any] = {
            "contract_version": "1.0",
            "original_features": original_features,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "encoding_info": encoding_info,
            "feature_labels": feature_labels,
        }

        baseline_values: Dict[str, Any] = {}
        training_stats: Dict[str, Any] = {}
        for col in numerical_features:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            baseline_values[col] = round(float(series.median()), 3)
            training_stats[col] = {
                "median": round(float(series.median()), 3),
                "mean": round(float(series.mean()), 6),
                "std": round(float(series.std()), 12),
                "q25": round(float(series.quantile(0.25)), 12),
                "q75": round(float(series.quantile(0.75)), 12),
            }
        for col in categorical_features:
            if col not in df.columns:
                continue
            vc = df[col].value_counts()
            mode_val = str(vc.index[0]) if len(vc) > 0 else ""
            baseline_values[col] = mode_val
            top5 = {str(k): int(v) for k, v in vc.head(5).items()}
            training_stats[col] = {
                "mode": mode_val,
                "n_unique": int(df[col].nunique()),
                "top5": top5,
            }

        metadata["baseline_values"] = baseline_values
        metadata["training_stats"] = training_stats

        import hashlib
        sig = hashlib.md5(str(sorted(original_features)).encode()).hexdigest()[:16]
        metadata["training_signature"] = sig

        return metadata

    def validate(self, df: pd.DataFrame) -> DataValidationArtifact:
        logging.info("Starting data validation")
        try:
            missing_results = self._check_missing_values(df)
            critical_missing = self._check_critical_missing(missing_results)
            dup_result = self._check_duplicates(df)
            abnormal_amounts = self._check_abnormal_amounts(df)
            unknown_cats = self._check_unknown_categories(df)
            time_parsing = self._check_time_parsing(df)
            target_dist = self._check_target_distribution(df)
            risk_items = self._build_risk_items(missing_results, dup_result, target_dist)

            errors = []
            warnings = []

            if critical_missing:
                errors.append(f"Critical features with excessive missing values: {critical_missing}")
            for item in risk_items:
                if item["level"] == "CRITICAL":
                    errors.append(item["message"])
                elif item["level"] == "WARNING":
                    warnings.append(item["message"])

            report = {
                "dataset_shape": list(df.shape),
                "missing_values": missing_results,
                "critical_missing_columns": critical_missing,
                "duplicates": dup_result,
                "abnormal_amounts": abnormal_amounts,
                "unknown_categories": unknown_cats,
                "time_parsing": time_parsing,
                "target_distribution": target_dist,
                "risk_items": risk_items,
                "validation_config": {
                    "critical_feature_columns": self.config.critical_feature_columns,
                    "categorical_columns": self.config.categorical_check_columns,
                    "categorical_whitelist": self.config.categorical_whitelist,
                    "amount_columns": self.config.amount_columns,
                    "timestamp_columns": self.config.timestamp_columns,
                    "thresholds": self.config.thresholds,
                },
            }

            report_path = self.registry.save_data_quality_report(report)
            logging.info(f"Data quality report saved to: {report_path}")

            is_valid = len(errors) == 0
            feature_metadata_path = ""

            return DataValidationArtifact(
                is_valid=is_valid,
                report_path=report_path,
                feature_metadata_path=feature_metadata_path,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            raise CustomerException(e, sys)

    def save_feature_metadata(
        self,
        df: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str],
        encoding_info: Dict[str, Any],
        feature_labels: Optional[Dict[str, str]] = None,
    ) -> str:
        try:
            feature_labels = feature_labels or {}
            metadata = self._build_feature_metadata(
                df, numerical_features, categorical_features, encoding_info, feature_labels
            )
            path = self.registry.save_feature_metadata(metadata)
            logging.info(f"Feature metadata saved to: {path}")
            return path
        except Exception as e:
            raise CustomerException(e, sys)
