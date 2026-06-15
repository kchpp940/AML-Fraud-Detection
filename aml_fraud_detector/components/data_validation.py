import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.configuration import TrainingConfig
from aml_fraud_detector.entity.artifact_entity import DataValidationArtifact
from aml_fraud_detector.entity.config_entity import DataValidationConfig


def _severity_from_ratio(
    ratio: float, critical: float, warning: float
) -> str:
    if ratio >= critical:
        return "CRITICAL"
    if ratio >= warning:
        return "WARNING"
    return "NONE"


class DataValidation:
    def __init__(
        self,
        training_config: Optional[TrainingConfig] = None,
        validation_config: Optional[DataValidationConfig] = None,
    ):
        self.training_config = training_config or TrainingConfig()
        self._resolved = self.training_config.to_resolved_dict()
        self.validation_config = validation_config or DataValidationConfig()

        tc = self.training_config
        self.report_path = tc.artifacts_subpath(
            self.validation_config.report_file_name
        )

        target_col = self._resolved["features"]["target_column"]
        drop_cols = set(self._resolved["features"]["drop_columns"])
        all_expected = self._infer_expected_columns()
        self.critical_feature_columns = [
            c for c in all_expected
            if c != target_col and c not in drop_cols
        ]
        self.validation_config.critical_feature_columns = self.critical_feature_columns

        logging.info(
            f"DataValidation initialized: report_out={self.report_path}, "
            f"critical_features={self.critical_feature_columns}"
        )

    def _infer_expected_columns(self) -> List[str]:
        all_cols = set()
        for amt in self.validation_config.amount_columns:
            all_cols.add(amt)
        for ts in self.validation_config.timestamp_columns:
            all_cols.add(ts)
        for cat in self.validation_config.categorical_columns:
            all_cols.add(cat)
        all_cols.add(self._resolved["features"]["target_column"])
        return sorted(all_cols)

    def _check_missing_values(
        self, df: pd.DataFrame
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        cfg = self.validation_config
        results: List[Dict[str, Any]] = []
        critical_cols: List[str] = []
        total = len(df)
        if total == 0:
            return results, critical_cols

        for col in df.columns:
            missing = int(df[col].isna().sum())
            ratio = missing / total if total > 0 else 0.0
            severity = _severity_from_ratio(
                ratio,
                cfg.missing_value_critical_ratio,
                cfg.missing_value_warning_ratio,
            )
            entry = {
                "column": col,
                "missing_count": missing,
                "missing_ratio": round(ratio, 6),
                "severity": severity,
                "threshold_critical": cfg.missing_value_critical_ratio,
                "threshold_warning": cfg.missing_value_warning_ratio,
            }
            results.append(entry)
            if severity == "CRITICAL" and col in self.critical_feature_columns:
                critical_cols.append(col)

        return results, critical_cols

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        cfg = self.validation_config
        total = len(df)
        dup_count = int(df.duplicated().sum()) if total > 0 else 0
        dup_ratio = dup_count / total if total > 0 else 0.0
        severity = (
            "WARNING" if dup_ratio >= cfg.duplicate_row_warning_ratio else "NONE"
        )
        return {
            "duplicate_row_count": dup_count,
            "duplicate_row_ratio": round(dup_ratio, 6),
            "severity": severity,
            "threshold_warning": cfg.duplicate_row_warning_ratio,
        }

    def _check_abnormal_amounts(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for col in self.validation_config.amount_columns:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            negative = int((series < 0).sum())
            zero = int((series == 0).sum())
            valid = series.dropna()
            if len(valid) >= 4:
                q1 = valid.quantile(0.25)
                q3 = valid.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count = int(((valid < lower) | (valid > upper)).sum())
            else:
                lower = None
                upper = None
                outlier_count = 0
            results.append({
                "column": col,
                "negative_count": negative,
                "zero_count": zero,
                "outlier_count": outlier_count,
                "outlier_lower": float(lower) if lower is not None else None,
                "outlier_upper": float(upper) if upper is not None else None,
            })
        return results

    def _check_unknown_categories(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        cfg = self.validation_config
        results: List[Dict[str, Any]] = []
        total = len(df)
        if total == 0:
            return results

        for col, whitelist in cfg.categorical_whitelist.items():
            if col not in df.columns:
                continue
            series = df[col].astype(str).replace({"nan": np.nan, "None": np.nan})
            valid_mask = series.isin(whitelist) | series.isna()
            unknown_count = int((~valid_mask).sum())
            ratio = unknown_count / total if total > 0 else 0.0
            severity = _severity_from_ratio(
                ratio,
                cfg.unknown_category_critical_ratio,
                cfg.unknown_category_warning_ratio,
            )
            unknown_values = sorted(
                [str(v) for v in series[~valid_mask].unique().tolist()]
            )
            results.append({
                "column": col,
                "unknown_count": unknown_count,
                "unknown_ratio": round(ratio, 6),
                "severity": severity,
                "unknown_values": unknown_values[:20],
                "threshold_critical": cfg.unknown_category_critical_ratio,
                "threshold_warning": cfg.unknown_category_warning_ratio,
            })
        return results

    def _check_time_parsing(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for col in self.validation_config.timestamp_columns:
            if col not in df.columns:
                continue
            total = len(df)
            if total == 0:
                results.append({
                    "column": col,
                    "total_count": 0,
                    "parse_fail_count": 0,
                    "parse_fail_ratio": 0.0,
                })
                continue
            parsed = pd.to_datetime(df[col], errors="coerce")
            fail_count = int(parsed.isna().sum() - df[col].isna().sum())
            fail_count = max(0, fail_count)
            results.append({
                "column": col,
                "total_count": total,
                "parse_fail_count": fail_count,
                "parse_fail_ratio": round(fail_count / total, 6) if total > 0 else 0.0,
            })
        return results

    def _check_target_distribution(
        self, df: pd.DataFrame
    ) -> Dict[str, Any]:
        cfg = self.validation_config
        target_col = self._resolved["features"]["target_column"]
        issues: List[Dict[str, Any]] = []
        overall_severity = "NONE"

        if target_col not in df.columns:
            issues.append({
                "code": "target_missing_column",
                "message": f"Target column '{target_col}' does not exist in dataset",
                "severity": "CRITICAL",
            })
            return {
                "column": target_col,
                "missing_count": 0,
                "missing_ratio": 0.0,
                "missing_threshold": cfg.target_missing_critical_ratio,
                "unique_classes": 0,
                "expected_classes": cfg.target_expected_classes,
                "value_counts": {},
                "total_count": 0,
                "positive_ratio": 0.0,
                "imbalance_ratio": 0.0,
                "imbalance_threshold_critical": cfg.target_imbalance_critical_ratio,
                "imbalance_threshold_warning": cfg.target_imbalance_warning_ratio,
                "issues": issues,
                "overall_severity": "CRITICAL",
            }

        total = len(df)
        missing = int(df[target_col].isna().sum())
        missing_ratio = missing / total if total > 0 else 0.0

        if (
            cfg.target_missing_critical_ratio > 0
            and missing_ratio > cfg.target_missing_critical_ratio
        ):
            issues.append({
                "code": "target_high_missing",
                "message": (
                    f"Target column has too many missing values: "
                    f"{missing_ratio:.2%} > critical threshold {cfg.target_missing_critical_ratio:.2%}"
                ),
                "severity": "CRITICAL",
                "ratio": missing_ratio,
                "threshold": cfg.target_missing_critical_ratio,
            })
            overall_severity = "CRITICAL"
        elif missing > 0:
            issues.append({
                "code": "target_has_missing",
                "message": f"Target column has {missing} missing values",
                "severity": "WARNING",
                "ratio": missing_ratio,
                "threshold": cfg.target_missing_critical_ratio,
            })
            if overall_severity != "CRITICAL":
                overall_severity = "WARNING"

        value_counts = df[target_col].dropna().value_counts().to_dict()
        value_counts_str = {str(k): int(v) for k, v in value_counts.items()}
        unique_classes = len(value_counts)
        total_non_missing = int(sum(value_counts.values()))

        if unique_classes != cfg.target_expected_classes:
            issues.append({
                "code": "target_class_count_mismatch",
                "message": (
                    f"Target column has {unique_classes} unique classes, "
                    f"expected {cfg.target_expected_classes}"
                ),
                "severity": "CRITICAL",
                "actual": unique_classes,
                "expected": cfg.target_expected_classes,
            })
            overall_severity = "CRITICAL"

        positive_ratio = 0.0
        imbalance_ratio = 0.0
        if total_non_missing > 0 and value_counts:
            counts = sorted(value_counts.values(), reverse=True)
            majority = counts[0]
            minority = counts[-1]
            imbalance_ratio = minority / total_non_missing
            if len([v for v in df[target_col].dropna().unique() if v not in (0, "0")]) > 0:
                pos_sum = sum(
                    v for k, v in value_counts.items()
                    if str(k) not in ("0", "0.0", "False", "false")
                )
                positive_ratio = pos_sum / total_non_missing
            else:
                positive_ratio = imbalance_ratio

            if imbalance_ratio < cfg.target_imbalance_critical_ratio:
                issues.append({
                    "code": "target_severe_imbalance",
                    "message": (
                        f"Target column is severely imbalanced: minority ratio = "
                        f"{imbalance_ratio:.4%}, critical threshold = {cfg.target_imbalance_critical_ratio:.2%}"
                    ),
                    "severity": "CRITICAL",
                    "ratio": imbalance_ratio,
                    "threshold": cfg.target_imbalance_critical_ratio,
                })
                overall_severity = "CRITICAL"
            elif imbalance_ratio < cfg.target_imbalance_warning_ratio:
                issues.append({
                    "code": "target_mild_imbalance",
                    "message": (
                        f"Target column is mildly imbalanced: minority ratio = "
                        f"{imbalance_ratio:.4%}, warning threshold = {cfg.target_imbalance_warning_ratio:.2%}"
                    ),
                    "severity": "WARNING",
                    "ratio": imbalance_ratio,
                    "threshold": cfg.target_imbalance_warning_ratio,
                })
                if overall_severity != "CRITICAL":
                    overall_severity = "WARNING"

        return {
            "column": target_col,
            "missing_count": missing,
            "missing_ratio": round(missing_ratio, 6),
            "missing_threshold": cfg.target_missing_critical_ratio,
            "unique_classes": unique_classes,
            "expected_classes": cfg.target_expected_classes,
            "value_counts": value_counts_str,
            "total_count": total_non_missing,
            "positive_ratio": round(positive_ratio, 6),
            "imbalance_ratio": round(imbalance_ratio, 6),
            "imbalance_threshold_critical": cfg.target_imbalance_critical_ratio,
            "imbalance_threshold_warning": cfg.target_imbalance_warning_ratio,
            "issues": issues,
            "overall_severity": overall_severity,
        }

    def _collect_risk_items(
        self,
        missing_results: List[Dict[str, Any]],
        duplicates: Dict[str, Any],
        unknown_cats: List[Dict[str, Any]],
        target_dist: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        risks: List[Dict[str, str]] = []

        for m in missing_results:
            if m["severity"] in ("WARNING", "CRITICAL"):
                risks.append({
                    "level": m["severity"],
                    "category": f"missing__{m['column']}",
                    "message": (
                        f"Column '{m['column']}' has "
                        f"{m['missing_count']} missing values "
                        f"({m['missing_ratio']:.2%})"
                    ),
                })

        if duplicates["severity"] == "WARNING":
            risks.append({
                "level": "WARNING",
                "category": "duplicates",
                "message": (
                    f"Dataset contains {duplicates['duplicate_row_count']} duplicate rows "
                    f"({duplicates['duplicate_row_ratio']:.2%})"
                ),
            })

        for uc in unknown_cats:
            if uc["severity"] in ("WARNING", "CRITICAL"):
                risks.append({
                    "level": uc["severity"],
                    "category": f"unknown_category__{uc['column']}",
                    "message": (
                        f"Column '{uc['column']}' has {uc['unknown_count']} unknown values "
                        f"({uc['unknown_ratio']:.2%}); samples: {uc['unknown_values'][:5]}"
                    ),
                })

        for issue in target_dist.get("issues", []):
            risks.append({
                "level": issue["severity"],
                "category": f"target__{issue['code']}",
                "message": issue["message"],
            })

        return risks

    def _save_report(self, report: Dict[str, Any]) -> str:
        os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logging.info(f"Data quality report saved to: {self.report_path}")
        return os.path.abspath(self.report_path)

    def _resolve_critical_feature_columns(self, df: pd.DataFrame) -> List[str]:
        target_col = self._resolved["features"]["target_column"]
        drop_cols = set(self._resolved["features"]["drop_columns"])
        drop_cols.add(target_col)
        critical_cols = [c for c in df.columns if c not in drop_cols]
        return critical_cols

    def initiate_data_validation(
        self, df: Optional[pd.DataFrame] = None, data_path: Optional[str] = None
    ) -> DataValidationArtifact:
        logging.info("Entered the 'data validation' method or component")
        try:
            if df is None:
                if data_path is None:
                    data_path = self.training_config.artifacts_subpath("data.csv")
                logging.info(f"Loading data from path for validation: {data_path}")
                df = pd.read_csv(data_path)
                df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')

            logging.info(f"Validating dataset with shape={df.shape}")

            self.critical_feature_columns = self._resolve_critical_feature_columns(df)
            self.validation_config.critical_feature_columns = self.critical_feature_columns
            logging.info(
                f"Resolved critical feature columns for validation: "
                f"{self.critical_feature_columns}"
            )

            missing_results, critical_missing_cols = self._check_missing_values(df)
            duplicates = self._check_duplicates(df)
            abnormal_amounts = self._check_abnormal_amounts(df)
            unknown_categories = self._check_unknown_categories(df)
            time_parsing = self._check_time_parsing(df)
            target_dist = self._check_target_distribution(df)

            risk_items = self._collect_risk_items(
                missing_results, duplicates, unknown_categories, target_dist
            )

            report: Dict[str, Any] = {
                "generated_at": datetime.now().isoformat(),
                "dataset_shape": [int(df.shape[0]), int(df.shape[1])],
                "missing_values": missing_results,
                "critical_missing_columns": critical_missing_cols,
                "duplicates": duplicates,
                "abnormal_amounts": abnormal_amounts,
                "unknown_categories": unknown_categories,
                "time_parsing": time_parsing,
                "target_distribution": target_dist,
                "risk_items": risk_items,
                "validation_config": {
                    "critical_feature_columns": list(self.critical_feature_columns),
                    "categorical_columns": list(self.validation_config.categorical_columns),
                    "categorical_whitelist": {
                        k: list(v)
                        for k, v in self.validation_config.categorical_whitelist.items()
                    },
                    "amount_columns": list(self.validation_config.amount_columns),
                    "timestamp_columns": list(self.validation_config.timestamp_columns),
                    "thresholds": {
                        "missing_value_critical_ratio": self.validation_config.missing_value_critical_ratio,
                        "missing_value_warning_ratio": self.validation_config.missing_value_warning_ratio,
                        "unknown_category_critical_ratio": self.validation_config.unknown_category_critical_ratio,
                        "unknown_category_warning_ratio": self.validation_config.unknown_category_warning_ratio,
                        "duplicate_row_warning_ratio": self.validation_config.duplicate_row_warning_ratio,
                        "target_missing_critical_ratio": self.validation_config.target_missing_critical_ratio,
                        "target_expected_classes": self.validation_config.target_expected_classes,
                        "target_imbalance_critical_ratio": self.validation_config.target_imbalance_critical_ratio,
                        "target_imbalance_warning_ratio": self.validation_config.target_imbalance_warning_ratio,
                    },
                },
            }

            report_path = self._save_report(report)

            critical_issues = [
                r["message"] for r in risk_items if r["level"] == "CRITICAL"
            ]
            warning_issues = [
                r["message"] for r in risk_items if r["level"] == "WARNING"
            ]
            is_valid = len(critical_issues) == 0

            logging.info(
                f"Data validation complete: is_valid={is_valid}, "
                f"critical_issues={len(critical_issues)}, "
                f"warning_issues={len(warning_issues)}"
            )

            return DataValidationArtifact(
                report_path=report_path,
                report=report,
                is_valid=is_valid,
                critical_issues=critical_issues,
                warning_issues=warning_issues,
            )

        except Exception as e:
            raise CustomerException(e, sys)
