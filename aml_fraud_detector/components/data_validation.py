import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.configuration import TrainingConfig


@dataclass
class MissingValueDetail:
    column: str
    missing_count: int
    missing_ratio: float
    is_critical: bool


@dataclass
class DuplicateDetail:
    duplicate_row_count: int
    duplicate_row_ratio: float


@dataclass
class AbnormalAmountDetail:
    column: str
    negative_count: int
    zero_count: int
    outlier_count: int
    outlier_lower: float
    outlier_upper: float


@dataclass
class UnknownCategoryDetail:
    column: str
    unique_values: List[str]
    sample_unknown: List[str]


@dataclass
class TimeParsingDetail:
    column: str
    total_count: int
    parse_fail_count: int
    parse_fail_ratio: float


@dataclass
class TargetDistributionDetail:
    column: str
    value_counts: Dict[str, int]
    total_count: int
    positive_ratio: float
    is_severely_imbalanced: bool


@dataclass
class DataValidationArtifact:
    quality_report_path: str


@dataclass
class DataQualityReport:
    generated_at: str = ""
    dataset_shape: List[int] = field(default_factory=list)
    missing_values: List[Dict[str, Any]] = field(default_factory=list)
    critical_missing_columns: List[str] = field(default_factory=list)
    duplicates: Dict[str, Any] = field(default_factory=dict)
    abnormal_amounts: List[Dict[str, Any]] = field(default_factory=list)
    unknown_categories: List[Dict[str, Any]] = field(default_factory=list)
    time_parsing: List[Dict[str, Any]] = field(default_factory=list)
    target_distribution: Dict[str, Any] = field(default_factory=dict)
    risk_items: List[Dict[str, str]] = field(default_factory=list)


AMOUNT_COLUMNS = [
    "amount_received", "amount_paid",
    "amount", "transaction_amount",
    "received_amount", "paid_amount",
]

CATEGORICAL_COLUMNS = [
    "payment_format", "payment_currency", "receiving_currency",
    "from_bank", "to_bank", "day",
]

TIME_COLUMNS = ["timestamp", "date", "time"]

CRITICAL_FEATURE_COLUMNS = [
    "amount_received", "amount_paid",
    "payment_format", "from_bank", "to_bank",
    "account", "account_1",
]

MISSING_RATIO_CRITICAL_THRESHOLD = 0.3
TARGET_IMBALANCE_RATIO_THRESHOLD = 0.01
OUTLIER_IQR_MULTIPLIER = 3.0


class DataValidation:
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config or TrainingConfig()
        self._resolved = self.training_config.to_resolved_dict()
        self.target_column = self.training_config.features.target_column
        self.drop_columns = self.training_config.features.drop_columns
        self.report_path = self.training_config.artifacts_subpath(
            "data_quality_report.json"
        )
        logging.info(
            f"DataValidation initialized: target={self.target_column}, "
            f"report_path={self.report_path}"
        )

    def _check_missing_values(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        logging.info("Checking missing values...")
        results: List[Dict[str, Any]] = []
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_ratio = missing_count / len(df) if len(df) > 0 else 0.0
            is_critical = (
                col in CRITICAL_FEATURE_COLUMNS
                and missing_ratio >= MISSING_RATIO_CRITICAL_THRESHOLD
            )
            results.append({
                "column": col,
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio, 6),
                "is_critical": is_critical,
            })
        return results

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        logging.info("Checking duplicate transactions...")
        dup_count = int(df.duplicated().sum())
        dup_ratio = dup_count / len(df) if len(df) > 0 else 0.0
        return {
            "duplicate_row_count": dup_count,
            "duplicate_row_ratio": round(dup_ratio, 6),
        }

    def _check_abnormal_amounts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        logging.info("Checking abnormal amounts...")
        results: List[Dict[str, Any]] = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        amount_cols = [c for c in AMOUNT_COLUMNS if c in numeric_cols]
        if not amount_cols:
            non_target_numeric = [
                c for c in numeric_cols if c != self.target_column
            ]
            amount_cols = non_target_numeric[:3]
        for col in amount_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            negative_count = int((series < 0).sum())
            zero_count = int((series == 0).sum())
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())
            results.append({
                "column": col,
                "negative_count": negative_count,
                "zero_count": zero_count,
                "outlier_count": outlier_count,
                "outlier_lower": round(float(lower), 4),
                "outlier_upper": round(float(upper), 4),
            })
        return results

    def _check_unknown_categories(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        logging.info("Checking unknown categories...")
        results: List[Dict[str, Any]] = []
        cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        for col in cat_cols:
            unique_vals = sorted(df[col].dropna().unique().tolist())
            sample_unknown: List[str] = []
            for val in unique_vals:
                val_str = str(val).strip()
                if val_str == "" or val_str.lower() in ("nan", "none", "null", "unknown", "n/a", "-"):
                    sample_unknown.append(val_str)
            results.append({
                "column": col,
                "unique_count": len(unique_vals),
                "sample_unknown": sample_unknown[:10],
            })
        return results

    def _check_time_parsing(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        logging.info("Checking time field parsing...")
        results: List[Dict[str, Any]] = []
        for col in TIME_COLUMNS:
            if col not in df.columns:
                continue
            total_count = len(df)
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                parse_fail_count = int(df[col].isna().sum())
            else:
                coerced = pd.to_datetime(df[col], errors="coerce")
                parse_fail_count = int(coerced.isna().sum() - df[col].isna().sum())
                parse_fail_count = max(parse_fail_count, 0)
            parse_fail_ratio = parse_fail_count / total_count if total_count > 0 else 0.0
            results.append({
                "column": col,
                "total_count": total_count,
                "parse_fail_count": parse_fail_count,
                "parse_fail_ratio": round(parse_fail_ratio, 6),
            })
        return results

    def _check_target_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        logging.info(f"Checking target column distribution: {self.target_column}")
        if self.target_column not in df.columns:
            return {
                "column": self.target_column,
                "error": f"Target column '{self.target_column}' not found in dataset",
                "value_counts": {},
                "total_count": 0,
                "positive_ratio": 0.0,
                "is_severely_imbalanced": True,
            }
        series = df[self.target_column]
        value_counts = series.value_counts().to_dict()
        value_counts_str = {str(k): int(v) for k, v in value_counts.items()}
        total_count = len(series)
        positive_values = series[series == 1]
        positive_ratio = len(positive_values) / total_count if total_count > 0 else 0.0
        is_imbalanced = positive_ratio < TARGET_IMBALANCE_RATIO_THRESHOLD and positive_ratio > 0.0
        if positive_ratio == 0.0 and total_count > 0:
            is_imbalanced = True
        return {
            "column": self.target_column,
            "value_counts": value_counts_str,
            "total_count": total_count,
            "positive_ratio": round(positive_ratio, 6),
            "is_severely_imbalanced": is_imbalanced,
        }

    def _build_risk_items(
        self,
        missing_values: List[Dict[str, Any]],
        duplicates: Dict[str, Any],
        abnormal_amounts: List[Dict[str, Any]],
        target_distribution: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        risks: List[Dict[str, str]] = []

        critical_missing = [
            m["column"] for m in missing_values if m["is_critical"]
        ]
        if critical_missing:
            risks.append({
                "level": "CRITICAL",
                "category": "missing_values",
                "message": (
                    f"Critical columns with >= {MISSING_RATIO_CRITICAL_THRESHOLD*100:.0f}% "
                    f"missing: {critical_missing}"
                ),
            })

        high_missing = [
            m["column"] for m in missing_values
            if m["missing_ratio"] >= 0.1 and not m["is_critical"]
        ]
        if high_missing:
            risks.append({
                "level": "WARNING",
                "category": "missing_values",
                "message": f"Columns with >= 10% missing: {high_missing}",
            })

        dup_ratio = duplicates.get("duplicate_row_ratio", 0)
        if dup_ratio >= 0.1:
            risks.append({
                "level": "WARNING",
                "category": "duplicates",
                "message": (
                    f"Duplicate rows: {duplicates['duplicate_row_count']} "
                    f"({dup_ratio*100:.1f}%)"
                ),
            })

        for amt in abnormal_amounts:
            if amt["negative_count"] > 0:
                risks.append({
                    "level": "WARNING",
                    "category": "abnormal_amount",
                    "message": (
                        f"Column '{amt['column']}' has {amt['negative_count']} "
                        f"negative values"
                    ),
                })

        target_error = target_distribution.get("error")
        if target_error:
            risks.append({
                "level": "CRITICAL",
                "category": "target",
                "message": target_error,
            })
        elif target_distribution.get("is_severely_imbalanced"):
            positive_ratio = target_distribution.get("positive_ratio", 0)
            risks.append({
                "level": "CRITICAL",
                "category": "target_imbalance",
                "message": (
                    f"Target column '{target_distribution['column']}' is severely "
                    f"imbalanced: positive ratio = {positive_ratio*100:.4f}% "
                    f"(threshold: {TARGET_IMBALANCE_RATIO_THRESHOLD*100:.1f}%)"
                ),
            })

        return risks

    def generate_report(self, df: pd.DataFrame) -> DataQualityReport:
        logging.info("Generating data quality report...")
        try:
            missing_values = self._check_missing_values(df)
            duplicates = self._check_duplicates(df)
            abnormal_amounts = self._check_abnormal_amounts(df)
            unknown_categories = self._check_unknown_categories(df)
            time_parsing = self._check_time_parsing(df)
            target_distribution = self._check_target_distribution(df)

            risk_items = self._build_risk_items(
                missing_values, duplicates, abnormal_amounts, target_distribution
            )

            critical_missing_cols = [
                m["column"] for m in missing_values if m["is_critical"]
            ]

            report = DataQualityReport(
                generated_at=datetime.now().isoformat(),
                dataset_shape=list(df.shape),
                missing_values=missing_values,
                critical_missing_columns=critical_missing_cols,
                duplicates=duplicates,
                abnormal_amounts=abnormal_amounts,
                unknown_categories=unknown_categories,
                time_parsing=time_parsing,
                target_distribution=target_distribution,
                risk_items=risk_items,
            )

            logging.info(
                f"Data quality report generated: {len(risk_items)} risk item(s) found"
            )
            return report

        except Exception as e:
            raise CustomerException(e, sys)

    def save_report(self, report: DataQualityReport) -> str:
        logging.info(f"Saving data quality report to: {self.report_path}")
        try:
            os.makedirs(os.path.dirname(self.report_path), exist_ok=True)
            data = asdict(report)
            abs_path = os.path.abspath(self.report_path)
            with open(abs_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            logging.info(f"Data quality report saved: {abs_path}")
            return abs_path
        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_validation(
        self, data_path: Optional[str] = None
    ) -> DataValidationArtifact:
        logging.info("Entered the 'data validation' method or component")
        try:
            resolved_path = data_path or self.training_config.resolve_source_path()
            if not os.path.isfile(resolved_path):
                raise CustomerException(
                    FileNotFoundError(
                        f"Data file not found for validation: {resolved_path}"
                    ),
                    sys,
                )
            logging.info(
                f"Reading full dataset for validation from: {resolved_path}"
            )
            df = pd.read_csv(resolved_path)
            logging.info(
                f"Loaded full dataset for quality validation: shape={df.shape}"
            )

            report = self.generate_report(df)
            abs_report_path = self.save_report(report)
            self._log_risk_summary(report)

            artifact = DataValidationArtifact(quality_report_path=abs_report_path)
            logging.info(
                f"Data validation artifact created: {artifact.quality_report_path}"
            )
            return artifact

        except Exception as e:
            raise CustomerException(e, sys)

    def _log_risk_summary(self, report: DataQualityReport) -> None:
        if not report.risk_items:
            logging.info("Data quality report: no risk items detected")
            return
        for item in report.risk_items:
            level = item["level"]
            message = item["message"]
            if level == "CRITICAL":
                logging.warning(f"[DATA QUALITY - CRITICAL] {message}")
            else:
                logging.info(f"[DATA QUALITY - WARNING] {message}")

    @staticmethod
    def load_report(report_path: str) -> Dict[str, Any]:
        logging.info(f"Loading data quality report from: {report_path}")
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise CustomerException(e, sys)

    @staticmethod
    def check_critical_issues(report_data: Dict[str, Any]) -> None:
        critical_missing = report_data.get("critical_missing_columns", [])
        target_dist = report_data.get("target_distribution", {})
        target_error = target_dist.get("error")

        if target_error:
            raise CustomerException(
                ValueError(
                    f"Cannot proceed with training: {target_error}"
                ),
                sys,
            )

        if critical_missing:
            raise CustomerException(
                ValueError(
                    f"Cannot proceed with training: critical columns with severe "
                    f"missing data (>= {MISSING_RATIO_CRITICAL_THRESHOLD*100:.0f}%): "
                    f"{critical_missing}. Fix data quality before training."
                ),
                sys,
            )

        if target_dist.get("is_severely_imbalanced"):
            positive_ratio = target_dist.get("positive_ratio", 0)
            col = target_dist.get("column", "unknown")
            raise CustomerException(
                ValueError(
                    f"Cannot proceed with training: target column '{col}' has "
                    f"abnormal distribution (positive ratio = "
                    f"{positive_ratio*100:.4f}%, threshold = "
                    f"{TARGET_IMBALANCE_RATIO_THRESHOLD*100:.1f}%). "
                    f"Review target column data."
                ),
                sys,
            )
