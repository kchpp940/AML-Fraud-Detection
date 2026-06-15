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
    severity: str


@dataclass
class DuplicateDetail:
    duplicate_row_count: int
    duplicate_row_ratio: float
    severity: str


@dataclass
class AbnormalAmountDetail:
    column: str
    negative_count: int
    zero_count: int
    outlier_count: int
    outlier_lower: float
    outlier_upper: float


@dataclass
class UnknownCategoryValue:
    value: str
    count: int
    ratio: float


@dataclass
class UnknownCategoryDetail:
    column: str
    whitelist_count: int
    unknown_values: List[UnknownCategoryValue]
    total_unknown_count: int
    total_unknown_ratio: float
    severity: str
    threshold_critical: float
    threshold_warning: float


@dataclass
class TimeParsingDetail:
    column: str
    total_count: int
    parse_fail_count: int
    parse_fail_ratio: float


@dataclass
class TargetCheckIssue:
    code: str
    message: str
    severity: str
    ratio: float
    threshold: Optional[float]


@dataclass
class TargetDistributionDetail:
    column: str
    missing_count: int
    missing_ratio: float
    unique_classes: int
    expected_classes: int
    value_counts: Dict[str, int]
    total_count: int
    positive_ratio: float
    imbalance_ratio: float
    issues: List[TargetCheckIssue]
    overall_severity: str


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

TIME_COLUMNS = ["timestamp", "date", "time"]

OUTLIER_IQR_MULTIPLIER = 3.0

SEVERITY_NONE = "NONE"
SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"


def _ratio_to_severity(
    ratio: float,
    critical_threshold: float,
    warning_threshold: float,
) -> str:
    if ratio >= critical_threshold:
        return SEVERITY_CRITICAL
    if ratio >= warning_threshold:
        return SEVERITY_WARNING
    return SEVERITY_NONE


class DataValidation:
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config or TrainingConfig()
        self._resolved = self.training_config.to_resolved_dict()
        self.target_column = self.training_config.features.target_column
        self.drop_columns = self.training_config.features.drop_columns
        self.report_path = self.training_config.artifacts_subpath(
            "data_quality_report.json"
        )
        self.validation_cfg = self.training_config.validation
        self.thresholds = self.validation_cfg.thresholds
        self.categorical_whitelist: Dict[str, set] = {
            col: set(vals)
            for col, vals in self.validation_cfg.categorical_whitelist.items()
        }
        logging.info(
            f"DataValidation initialized: target={self.target_column}, "
            f"report_path={self.report_path}, "
            f"whitelist_columns={list(self.categorical_whitelist.keys())}"
        )

    def _check_missing_values(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        logging.info("Checking missing values...")
        results: List[Dict[str, Any]] = []
        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_ratio = missing_count / len(df) if len(df) > 0 else 0.0
            severity = _ratio_to_severity(
                missing_ratio,
                critical_threshold=self.thresholds.missing_value_critical_ratio,
                warning_threshold=self.thresholds.missing_value_warning_ratio,
            )
            results.append({
                "column": col,
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio, 6),
                "severity": severity,
                "threshold_critical": self.thresholds.missing_value_critical_ratio,
                "threshold_warning": self.thresholds.missing_value_warning_ratio,
            })
        return results

    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        logging.info("Checking duplicate transactions...")
        dup_count = int(df.duplicated().sum())
        dup_ratio = dup_count / len(df) if len(df) > 0 else 0.0
        severity = _ratio_to_severity(
            dup_ratio,
            critical_threshold=1.1,
            warning_threshold=self.thresholds.duplicate_row_warning_ratio,
        )
        return {
            "duplicate_row_count": dup_count,
            "duplicate_row_ratio": round(dup_ratio, 6),
            "severity": severity,
            "threshold_warning": self.thresholds.duplicate_row_warning_ratio,
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
        logging.info("Checking unknown categories against whitelist...")
        results: List[Dict[str, Any]] = []
        total_rows = len(df)
        for col in self.validation_cfg.categorical_columns:
            if col not in df.columns:
                continue
            whitelist = self.categorical_whitelist.get(col, set())
            series = df[col].dropna()
            series_str = series.astype(str).str.strip()
            value_counts = series_str.value_counts()

            if not whitelist:
                results.append({
                    "column": col,
                    "whitelist_defined": False,
                    "whitelist_count": 0,
                    "unique_count": int(value_counts.shape[0]),
                    "unknown_values": [],
                    "total_unknown_count": 0,
                    "total_unknown_ratio": 0.0,
                    "severity": SEVERITY_NONE,
                    "threshold_critical": self.thresholds.unknown_category_critical_ratio,
                    "threshold_warning": self.thresholds.unknown_category_warning_ratio,
                })
                continue

            unknown_counts: List[Dict[str, Any]] = []
            total_unknown = 0
            for val, cnt in value_counts.items():
                if val not in whitelist:
                    ratio = cnt / total_rows if total_rows > 0 else 0.0
                    unknown_counts.append({
                        "value": val,
                        "count": int(cnt),
                        "ratio": round(ratio, 6),
                    })
                    total_unknown += int(cnt)
            total_unknown_ratio = total_unknown / total_rows if total_rows > 0 else 0.0
            severity = _ratio_to_severity(
                total_unknown_ratio,
                critical_threshold=self.thresholds.unknown_category_critical_ratio,
                warning_threshold=self.thresholds.unknown_category_warning_ratio,
            )
            results.append({
                "column": col,
                "whitelist_defined": True,
                "whitelist_count": len(whitelist),
                "unique_count": int(value_counts.shape[0]),
                "unknown_values": unknown_counts,
                "total_unknown_count": total_unknown,
                "total_unknown_ratio": round(total_unknown_ratio, 6),
                "severity": severity,
                "threshold_critical": self.thresholds.unknown_category_critical_ratio,
                "threshold_warning": self.thresholds.unknown_category_warning_ratio,
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
                original_na = int(df[col].isna().sum())
                coerced = pd.to_datetime(df[col], errors="coerce")
                parse_fail_count = max(int(coerced.isna().sum()) - original_na, 0)
            parse_fail_ratio = parse_fail_count / total_count if total_count > 0 else 0.0
            results.append({
                "column": col,
                "total_count": total_count,
                "parse_fail_count": parse_fail_count,
                "parse_fail_ratio": round(parse_fail_ratio, 6),
            })
        return results

    def _check_target_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        logging.info(
            f"Checking target column: missing / classes / imbalance "
            f"(col={self.target_column}, expected_classes={self.thresholds.target_expected_classes})"
        )
        result: Dict[str, Any] = {
            "column": self.target_column,
            "missing_count": 0,
            "missing_ratio": 0.0,
            "missing_threshold": self.thresholds.target_missing_critical_ratio,
            "unique_classes": 0,
            "expected_classes": self.thresholds.target_expected_classes,
            "value_counts": {},
            "total_count": 0,
            "positive_ratio": 0.0,
            "imbalance_ratio": 0.0,
            "imbalance_threshold_critical": self.thresholds.target_imbalance_critical_ratio,
            "imbalance_threshold_warning": self.thresholds.target_imbalance_warning_ratio,
            "issues": [],
            "overall_severity": SEVERITY_NONE,
        }

        if self.target_column not in df.columns:
            result["issues"].append({
                "code": "target_not_found",
                "message": f"Target column '{self.target_column}' not found in dataset",
                "severity": SEVERITY_CRITICAL,
                "ratio": 1.0,
                "threshold": None,
            })
            result["overall_severity"] = SEVERITY_CRITICAL
            return result

        series = df[self.target_column]
        total_count = len(series)
        missing_count = int(series.isna().sum())
        missing_ratio = missing_count / total_count if total_count > 0 else 0.0
        result["missing_count"] = missing_count
        result["missing_ratio"] = round(missing_ratio, 6)

        if missing_ratio > self.thresholds.target_missing_critical_ratio:
            result["issues"].append({
                "code": "target_missing",
                "message": (
                    f"Target column has {missing_count} missing values "
                    f"({missing_ratio*100:.4f}%), "
                    f"exceeds critical threshold {self.thresholds.target_missing_critical_ratio*100:.1f}%"
                ),
                "severity": SEVERITY_CRITICAL,
                "ratio": missing_ratio,
                "threshold": self.thresholds.target_missing_critical_ratio,
            })

        series_valid = series.dropna()
        unique_classes = series_valid.nunique()
        value_counts = series_valid.value_counts().to_dict()
        value_counts_str = {str(k): int(v) for k, v in value_counts.items()}
        result["unique_classes"] = int(unique_classes)
        result["value_counts"] = value_counts_str
        result["total_count"] = total_count

        if unique_classes == 0:
            result["issues"].append({
                "code": "target_empty",
                "message": "Target column has no valid (non-null) values",
                "severity": SEVERITY_CRITICAL,
                "ratio": 1.0,
                "threshold": None,
            })
        elif unique_classes == 1:
            result["issues"].append({
                "code": "target_single_class",
                "message": (
                    f"Target column has only 1 class (expected "
                    f"{self.thresholds.target_expected_classes}): "
                    f"value={list(value_counts_str.keys())[0]}, "
                    f"count={list(value_counts_str.values())[0]}"
                ),
                "severity": SEVERITY_CRITICAL,
                "ratio": 0.0,
                "threshold": float(self.thresholds.target_expected_classes),
            })
        elif unique_classes != self.thresholds.target_expected_classes:
            result["issues"].append({
                "code": "target_non_binary",
                "message": (
                    f"Target column has {unique_classes} unique classes, "
                    f"expected {self.thresholds.target_expected_classes}. "
                    f"Classes: {list(value_counts_str.keys())}"
                ),
                "severity": SEVERITY_WARNING,
                "ratio": float(unique_classes),
                "threshold": float(self.thresholds.target_expected_classes),
            })

        valid_count = len(series_valid)
        if valid_count > 0:
            numeric_series = pd.to_numeric(series_valid, errors="coerce")
            numeric_valid = numeric_series.dropna()
            if len(numeric_valid) > 0:
                positive_count = int((numeric_valid == 1).sum())
            else:
                positive_count = 0
            positive_ratio = positive_count / valid_count
            minority_ratio = min(positive_count / valid_count, (valid_count - positive_count) / valid_count) if valid_count > 0 else 0.0
        else:
            positive_ratio = 0.0
            minority_ratio = 0.0
        result["positive_ratio"] = round(positive_ratio, 6)
        result["imbalance_ratio"] = round(minority_ratio, 6)

        if valid_count > 0 and minority_ratio > 0 and minority_ratio < self.thresholds.target_imbalance_critical_ratio:
            result["issues"].append({
                "code": "target_severe_imbalance",
                "message": (
                    f"Target column is severely imbalanced: minority ratio = "
                    f"{minority_ratio*100:.4f}%, "
                    f"critical threshold = "
                    f"{self.thresholds.target_imbalance_critical_ratio*100:.1f}%"
                ),
                "severity": SEVERITY_CRITICAL,
                "ratio": minority_ratio,
                "threshold": self.thresholds.target_imbalance_critical_ratio,
            })
        elif valid_count > 0 and minority_ratio > 0 and minority_ratio < self.thresholds.target_imbalance_warning_ratio:
            result["issues"].append({
                "code": "target_mild_imbalance",
                "message": (
                    f"Target column is mildly imbalanced: minority ratio = "
                    f"{minority_ratio*100:.4f}%, "
                    f"warning threshold = "
                    f"{self.thresholds.target_imbalance_warning_ratio*100:.1f}%"
                ),
                "severity": SEVERITY_WARNING,
                "ratio": minority_ratio,
                "threshold": self.thresholds.target_imbalance_warning_ratio,
            })

        severities = [it["severity"] for it in result["issues"]]
        if SEVERITY_CRITICAL in severities:
            result["overall_severity"] = SEVERITY_CRITICAL
        elif SEVERITY_WARNING in severities:
            result["overall_severity"] = SEVERITY_WARNING
        return result

    def _build_risk_items(
        self,
        missing_values: List[Dict[str, Any]],
        duplicates: Dict[str, Any],
        unknown_categories: List[Dict[str, Any]],
        target_distribution: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        risks: List[Dict[str, str]] = []

        for m in missing_values:
            if m["severity"] == SEVERITY_CRITICAL:
                risks.append({
                    "level": SEVERITY_CRITICAL,
                    "category": "missing_values",
                    "message": (
                        f"Column '{m['column']}' has severe missing data: "
                        f"{m['missing_count']} rows ({m['missing_ratio']*100:.2f}%), "
                        f"critical threshold = {m['threshold_critical']*100:.1f}%"
                    ),
                })
            elif m["severity"] == SEVERITY_WARNING:
                risks.append({
                    "level": SEVERITY_WARNING,
                    "category": "missing_values",
                    "message": (
                        f"Column '{m['column']}' has notable missing data: "
                        f"{m['missing_count']} rows ({m['missing_ratio']*100:.2f}%), "
                        f"warning threshold = {m['threshold_warning']*100:.1f}%"
                    ),
                })

        if duplicates.get("severity") == SEVERITY_WARNING:
            risks.append({
                "level": SEVERITY_WARNING,
                "category": "duplicates",
                "message": (
                    f"Duplicate rows: {duplicates['duplicate_row_count']} "
                    f"({duplicates['duplicate_row_ratio']*100:.2f}%), "
                    f"warning threshold = {duplicates['threshold_warning']*100:.1f}%"
                ),
            })

        for uc in unknown_categories:
            if uc["severity"] == SEVERITY_CRITICAL:
                risks.append({
                    "level": SEVERITY_CRITICAL,
                    "category": "unknown_categories",
                    "message": (
                        f"Column '{uc['column']}' has severe unknown category values: "
                        f"{uc['total_unknown_count']} rows ({uc['total_unknown_ratio']*100:.2f}%), "
                        f"critical threshold = {uc['threshold_critical']*100:.1f}%"
                    ),
                })
            elif uc["severity"] == SEVERITY_WARNING:
                risks.append({
                    "level": SEVERITY_WARNING,
                    "category": "unknown_categories",
                    "message": (
                        f"Column '{uc['column']}' has notable unknown category values: "
                        f"{uc['total_unknown_count']} rows ({uc['total_unknown_ratio']*100:.2f}%), "
                        f"warning threshold = {uc['threshold_warning']*100:.1f}%"
                    ),
                })

        for issue in target_distribution.get("issues", []):
            risks.append({
                "level": issue["severity"],
                "category": f"target__{issue['code']}",
                "message": issue["message"],
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
                missing_values, duplicates, unknown_categories, target_distribution
            )

            critical_missing_cols = [
                m["column"] for m in missing_values
                if m["severity"] == SEVERITY_CRITICAL
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
            if level == SEVERITY_CRITICAL:
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
        risk_items = report_data.get("risk_items", [])
        critical_issues = [
            item for item in risk_items
            if item.get("level") == SEVERITY_CRITICAL
        ]
        if not critical_issues:
            return

        error_messages = []
        seen_categories = set()
        for item in critical_issues:
            category = item.get("category", "unknown")
            message = item.get("message", "")
            if category not in seen_categories:
                error_messages.append(f"- [{category}] {message}")
                seen_categories.add(category)
            else:
                error_messages.append(f"  - ({category}) {message}")

        summary = "\n".join(error_messages)
        raise CustomerException(
            ValueError(
                f"Cannot proceed with training: {len(critical_issues)} critical "
                f"data quality issue(s) detected. Review the quality report and "
                f"fix data before training.\n{summary}"
            ),
            sys,
        )
