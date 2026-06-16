
import sys
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.configuration import TrainingConfig
from aml_fraud_detector.utils.feature_normalization import (
    normalize_column_names,
    derive_temporal_features,
    locate_target_column,
    clean_categorical_fields,
    report_missing_fields,
    infer_feature_types,
    NormalizationReport,
)


@dataclass
class DataValidationConfig:
    data_quality_report_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    data_quality_report_path: str
    feature_columns: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    target_column: str = ""
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    normalization_report: Optional[NormalizationReport] = None


class DataValidation:
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config or TrainingConfig()
        self._resolved = self.training_config.to_resolved_dict()
        tc = self.training_config
        self.data_validation_config = DataValidationConfig(
            data_quality_report_path=tc.artifacts_subpath("data_quality_report.json")
        )

        self.critical_feature_columns = ["account", "account_1", "amount_received", "payment_format"]
        self.categorical_whitelist = {}
        self.amount_columns = ["amount_received", "amount_paid"]
        self.timestamp_columns = ["timestamp"]
        self.thresholds = {
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

        logging.info(
            f"DataValidation initialized with resolved config: "
            f"target={self._resolved['features']['target_column']}, "
            f"drop={self._resolved['features']['drop_columns']}"
        )

    def _check_missing_values(self, df: pd.DataFrame) -> List[Dict]:
        missing_values = []
        total_rows = len(df)
        if total_rows == 0:
            return missing_values

        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_ratio = missing_count / total_rows

            if missing_ratio >= self.thresholds["missing_value_critical_ratio"]:
                severity = "CRITICAL"
            elif missing_ratio >= self.thresholds["missing_value_warning_ratio"]:
                severity = "WARNING"
            else:
                severity = "NONE"

            missing_values.append({
                "column": col,
                "missing_count": missing_count,
                "missing_ratio": missing_ratio,
                "severity": severity,
                "threshold_critical": self.thresholds["missing_value_critical_ratio"],
                "threshold_warning": self.thresholds["missing_value_warning_ratio"],
            })

        return missing_values

    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        total_rows = len(df)
        if total_rows == 0:
            return {
                "duplicate_row_count": 0,
                "duplicate_row_ratio": 0.0,
                "severity": "NONE",
                "threshold_warning": self.thresholds["duplicate_row_warning_ratio"],
            }

        duplicate_count = int(df.duplicated().sum())
        duplicate_ratio = duplicate_count / total_rows

        if duplicate_ratio >= self.thresholds["duplicate_row_warning_ratio"]:
            severity = "WARNING"
        else:
            severity = "NONE"

        return {
            "duplicate_row_count": duplicate_count,
            "duplicate_row_ratio": duplicate_ratio,
            "severity": severity,
            "threshold_warning": self.thresholds["duplicate_row_warning_ratio"],
        }

    def _check_target_distribution(
        self, df: pd.DataFrame, target_column: str
    ) -> Dict:
        if target_column not in df.columns:
            return {}

        target_series = df[target_column]
        value_counts = target_series.value_counts().to_dict()
        distribution = {str(k): int(v) for k, v in value_counts.items()}

        return distribution

    def _check_time_parsing(self, df: pd.DataFrame) -> List[Dict]:
        time_parsing = []
        for ts_col in self.timestamp_columns:
            if ts_col not in df.columns:
                continue

            try:
                parsed = pd.to_datetime(df[ts_col], errors="coerce")
                failed_count = int(parsed.isna().sum() - df[ts_col].isna().sum())
                total_count = len(df)
                failed_ratio = failed_count / total_count if total_count > 0 else 0.0

                if failed_ratio > 0:
                    severity = "WARNING"
                else:
                    severity = "NONE"

                time_parsing.append({
                    "column": ts_col,
                    "failed_parse_count": max(0, failed_count),
                    "failed_ratio": max(0.0, failed_ratio),
                    "severity": severity,
                })
            except Exception as e:
                logging.warning(f"Failed to check time parsing for {ts_col}: {e}")

        return time_parsing

    def _check_abnormal_amounts(self, df: pd.DataFrame) -> List[Dict]:
        abnormal_amounts = []
        for amt_col in self.amount_columns:
            if amt_col not in df.columns:
                continue

            try:
                numeric_vals = pd.to_numeric(df[amt_col], errors="coerce")
                negative_count = int((numeric_vals < 0).sum())
                total_count = len(df)
                negative_ratio = negative_count / total_count if total_count > 0 else 0.0

                if negative_count > 0:
                    severity = "WARNING"
                else:
                    severity = "NONE"

                abnormal_amounts.append({
                    "column": amt_col,
                    "negative_count": negative_count,
                    "negative_ratio": negative_ratio,
                    "severity": severity,
                })
            except Exception as e:
                logging.warning(f"Failed to check abnormal amounts for {amt_col}: {e}")

        return abnormal_amounts

    def _collect_risk_items(self, missing_values, duplicates, time_parsing, abnormal_amounts) -> List[Dict]:
        risk_items = []

        for mv in missing_values:
            if mv["severity"] in ("WARNING", "CRITICAL"):
                risk_items.append({
                    "type": "missing_value",
                    "severity": mv["severity"],
                    "column": mv["column"],
                    "message": f"Column '{mv['column']}' has {mv['missing_ratio']:.1%} missing values",
                })

        if duplicates["severity"] == "WARNING":
            risk_items.append({
                "type": "duplicate_rows",
                "severity": "WARNING",
                "message": f"Dataset has {duplicates['duplicate_row_ratio']:.1%} duplicate rows",
            })

        for tp in time_parsing:
            if tp["severity"] == "WARNING":
                risk_items.append({
                    "type": "time_parsing",
                    "severity": "WARNING",
                    "column": tp["column"],
                    "message": f"Column '{tp['column']}' has {tp['failed_parse_count']} failed parses",
                })

        for aa in abnormal_amounts:
            if aa["severity"] == "WARNING":
                risk_items.append({
                    "type": "abnormal_amount",
                    "severity": "WARNING",
                    "column": aa["column"],
                    "message": f"Column '{aa['column']}' has {aa['negative_count']} negative values",
                })

        return risk_items

    def _save_data_quality_report(
        self,
        df: pd.DataFrame,
        missing_values: List[Dict],
        duplicates: Dict,
        abnormal_amounts: List[Dict],
        unknown_categories: List[Dict],
        time_parsing: List[Dict],
        target_distribution: Dict,
        risk_items: List[Dict],
        norm_report: NormalizationReport,
    ) -> str:
        report = {
            "dataset_shape": [df.shape[0], df.shape[1]],
            "missing_values": missing_values,
            "critical_missing_columns": [
                mv["column"] for mv in missing_values if mv["severity"] == "CRITICAL"
            ],
            "duplicates": duplicates,
            "abnormal_amounts": abnormal_amounts,
            "unknown_categories": unknown_categories,
            "time_parsing": time_parsing,
            "target_distribution": target_distribution,
            "risk_items": risk_items,
            "normalization_report": norm_report.to_dict(),
            "validation_config": {
                "critical_feature_columns": self.critical_feature_columns,
                "categorical_columns": norm_report.categorical_columns,
                "categorical_whitelist": self.categorical_whitelist,
                "amount_columns": self.amount_columns,
                "timestamp_columns": self.timestamp_columns,
                "thresholds": self.thresholds,
            },
            "generated_at": pd.Timestamp.now().isoformat(),
        }

        report_path = self.data_validation_config.data_quality_report_path
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        logging.info(f"Data quality report saved to: {report_path}")
        return report_path

    def initiate_data_validation(
        self,
        train_path: str,
        test_path: Optional[str] = None,
    ) -> DataValidationArtifact:
        logging.info("\nEntered the 'data validation' method or component")
        try:
            logging.info(f"Reading training data from: {train_path}")
            train_df = pd.read_csv(train_path)

            target_column_name = self.training_config.features.target_column
            drop_columns = self.training_config.features.drop_columns

            norm_report = NormalizationReport()
            norm_report.original_columns = list(train_df.columns)

            logging.info("Applying unified column name normalization")
            train_df, rename_map = normalize_column_names(train_df)
            norm_report.normalized_columns = list(train_df.columns)
            norm_report.renamed_columns = rename_map

            logging.info("Applying unified temporal feature derivation")
            train_df, derived_cols = derive_temporal_features(train_df)
            norm_report.derived_columns = derived_cols

            logging.info("Applying unified categorical field cleaning")
            train_df, converted_cols = clean_categorical_fields(train_df)

            logging.info("Locating target column using unified logic")
            target_col, target_found = locate_target_column(
                train_df, [target_column_name]
            )
            norm_report.target_column = target_col
            norm_report.target_column_found = target_found

            logging.info("Checking missing required fields using unified logic")
            missing_required, unexpected = report_missing_fields(
                train_df,
                required_columns=self.critical_feature_columns,
            )
            norm_report.missing_required_columns = missing_required
            norm_report.unexpected_columns = unexpected

            if missing_required:
                logging.warning(f"Missing critical feature columns: {missing_required}")

            exclude_for_infer = [target_column_name] if target_column_name else []
            exclude_for_infer.extend(drop_columns)
            numerical_all, categorical_all = infer_feature_types(
                train_df, exclude_columns=exclude_for_infer
            )
            norm_report.numerical_columns = numerical_all
            norm_report.categorical_columns = categorical_all

            logging.info("Checking missing values")
            missing_values = self._check_missing_values(train_df)

            logging.info("Checking duplicate rows")
            duplicates = self._check_duplicates(train_df)

            logging.info("Checking time parsing")
            time_parsing = self._check_time_parsing(train_df)

            logging.info("Checking abnormal amounts")
            abnormal_amounts = self._check_abnormal_amounts(train_df)

            unknown_categories = []

            logging.info("Checking target distribution")
            target_distribution = self._check_target_distribution(train_df, target_col)

            logging.info("Collecting risk items")
            risk_items = self._collect_risk_items(
                missing_values, duplicates, time_parsing, abnormal_amounts
            )

            logging.info("Saving data quality report with normalization metadata")
            report_path = self._save_data_quality_report(
                train_df,
                missing_values,
                duplicates,
                abnormal_amounts,
                unknown_categories,
                time_parsing,
                target_distribution,
                risk_items,
                norm_report,
            )

            input_feature_cols = [
                c for c in train_df.columns
                if c not in drop_columns and c != target_col
            ]
            numerical_features = [
                c for c in input_feature_cols
                if pd.api.types.is_numeric_dtype(train_df[c])
            ]
            categorical_features = [
                c for c in input_feature_cols
                if not pd.api.types.is_numeric_dtype(train_df[c])
            ]

            critical_severities = [
                mv for mv in missing_values
                if mv["severity"] == "CRITICAL"
            ]
            is_valid = len(critical_severities) == 0

            validation_errors = [
                f"Critical missing values in column '{mv['column']}': "
                f"{mv['missing_ratio']:.1%}"
                for mv in missing_values
                if mv["severity"] == "CRITICAL"
            ]
            if missing_required:
                validation_errors.append(
                    f"Missing required columns: {missing_required}"
                )

            validation_warnings = [
                f"{risk['type']}: {risk['message']}"
                for risk in risk_items
                if risk["severity"] == "WARNING"
            ]

            if is_valid:
                logging.info("Data validation PASSED")
            else:
                logging.error(
                    f"Data validation FAILED with {len(validation_errors)} errors"
                )

            if validation_warnings:
                for warn in validation_warnings:
                    logging.warning(f"  - {warn}")

            return DataValidationArtifact(
                validation_status=is_valid,
                data_quality_report_path=os.path.abspath(report_path),
                feature_columns=input_feature_cols,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                target_column=target_col if target_found else "",
                validation_errors=validation_errors,
                validation_warnings=validation_warnings,
                normalization_report=norm_report,
            )

        except Exception as e:
            logging.error("Data validation failed", exc_info=True)
            raise CustomerException(e, sys)
