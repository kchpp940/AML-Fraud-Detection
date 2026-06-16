
import sys
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.configuration import TrainingConfig
from aml_fraud_detector.utils import (
    normalize_dataframe,
    report_missing_fields,
    locate_target_column,
    NormalizationReport,
)


@dataclass
class DataValidationConfig:
    data_quality_report_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    data_quality_report_path: str
    normalization_report: NormalizationReport
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)


class DataValidation:
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config or TrainingConfig()
        self._resolved = self.training_config.to_resolved_dict()
        tc = self.training_config
        self.data_validation_config = DataValidationConfig(
            data_quality_report_path=tc.artifacts_subpath("data_quality_report.json")
        )
        logging.info(
            f"DataValidation initialized with resolved config: "
            f"target={self._resolved['features']['target_column']}, "
            f"drop={self._resolved['features']['drop_columns']}"
        )

    def _validate_schema(
        self,
        df: pd.DataFrame,
        norm_report: NormalizationReport,
    ) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        warnings: List[str] = []

        target_column_name = self.training_config.features.target_column
        if not norm_report.target_column_found:
            errors.append(
                f"Target column '{target_column_name}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        if norm_report.missing_required_columns:
            errors.append(
                f"Missing required columns: {norm_report.missing_required_columns}"
            )

        expected_columns = self._get_expected_columns()
        if expected_columns:
            missing, unexpected = report_missing_fields(
                df, expected_columns=expected_columns
            )
            if unexpected:
                warnings.append(
                    f"Unexpected columns in dataset: {unexpected}"
                )

        return errors, warnings

    def _validate_data_types(
        self,
        df: pd.DataFrame,
        norm_report: NormalizationReport,
    ) -> Tuple[List[str], List[str]]:
        errors: List[str] = []
        warnings: List[str] = []

        for col in norm_report.numerical_columns:
            if col in df.columns:
                non_numeric = pd.to_numeric(df[col], errors="coerce").isna()
                non_numeric_count = non_numeric.sum()
                if non_numeric_count > 0:
                    warnings.append(
                        f"Column '{col}' has {non_numeric_count} non-numeric values"
                    )

        for col in norm_report.categorical_columns:
            if col in df.columns:
                unique_count = df[col].nunique()
                if unique_count > 1000:
                    warnings.append(
                        f"Column '{col}' has high cardinality: {unique_count} unique values"
                    )

        return errors, warnings

    def _validate_missing_values(
        self,
        df: pd.DataFrame,
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        errors: List[str] = []
        warnings: List[str] = []
        missing_ratios: Dict[str, float] = {}

        total_rows = len(df)
        if total_rows == 0:
            errors.append("Dataset is empty")
            return errors, warnings, missing_ratios

        for col in df.columns:
            missing_count = df[col].isna().sum()
            missing_ratio = missing_count / total_rows
            missing_ratios[col] = missing_ratio

            if missing_ratio > 0.5:
                errors.append(
                    f"Column '{col}' has {missing_ratio:.1%} missing values (exceeds 50% threshold)"
                )
            elif missing_ratio > 0.1:
                warnings.append(
                    f"Column '{col}' has {missing_ratio:.1%} missing values (exceeds 10% threshold)"
                )

        return errors, warnings, missing_ratios

    def _validate_target_distribution(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> Tuple[List[str], List[str], Dict[str, int]]:
        errors: List[str] = []
        warnings: List[str] = []
        class_counts: Dict[str, int] = {}

        if target_column not in df.columns:
            return errors, warnings, class_counts

        target_series = df[target_column]
        class_counts = target_series.value_counts().to_dict()

        if len(class_counts) < 2:
            errors.append(
                f"Target column '{target_column}' has only {len(class_counts)} class(es): {class_counts}"
            )
        else:
            min_class = min(class_counts.values())
            max_class = max(class_counts.values())
            imbalance_ratio = max_class / min_class if min_class > 0 else float("inf")

            if imbalance_ratio > 10:
                warnings.append(
                    f"Target column '{target_column}' is highly imbalanced: "
                    f"imbalance ratio = {imbalance_ratio:.1f}:1, class counts = {class_counts}"
                )

        return errors, warnings, class_counts

    def _get_expected_columns(self) -> Optional[List[str]]:
        try:
            feature_metadata_path = self.training_config.artifacts_subpath(
                "feature_metadata.json"
            )
            if os.path.exists(feature_metadata_path):
                with open(feature_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                return metadata.get("original_features")
        except Exception as e:
            logging.warning(f"Failed to load feature metadata: {e}")
        return None

    def _save_data_quality_report(
        self,
        df: pd.DataFrame,
        norm_report: NormalizationReport,
        errors: List[str],
        warnings: List[str],
        missing_ratios: Dict[str, float],
        class_counts: Dict[str, int],
    ) -> str:
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
            "normalization_report": norm_report.to_dict(),
            "missing_value_ratios": missing_ratios,
            "target_class_distribution": class_counts,
            "validation_errors": errors,
            "validation_warnings": warnings,
            "column_types": {
                col: str(dtype) for col, dtype in df.dtypes.items()
            },
            "basic_statistics": {
                col: {
                    "count": int(df[col].count()),
                    "unique": int(df[col].nunique()),
                }
                for col in df.columns
            },
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
            all_errors: List[str] = []
            all_warnings: List[str] = []

            logging.info(f"Reading training data from: {train_path}")
            train_df = pd.read_csv(train_path)

            target_column_name = self.training_config.features.target_column
            drop_columns = self.training_config.features.drop_columns

            logging.info("Applying unified feature normalization to training data")
            train_df_normalized, norm_report = normalize_dataframe(
                train_df,
                target_column_name=target_column_name,
                drop_columns=drop_columns,
                derive_temporal=True,
                clean_categorical=True,
            )

            logging.info("Validating schema integrity")
            schema_errors, schema_warnings = self._validate_schema(
                train_df_normalized, norm_report
            )
            all_errors.extend(schema_errors)
            all_warnings.extend(schema_warnings)

            logging.info("Validating data types")
            dtype_errors, dtype_warnings = self._validate_data_types(
                train_df_normalized, norm_report
            )
            all_errors.extend(dtype_errors)
            all_warnings.extend(dtype_warnings)

            logging.info("Validating missing values")
            missing_errors, missing_warnings, missing_ratios = (
                self._validate_missing_values(train_df_normalized)
            )
            all_errors.extend(missing_errors)
            all_warnings.extend(missing_warnings)

            logging.info("Validating target column distribution")
            target_errors, target_warnings, class_counts = (
                self._validate_target_distribution(
                    train_df_normalized, norm_report.target_column
                )
            )
            all_errors.extend(target_errors)
            all_warnings.extend(target_warnings)

            if test_path:
                logging.info(f"Reading test data from: {test_path}")
                test_df = pd.read_csv(test_path)

                logging.info("Applying unified feature normalization to test data")
                test_df_normalized, test_norm_report = normalize_dataframe(
                    test_df,
                    target_column_name=target_column_name,
                    drop_columns=drop_columns,
                    derive_temporal=True,
                    clean_categorical=True,
                )

                logging.info("Validating train/test feature consistency")
                train_features = set(norm_report.numerical_columns + norm_report.categorical_columns)
                test_features = set(test_norm_report.numerical_columns + test_norm_report.categorical_columns)

                missing_in_test = train_features - test_features
                extra_in_test = test_features - train_features

                if missing_in_test:
                    all_errors.append(
                        f"Test data missing features present in train: {sorted(missing_in_test)}"
                    )
                if extra_in_test:
                    all_warnings.append(
                        f"Test data has extra features not in train: {sorted(extra_in_test)}"
                    )

            is_valid = len(all_errors) == 0

            report_path = self._save_data_quality_report(
                train_df_normalized,
                norm_report,
                all_errors,
                all_warnings,
                missing_ratios,
                class_counts,
            )

            if is_valid:
                logging.info("Data validation PASSED")
            else:
                logging.error(f"Data validation FAILED with {len(all_errors)} errors")
                for err in all_errors:
                    logging.error(f"  - {err}")

            for warn in all_warnings:
                logging.warning(f"  - {warn}")

            return DataValidationArtifact(
                validation_status=is_valid,
                data_quality_report_path=os.path.abspath(report_path),
                normalization_report=norm_report,
                validation_errors=all_errors,
                validation_warnings=all_warnings,
            )

        except Exception as e:
            logging.error("Data validation failed", exc_info=True)
            raise CustomerException(e, sys)
