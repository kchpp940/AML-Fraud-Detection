import os
import sys
import pandas as pd
from typing import Optional, List, Tuple
from dataclasses import dataclass

from aml_fraud_detector.logger import logging
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.constants import (
    SCHEMA_REQUIRED_COLUMNS_KEY,
    SCHEMA_TARGET_COLUMN_KEY,
)
from aml_fraud_detector.entity.config_entity import DataValidationConfig
from aml_fraud_detector.configuration import Configuration


@dataclass
class DataValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    dataframe_info: Optional[dict] = None


class DataValidation:
    def __init__(
        self,
        config: Optional[DataValidationConfig] = None,
        schema: Optional[dict] = None,
    ):
        self.validation_config = config or DataValidationConfig()
        self._config_manager = Configuration()
        self.schema = schema or self._config_manager.get_schema()
        self.required_columns = self.schema.get(SCHEMA_REQUIRED_COLUMNS_KEY, [])
        self.target_column = self.schema.get(SCHEMA_TARGET_COLUMN_KEY)

    def validate_file_exists(self, file_path: str) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []
        if not file_path:
            errors.append("Data file path is empty or None")
            return False, errors, warnings
        if not os.path.exists(file_path):
            errors.append(f"Data file does not exist: {file_path}")
            return False, errors, warnings
        if not os.path.isfile(file_path):
            errors.append(f"Path is not a regular file: {file_path}")
            return False, errors, warnings
        file_size = os.path.getsize(file_path)
        logging.info(f"Data file size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        if file_size == 0:
            errors.append(f"Data file is empty (0 bytes): {file_path}")
            return False, errors, warnings
        if file_size < 100:
            warnings.append(f"Data file is very small ({file_size} bytes): {file_path}")
        return True, errors, warnings

    def validate_dataframe_rows(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []
        row_count = len(df)
        logging.info(f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns")
        if row_count == 0:
            errors.append("DataFrame has 0 rows after reading CSV")
            return False, errors, warnings
        if row_count < 10:
            warnings.append(f"DataFrame has very few rows: {row_count}")
        return True, errors, warnings

    def validate_required_columns(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []
        actual_columns = list(df.columns)
        logging.info(f"Actual columns in DataFrame ({len(actual_columns)}): {actual_columns}")
        logging.info(f"Required columns ({len(self.required_columns)}): {self.required_columns}")
        if not self.required_columns:
            errors.append("No required columns defined in schema")
            return False, errors, warnings
        missing_columns = [col for col in self.required_columns if col not in actual_columns]
        if missing_columns:
            errors.append(
                f"Missing {len(missing_columns)} required column(s): {missing_columns}. "
                f"DataFrame has columns: {actual_columns}"
            )
            return False, errors, warnings
        extra_columns = [col for col in actual_columns if col not in self.required_columns]
        if extra_columns:
            warnings.append(f"DataFrame contains extra columns not in schema: {extra_columns}")
        return True, errors, warnings

    def validate_target_column(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []
        if not self.target_column:
            errors.append("Target column is not defined in schema")
            return False, errors, warnings
        if self.target_column not in df.columns:
            errors.append(
                f"Target column '{self.target_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
            return False, errors, warnings
        target_series = df[self.target_column]
        null_count = target_series.isnull().sum()
        logging.info(
            f"Target column '{self.target_column}': {target_series.nunique()} unique values, "
            f"{null_count} null values out of {len(target_series)} rows"
        )
        if null_count > 0:
            null_pct = (null_count / len(target_series)) * 100
            msg = f"Target column '{self.target_column}' contains {null_count} null values ({null_pct:.2f}%)"
            if null_pct > 50:
                errors.append(msg + " — too many nulls, cannot proceed")
                return False, errors, warnings
            else:
                warnings.append(msg)
        value_counts = target_series.value_counts().to_dict()
        logging.info(f"Target column distribution: {value_counts}")
        if len(value_counts) < 2:
            warnings.append(f"Target column has only {len(value_counts)} unique value(s): {value_counts}")
        return True, errors, warnings

    def validate_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        errors = []
        warnings = []
        dtypes_info = df.dtypes.astype(str).to_dict()
        logging.info(f"DataFrame dtypes: {dtypes_info}")
        return True, errors, warnings

    def validate_source_data(self, file_path: str, df: Optional[pd.DataFrame] = None) -> DataValidationResult:
        logging.info("=" * 60)
        logging.info(f"Starting data validation for: {file_path}")
        logging.info("=" * 60)
        all_errors: List[str] = []
        all_warnings: List[str] = []

        valid, errs, warns = self.validate_file_exists(file_path)
        all_errors.extend(errs)
        all_warnings.extend(warns)
        if not valid:
            for err in all_errors:
                logging.error(f"[VALIDATION ERROR] {err}")
            for warn in all_warnings:
                logging.warning(f"[VALIDATION WARNING] {warn}")
            return DataValidationResult(
                is_valid=False,
                errors=all_errors,
                warnings=all_warnings,
            )

        if df is None:
            try:
                logging.info(f"Reading CSV for validation: {file_path}")
                df = pd.read_csv(file_path)
                logging.info("CSV read successfully")
            except Exception as e:
                err_msg = f"Failed to read CSV file {file_path}: {str(e)}"
                all_errors.append(err_msg)
                logging.error(f"[VALIDATION ERROR] {err_msg}")
                return DataValidationResult(
                    is_valid=False,
                    errors=all_errors,
                    warnings=all_warnings,
                )

        dataframe_info = {
            "file_path": file_path,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        valid, errs, warns = self.validate_dataframe_rows(df)
        all_errors.extend(errs)
        all_warnings.extend(warns)
        if not valid:
            for err in all_errors:
                logging.error(f"[VALIDATION ERROR] {err}")
            for warn in all_warnings:
                logging.warning(f"[VALIDATION WARNING] {warn}")
            return DataValidationResult(
                is_valid=False,
                errors=all_errors,
                warnings=all_warnings,
                dataframe_info=dataframe_info,
            )

        valid, errs, warns = self.validate_required_columns(df)
        all_errors.extend(errs)
        all_warnings.extend(warns)
        if not valid:
            for err in all_errors:
                logging.error(f"[VALIDATION ERROR] {err}")
            for warn in all_warnings:
                logging.warning(f"[VALIDATION WARNING] {warn}")
            return DataValidationResult(
                is_valid=False,
                errors=all_errors,
                warnings=all_warnings,
                dataframe_info=dataframe_info,
            )

        valid, errs, warns = self.validate_target_column(df)
        all_errors.extend(errs)
        all_warnings.extend(warns)
        if not valid:
            for err in all_errors:
                logging.error(f"[VALIDATION ERROR] {err}")
            for warn in all_warnings:
                logging.warning(f"[VALIDATION WARNING] {warn}")
            return DataValidationResult(
                is_valid=False,
                errors=all_errors,
                warnings=all_warnings,
                dataframe_info=dataframe_info,
            )

        _, errs, warns = self.validate_data_types(df)
        all_errors.extend(errs)
        all_warnings.extend(warns)

        is_valid = len(all_errors) == 0
        for err in all_errors:
            logging.error(f"[VALIDATION ERROR] {err}")
        for warn in all_warnings:
            logging.warning(f"[VALIDATION WARNING] {warn}")

        if is_valid:
            logging.info("=" * 60)
            logging.info(f"Data validation PASSED for: {file_path}")
            logging.info("=" * 60)
        else:
            logging.error("=" * 60)
            logging.error(f"Data validation FAILED for: {file_path}")
            logging.error("=" * 60)

        return DataValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            dataframe_info=dataframe_info,
        )

    def initiate_data_validation(self, source_data_path: str) -> DataValidationResult:
        logging.info("Entered the 'data validation' method or component")
        try:
            result = self.validate_source_data(source_data_path)
            if not result.is_valid:
                error_summary = "; ".join(result.errors)
                raise CustomerException(
                    ValueError(f"Data validation failed: {error_summary}"),
                    sys,
                )
            return result
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)
