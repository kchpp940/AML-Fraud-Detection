
import sys
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


CATEGORICAL_FORCE_OBJECT_COLS: List[str] = ["from_bank", "to_bank"]

TIMESTAMP_COLUMN_NAMES: List[str] = ["timestamp"]


@dataclass
class NormalizationReport:
    original_columns: List[str] = field(default_factory=list)
    normalized_columns: List[str] = field(default_factory=list)
    renamed_columns: Dict[str, str] = field(default_factory=dict)
    derived_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    target_column_found: bool = False
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    missing_required_columns: List[str] = field(default_factory=list)
    unexpected_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "original_columns": self.original_columns,
            "normalized_columns": self.normalized_columns,
            "renamed_columns": self.renamed_columns,
            "derived_columns": self.derived_columns,
            "target_column": self.target_column,
            "target_column_found": self.target_column_found,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "missing_required_columns": self.missing_required_columns,
            "unexpected_columns": self.unexpected_columns,
            "warnings": self.warnings,
        }


def normalize_column_name(col: str) -> str:
    if not isinstance(col, str):
        col = str(col)
    normalized = col.lower()
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace(".", "_")
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized


def normalize_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    rename_map: Dict[str, str] = {}
    new_columns: List[str] = []
    for col in df.columns:
        normalized = normalize_column_name(col)
        if normalized != col:
            rename_map[col] = normalized
        new_columns.append(normalized)

    if len(set(new_columns)) != len(new_columns):
        seen: Dict[str, int] = {}
        deduped: List[str] = []
        for col in new_columns:
            if col in seen:
                seen[col] += 1
                deduped_col = f"{col}_{seen[col]}"
                logging.warning(
                    f"Duplicate column name '{col}' after normalization, "
                    f"renamed to '{deduped_col}'"
                )
                deduped.append(deduped_col)
            else:
                seen[col] = 0
                deduped.append(col)
        new_columns = deduped

    df = df.copy()
    df.columns = new_columns
    return df, rename_map


def derive_temporal_features(
    df: pd.DataFrame,
    timestamp_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if timestamp_cols is None:
        timestamp_cols = TIMESTAMP_COLUMN_NAMES

    df = df.copy()
    derived: List[str] = []

    for ts_col in timestamp_cols:
        if ts_col not in df.columns:
            continue

        try:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        except Exception as e:
            logging.warning(f"Failed to parse '{ts_col}' as datetime: {e}")
            continue

        date_col = "date"
        day_col = "day"
        time_col = "time"

        if date_col not in df.columns:
            df[date_col] = df[ts_col].dt.date
            derived.append(date_col)
            logging.info(f"Derived column '{date_col}' from '{ts_col}'")

        if day_col not in df.columns:
            df[day_col] = df[ts_col].dt.day_name()
            derived.append(day_col)
            logging.info(f"Derived column '{day_col}' from '{ts_col}'")

        if time_col not in df.columns:
            df[time_col] = df[ts_col].dt.time
            derived.append(time_col)
            logging.info(f"Derived column '{time_col}' from '{ts_col}'")

    return df, derived


def locate_target_column(
    df: pd.DataFrame,
    target_candidates: List[str],
) -> Tuple[Optional[str], bool]:
    normalized_candidates = [normalize_column_name(c) for c in target_candidates]
    for col in df.columns:
        normalized_col = normalize_column_name(col)
        if normalized_col in normalized_candidates:
            return col, True
    return None, False


def clean_categorical_fields(
    df: pd.DataFrame,
    force_object_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if force_object_cols is None:
        force_object_cols = CATEGORICAL_FORCE_OBJECT_COLS

    df = df.copy()
    converted: List[str] = []

    for col in force_object_cols:
        if col in df.columns:
            if df[col].dtype != object:
                df[col] = df[col].astype(object)
                converted.append(col)
                logging.info(
                    f"Converted column '{col}' from {df[col].dtype} to object"
                )

    return df, converted


def report_missing_fields(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    expected_columns: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    unexpected: List[str] = []

    if required_columns:
        normalized_required = {normalize_column_name(c) for c in required_columns}
        normalized_df_cols = {normalize_column_name(c) for c in df.columns}
        missing = sorted(normalized_required - normalized_df_cols)
        if missing:
            logging.warning(f"Missing required columns: {missing}")

    if expected_columns:
        normalized_expected = {normalize_column_name(c) for c in expected_columns}
        normalized_df_cols = {normalize_column_name(c) for c in df.columns}
        unexpected = sorted(normalized_df_cols - normalized_expected)
        if unexpected:
            logging.info(f"Unexpected columns present: {unexpected}")

    return missing, unexpected


def infer_feature_types(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    if exclude_columns is None:
        exclude_columns = []

    normalized_exclude = {normalize_column_name(c) for c in exclude_columns}
    cols_to_analyze = [
        c for c in df.columns
        if normalize_column_name(c) not in normalized_exclude
    ]

    df_subset = df[cols_to_analyze]

    numerical = df_subset.select_dtypes(include=np.number).columns.tolist()
    categorical = df_subset.select_dtypes(include=object).columns.tolist()

    return numerical, categorical


def normalize_dataframe(
    df: pd.DataFrame,
    target_column_name: Optional[str] = None,
    drop_columns: Optional[List[str]] = None,
    required_columns: Optional[List[str]] = None,
    expected_columns: Optional[List[str]] = None,
    derive_temporal: bool = True,
    clean_categorical: bool = True,
) -> Tuple[pd.DataFrame, NormalizationReport]:
    try:
        report = NormalizationReport()
        report.original_columns = list(df.columns)
        logging.info(
            f"Starting dataframe normalization: {len(df.columns)} columns, {len(df)} rows"
        )

        df, rename_map = normalize_column_names(df)
        report.normalized_columns = list(df.columns)
        report.renamed_columns = rename_map
        if rename_map:
            logging.info(f"Renamed {len(rename_map)} columns")

        if derive_temporal:
            df, derived = derive_temporal_features(df)
            report.derived_columns = derived

        if clean_categorical:
            df, converted = clean_categorical_fields(df)
            if converted:
                report.warnings.append(
                    f"Converted {len(converted)} columns to object type: {converted}"
                )

        if target_column_name:
            target_col, found = locate_target_column(
                df, [target_column_name]
            )
            report.target_column = target_col
            report.target_column_found = found
            if not found:
                report.warnings.append(
                    f"Target column '{target_column_name}' not found in dataframe"
                )

        if required_columns or expected_columns:
            missing, unexpected = report_missing_fields(
                df, required_columns, expected_columns
            )
            report.missing_required_columns = missing
            report.unexpected_columns = unexpected
            if missing:
                report.warnings.append(
                    f"Missing {len(missing)} required columns: {missing}"
                )

        exclude_for_type_infer = []
        if target_column_name:
            exclude_for_type_infer.append(target_column_name)
        if drop_columns:
            normalized_drop = [normalize_column_name(c) for c in drop_columns]
            exclude_for_type_infer.extend(normalized_drop)

        numerical, categorical = infer_feature_types(df, exclude_for_type_infer)
        report.numerical_columns = numerical
        report.categorical_columns = categorical

        logging.info(
            f"Normalization complete: {len(report.numerical_columns)} numerical, "
            f"{len(report.categorical_columns)} categorical columns"
        )

        return df, report

    except Exception as e:
        logging.error(f"Error during dataframe normalization: {e}", exc_info=True)
        raise CustomerException(e, sys)


def get_prediction_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    derive_temporal: bool = True,
    clean_categorical: bool = True,
) -> Tuple[pd.DataFrame, NormalizationReport]:
    try:
        report = NormalizationReport()
        report.original_columns = list(df.columns)

        df, rename_map = normalize_column_names(df)
        report.normalized_columns = list(df.columns)
        report.renamed_columns = rename_map

        if derive_temporal:
            df, derived = derive_temporal_features(df)
            report.derived_columns = derived

        if clean_categorical:
            df, converted = clean_categorical_fields(df)

        normalized_feature_cols = [normalize_column_name(c) for c in feature_columns]
        missing_features = [
            c for c in normalized_feature_cols if c not in df.columns
        ]
        if missing_features:
            raise CustomerException(
                ValueError(
                    f"Missing required prediction features: {missing_features}. "
                    f"Available columns: {list(df.columns)}"
                ),
                sys,
            )

        df = df[normalized_feature_cols]

        for col in normalized_feature_cols:
            if col not in report.numerical_columns and col not in report.categorical_columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    report.numerical_columns.append(col)
                else:
                    report.categorical_columns.append(col)

        return df, report

    except Exception as e:
        logging.error(f"Error preparing prediction features: {e}", exc_info=True)
        raise CustomerException(e, sys)
