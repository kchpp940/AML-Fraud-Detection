import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.inference.contracts import (
    REQUIRED_INPUT_FIELDS,
    TransactionInput,
    ModelArtifacts,
)


class SchemaAligner:
    def __init__(self, artifacts: Optional[ModelArtifacts] = None):
        self._feature_metadata: Dict[str, Any] = {}
        self._encoding_info: Dict[str, Any] = {}
        self._numerical_features: List[str] = []
        self._categorical_features: List[str] = []
        self._baseline_values: Dict[str, Any] = {}
        self._initialized = False
        if artifacts is not None:
            self.bind(artifacts)

    def bind(self, artifacts: ModelArtifacts) -> None:
        self._feature_metadata = artifacts.feature_metadata or {}
        self._encoding_info = self._feature_metadata.get("encoding_info", {})
        self._numerical_features = list(
            self._feature_metadata.get("numerical_features", [])
        )
        self._categorical_features = list(
            self._feature_metadata.get("categorical_features", [])
        )
        self._baseline_values = self._feature_metadata.get("baseline_values", {})
        self._initialized = True
        logging.info(
            "SchemaAligner bound: "
            f"numerical={self._numerical_features}, "
            f"categorical={self._categorical_features}"
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise CustomerException(
                RuntimeError("SchemaAligner is not bound to artifacts; call bind() first"),
                sys,
            )

    @staticmethod
    def _coerce_types(df: pd.DataFrame, feature_metadata: Dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        numerical_features = feature_metadata.get("numerical_features", [])
        categorical_features = feature_metadata.get("categorical_features", [])
        training_stats = feature_metadata.get("training_stats", {})

        for col in numerical_features:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                except Exception as e:
                    logging.warning(f"Failed to coerce {col} to numeric: {e}")
                stats = training_stats.get(col, {})
                fill_val = stats.get("median") or stats.get("mean") or 0.0
                df[col] = df[col].fillna(fill_val)

        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].astype(str).replace({"nan": None, "None": None, "": None})
                baseline = training_stats.get(col, {}).get("mode")
                if baseline is not None:
                    df[col] = df[col].fillna(str(baseline))
                else:
                    df[col] = df[col].fillna("Unknown")

        return df

    def _align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_cols = self._numerical_features + self._categorical_features
        missing = [c for c in expected_cols if c not in df.columns]
        extra = [c for c in df.columns if c not in expected_cols]

        if missing:
            logging.warning(
                f"Input missing expected feature columns: {missing}. "
                f"Filling with baseline values from feature_metadata."
            )
            for col in missing:
                baseline = self._baseline_values.get(col)
                if baseline is None:
                    if col in self._numerical_features:
                        stats = self._feature_metadata.get("training_stats", {}).get(col, {})
                        baseline = stats.get("median") or stats.get("mean") or 0.0
                    else:
                        stats = self._feature_metadata.get("training_stats", {}).get(col, {})
                        baseline = stats.get("mode") or "Unknown"
                df[col] = baseline

        if extra:
            logging.info(
                f"Input has extra columns not in feature schema: {extra}. "
                f"These will be dropped before preprocessing."
            )

        ordered = [c for c in expected_cols if c in df.columns]
        return df[ordered]

    def validate_required_fields(self, raw_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        for field in REQUIRED_INPUT_FIELDS:
            if field not in raw_data:
                errors.append(f"Missing required input field: '{field}'")
        return (len(errors) == 0, errors)

    def align_single(self, transaction: TransactionInput) -> pd.DataFrame:
        self._ensure_initialized()
        try:
            raw_df = transaction.to_dataframe()
            return self.align_dataframe(raw_df)
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)

    def align_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_initialized()
        try:
            original_cols = list(df.columns)
            if "from_bank" in df.columns:
                df["from_bank"] = df["from_bank"].astype("object")
            if "to_bank" in df.columns:
                df["to_bank"] = df["to_bank"].astype("object")
            df = self._align_columns(df)
            df = self._coerce_types(df, self._feature_metadata)
            logging.info(
                f"SchemaAligner: input cols={original_cols}, "
                f"output cols={list(df.columns)}, shape={df.shape}"
            )
            return df
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)

    @property
    def expected_columns(self) -> List[str]:
        return list(self._numerical_features) + list(self._categorical_features)

    @property
    def numerical_features(self) -> List[str]:
        return list(self._numerical_features)

    @property
    def categorical_features(self) -> List[str]:
        return list(self._categorical_features)
