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
        self._numerical_features: List[str] = []
        self._categorical_features: List[str] = []
        self._initialized = False
        if artifacts is not None:
            self.bind(artifacts)

    def bind(self, artifacts: ModelArtifacts) -> None:
        self._feature_metadata = artifacts.feature_metadata or {}
        self._numerical_features = list(
            self._feature_metadata.get("numerical_features", [])
        )
        self._categorical_features = list(
            self._feature_metadata.get("categorical_features", [])
        )
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
            aligned = df.copy()
            if "from_bank" in aligned.columns:
                aligned["from_bank"] = aligned["from_bank"].astype("object")
            if "to_bank" in aligned.columns:
                aligned["to_bank"] = aligned["to_bank"].astype("object")
            logging.info(
                f"SchemaAligner: input cols={original_cols}, "
                f"output cols={list(aligned.columns)}, shape={aligned.shape}"
            )
            return aligned
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
