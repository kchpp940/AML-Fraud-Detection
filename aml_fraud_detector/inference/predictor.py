import sys
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.inference.contracts import FRAUD_LABEL, LEGIT_LABEL, ModelArtifacts


class Predictor:
    def __init__(self, artifacts: Optional[ModelArtifacts] = None):
        self._model: Any = None
        self._preprocessor: Any = None
        self._model_version: str = "unknown"
        self._model_name: str = "unknown"
        self._initialized = False
        if artifacts is not None:
            self.bind(artifacts)

    def bind(self, artifacts: ModelArtifacts) -> None:
        self._model = artifacts.model
        self._preprocessor = artifacts.preprocessor
        self._model_version = str(artifacts.model_metadata.get("model_version", "unknown"))
        self._model_name = str(artifacts.model_metadata.get("best_model_name", "unknown"))
        if self._model is None or self._preprocessor is None:
            raise CustomerException(
                RuntimeError("ModelArtifacts does not contain both model and preprocessor"),
                sys,
            )
        self._initialized = True
        logging.info(
            "Predictor bound: "
            f"model={type(self._model).__name__}, "
            f"preprocessor={type(self._preprocessor).__name__}, "
            f"version={self._model_version}"
        )

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise CustomerException(
                RuntimeError("Predictor is not bound to artifacts; call bind() first"),
                sys,
            )

    @staticmethod
    def _to_numpy(matrix: Any) -> np.ndarray:
        if hasattr(matrix, "toarray"):
            return matrix.toarray()
        return np.asarray(matrix)

    def _transform(self, aligned_df: pd.DataFrame) -> np.ndarray:
        try:
            transformed = self._preprocessor.transform(aligned_df)
            transformed_np = self._to_numpy(transformed)
            logging.info(f"Preprocessor transform: input_shape={aligned_df.shape}, output_shape={transformed_np.shape}")
            return transformed_np
        except Exception as e:
            raise CustomerException(
                RuntimeError(f"Preprocessor transform failed: {e}"), sys
            )

    def predict(self, aligned_df: pd.DataFrame) -> np.ndarray:
        self._ensure_initialized()
        try:
            X = self._transform(aligned_df)
            preds = self._model.predict(X)
            preds = np.asarray(preds).astype(int).flatten()
            logging.info(f"Prediction produced {len(preds)} label(s)")
            return preds
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)

    def predict_proba(self, aligned_df: pd.DataFrame) -> np.ndarray:
        self._ensure_initialized()
        try:
            X = self._transform(aligned_df)
            proba = self._model.predict_proba(X)
            proba = self._to_numpy(proba)
            if proba.ndim != 2 or proba.shape[1] < 2:
                raise CustomerException(
                    RuntimeError(f"Unexpected predict_proba output shape: {proba.shape}"),
                    sys,
                )
            logging.info(f"Probability produced shape={proba.shape}")
            return proba
        except CustomerException:
            raise
        except AttributeError:
            raise CustomerException(
                NotImplementedError("Loaded model does not support predict_proba"),
                sys,
            )
        except Exception as e:
            raise CustomerException(e, sys)

    def predict_with_proba(
        self, aligned_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_initialized()
        try:
            X = self._transform(aligned_df)
            preds = np.asarray(self._model.predict(X)).astype(int).flatten()
            proba = self._to_numpy(self._model.predict_proba(X))
            if proba.ndim != 2 or proba.shape[1] < 2:
                raise CustomerException(
                    RuntimeError(f"Unexpected predict_proba output shape: {proba.shape}"),
                    sys,
                )
            logging.info(
                f"Prediction complete: n={len(preds)}, fraud_count={int((preds == FRAUD_LABEL).sum())}"
            )
            return preds, proba
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)

    def get_preprocessed_features(self, aligned_df: pd.DataFrame) -> np.ndarray:
        self._ensure_initialized()
        return self._transform(aligned_df)

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def model_name(self) -> str:
        return self._model_name

    def class_labels(self) -> List[int]:
        if hasattr(self._model, "classes_"):
            return [int(c) for c in list(self._model.classes_)]
        return [LEGIT_LABEL, FRAUD_LABEL]
