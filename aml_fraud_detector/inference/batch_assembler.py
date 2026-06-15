import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.inference.contracts import (
    FRAUD_LABEL,
    LEGIT_LABEL,
    BatchPredictionResult,
    PredictionResult,
    RiskExplanation,
)


class BatchAssembler:
    FRAUD_CLASS_NAME = "Fraud"
    LEGIT_CLASS_NAME = "Not Fraud"

    @classmethod
    def _label_for(cls, pred: int) -> str:
        return cls.FRAUD_CLASS_NAME if int(pred) == FRAUD_LABEL else cls.LEGIT_CLASS_NAME

    def assemble_single(
        self,
        prediction: int,
        proba_row: np.ndarray,
        model_version: str = "unknown",
        transaction_id: Optional[str] = None,
        explanation: Optional[RiskExplanation] = None,
    ) -> PredictionResult:
        try:
            pred_int = int(prediction)
            legit_prob = float(proba_row[0]) if proba_row.ndim == 1 else float(proba_row[0, 0])
            fraud_prob = float(proba_row[1]) if proba_row.ndim == 1 else float(proba_row[0, 1])
            return PredictionResult(
                prediction=pred_int,
                fraud_probability=fraud_prob,
                legit_probability=legit_prob,
                class_label=self._label_for(pred_int),
                explanation=explanation,
                transaction_id=transaction_id,
                model_version=str(model_version),
            )
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)

    def assemble_batch(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        model_version: str = "unknown",
        transaction_ids: Optional[List[str]] = None,
        explanations: Optional[List[Optional[RiskExplanation]]] = None,
    ) -> BatchPredictionResult:
        try:
            preds = np.asarray(predictions).astype(int).flatten()
            proba = np.asarray(probabilities)
            n = len(preds)
            if proba.ndim != 2 or proba.shape[0] != n or proba.shape[1] < 2:
                raise CustomerException(
                    ValueError(
                        f"Probability shape mismatch: predictions={preds.shape}, "
                        f"probabilities={proba.shape}; expected ({n}, 2)"
                    ),
                    sys,
                )
            results: List[PredictionResult] = []
            for i in range(n):
                tx_id = transaction_ids[i] if transaction_ids and i < len(transaction_ids) else None
                expl = explanations[i] if explanations and i < len(explanations) else None
                legit_prob = float(proba[i, 0])
                fraud_prob = float(proba[i, 1])
                pred_int = int(preds[i])
                results.append(
                    PredictionResult(
                        prediction=pred_int,
                        fraud_probability=fraud_prob,
                        legit_probability=legit_prob,
                        class_label=self._label_for(pred_int),
                        explanation=expl,
                        transaction_id=tx_id,
                        model_version=str(model_version),
                    )
                )

            fraud_count = int((preds == FRAUD_LABEL).sum())
            legit_count = n - fraud_count
            fraud_rate = (fraud_count / n) if n > 0 else 0.0
            batch = BatchPredictionResult(
                results=results,
                total_count=n,
                fraud_count=fraud_count,
                legit_count=legit_count,
                fraud_rate=float(fraud_rate),
            )
            logging.info(
                f"BatchAssembler: assembled {n} predictions, "
                f"fraud={fraud_count}, legit={legit_count}, fraud_rate={fraud_rate:.4f}"
            )
            return batch
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)
