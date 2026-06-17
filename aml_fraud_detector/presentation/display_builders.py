from __future__ import annotations

from typing import List

import pandas as pd

from aml_fraud_detector.entity.artifact_entity import BatchPredictionResult, PredictionResult


BATCH_DF_COLUMNS: List[str] = [
    "transaction_id",
    "prediction_code",
    "prediction_label",
    "fraud_probability",
    "legit_probability",
    "risk_level",
    "risk_summary",
    "top_factor_1",
    "top_factor_2",
    "top_factor_3",
    "process_status",
]


def batch_predictions_to_dataframe(
    batch_result: BatchPredictionResult,
) -> pd.DataFrame:
    rows = []
    for pr in batch_result.predictions:
        factors = list(pr.top_factors or [])
        rows.append({
            "transaction_id": pr.transaction_id or "",
            "prediction_code": int(pr.prediction),
            "prediction_label": str(pr.class_label),
            "fraud_probability": float(pr.fraud_probability),
            "legit_probability": float(pr.legit_probability),
            "risk_level": pr.risk_level.value if hasattr(pr.risk_level, "value") else str(pr.risk_level),
            "risk_summary": "",
            "top_factor_1": factors[0].get("feature", "") if len(factors) > 0 else "",
            "top_factor_2": factors[1].get("feature", "") if len(factors) > 1 else "",
            "top_factor_3": factors[2].get("feature", "") if len(factors) > 2 else "",
            "process_status": pr.process_status.value if hasattr(pr.process_status, "value") else str(pr.process_status),
        })
    return pd.DataFrame(rows, columns=BATCH_DF_COLUMNS)
