"""
Legacy shim module for PredictionPipeline / CustomData.

The actual implementation has moved to :mod:`aml_fraud_detector.inference`.
This module re-exports the symbols from the new package to preserve
backwards compatibility with existing callers such as the Flask/Streamlit UIs
and any external code that imports from this location.

**Do NOT add new inference logic here.**  Implement features in the dedicated
services under :mod:`aml_fraud_detector.inference` and have
:class:`aml_fraud_detector.inference.PredictionPipeline` orchestrate them.
"""

import warnings

from aml_fraud_detector.inference import (
    PredictionPipeline,
    TransactionInput,
)
from aml_fraud_detector.inference.pipeline import CustomData


__all__ = [
    "PredictionPipeline",
    "CustomData",
    "TransactionInput",
]
