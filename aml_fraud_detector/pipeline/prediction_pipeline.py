    
import sys
import os
from typing import Optional

import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.utils.feature_normalization import (
    normalize_column_names,
    derive_temporal_features,
    clean_categorical_fields,
)


class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self._model = None
        self._preprocessor = None

    def _load_artifacts(self):
        if self._model is None:
            self._model = load_object(file_path=self.model_path)
        if self._preprocessor is None:
            self._preprocessor = load_object(file_path=self.preprocessor_path)

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()

        df, _ = normalize_column_names(df)

        if "timestamp" in df.columns:
            df, _ = derive_temporal_features(df)

        df, _ = clean_categorical_fields(df)

        return df

    def predict(self, features):
        try:
            if self._model is None or self._preprocessor is None:
                self._load_artifacts()

            normalized_features = self._prepare_features(features)
            data_scaled = self._preprocessor.transform(normalized_features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            predictions = self._model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomerException(e, sys)
        
    def predict_proba(self, features):
        try:
            if self._model is None or self._preprocessor is None:
                self._load_artifacts()

            normalized_features = self._prepare_features(features)
            data_scaled = self._preprocessor.transform(normalized_features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            predictions_prob = self._model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            raise CustomerException(e, sys)


class CustomData:
    def __init__(self,
            from_bank: int,
            account: str,
            to_bank: int,
            account_1: str,
            amount_received: float,
            receiving_currency: str,
            payment_currency: str,
            payment_format: str,
            day: str):
    
        self.from_bank = from_bank
        self.account = account
        self.to_bank = to_bank
        self.account_1 = account_1
        self.amount_received =  amount_received
        self.receiving_currency = receiving_currency
        self.payment_currency = payment_currency
        self.payment_format = payment_format
        self.day =  day
    
    def get_data_as_DataFrame(self):
        try:
            custom_data_input_dict = {
                "from_bank": [self.from_bank],
                "account": [self.account],
                "to_bank": [self.to_bank],
                "account_1": [self.account_1],
                "amount_received": [self.amount_received],
                "receiving_currency": [self.receiving_currency],
                "payment_currency": [self.payment_currency],
                "payment_format": [self.payment_format],
                "day": [self.day]
            }
            df = pd.DataFrame(custom_data_input_dict)

            df, _ = clean_categorical_fields(df)

            return df

        except Exception as e:
            raise CustomerException(e, sys)
