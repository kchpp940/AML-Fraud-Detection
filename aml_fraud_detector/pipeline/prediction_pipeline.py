
import sys
import os
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.utils.risk_explainer import explain_risk, load_feature_metadata


class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.feature_metadata_path = os.path.join("artifacts", "feature_metadata.json")

    def _load_artifacts(self):
        model = load_object(file_path=self.model_path)
        preprocessor = load_object(file_path=self.preprocessor_path)
        return model, preprocessor

    def predict(self, features):
        try:
            model, preprocessor = self._load_artifacts()
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomerException(e, sys)

    def predict_proba(self, features):
        try:
            model, preprocessor = self._load_artifacts()
            data_scaled = preprocessor.transform(features)
            predictions_prob = model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            raise CustomerException(e, sys)

    def predict_with_explanation(self, features, top_n: int = 5):
        try:
            model, preprocessor = self._load_artifacts()

            feature_metadata = None
            if os.path.exists(self.feature_metadata_path):
                try:
                    feature_metadata = load_feature_metadata(self.feature_metadata_path)
                except Exception:
                    logging.warning("Failed to load feature_metadata.json, explanation will be unavailable")

            if feature_metadata is None:
                data_scaled = preprocessor.transform(features)
                prediction = int(model.predict(data_scaled)[0])
                proba = float(model.predict_proba(data_scaled)[0, 1])
                return {
                    "prediction": prediction,
                    "prediction_label": "欺诈交易" if prediction == 1 else "正常交易",
                    "fraud_probability": round(proba, 4),
                    "top_factors": [],
                }

            explanation = explain_risk(
                features_df=features,
                model=model,
                preprocessor=preprocessor,
                feature_metadata=feature_metadata,
                top_n=top_n,
            )
            return explanation
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
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomerException(e, sys)
