    
import sys
import os
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object, validate_artifacts


class PredictionPipeline:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        validation = validate_artifacts(artifacts_dir)
        self.validation_ok: bool = validation["ok"]
        self.validation_warnings = validation["warnings"]
        self.validation_errors = validation["errors"]
        self.model_metadata = validation["metadata"]
        self.artifact_manifest = validation["manifest"]
        if not self.validation_ok:
            logging.warning(
                f"PredictionPipeline artifact validation FAILED: "
                f"errors={self.validation_errors}, warnings={self.validation_warnings}"
            )
        else:
            logging.info(
                "PredictionPipeline artifact validation passed, "
                f"model version={self.model_metadata.get('model_version')}"
            )

    def predict(self, features):
        try: 
            model_path = os.path.join(self.artifacts_dir, "model.pkl")
            preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomerException(e, sys)
        
    def predict_proba(self, features):
        try: 
            model_path = os.path.join(self.artifacts_dir, "model.pkl")
            preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions_prob = model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            raise CustomerException(e, sys)
        

# ['from_bank', 'to_bank', 'amount_received']
# ['account', 'account_1', 'receiving_currency', 'payment_currency', 'payment_format', 'day']
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
