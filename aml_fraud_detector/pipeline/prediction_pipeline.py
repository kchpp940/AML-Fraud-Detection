
import sys
import os
import pandas as pd
from typing import Optional, Dict, Any, Tuple

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.utils.risk_explainer import (
    explain_risk,
    load_feature_metadata,
    load_training_summary,
    validate_training_signature,
    INFERENCE_CONTRACT_VERSION,
)


class PredictionPipeline:
    def __init__(self):
        self.artifacts_dir = "artifacts"
        self.model_path = os.path.join(self.artifacts_dir, "model.pkl")
        self.preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")
        self.feature_metadata_path = os.path.join(self.artifacts_dir, "feature_metadata.json")
        self.training_summary_path = os.path.join(self.artifacts_dir, "training_summary.json")
        self._model = None
        self._preprocessor = None
        self._feature_metadata = None
        self._training_summary = None
        self._signature_valid = False
        self._signature_msg = ""

    def _load_artifacts(self):
        if self._model is None or self._preprocessor is None:
            self._model = load_object(file_path=self.model_path)
            self._preprocessor = load_object(file_path=self.preprocessor_path)
        return self._model, self._preprocessor

    def _validate_signatures(self) -> Tuple[bool, str]:
        if self._signature_valid and self._signature_msg:
            return self._signature_valid, self._signature_msg

        metadata = self._load_feature_metadata()
        summary = load_training_summary(self.training_summary_path)

        if metadata is None or summary is None:
            self._signature_valid = False
            self._signature_msg = "缺失特征元数据或训练摘要，无法验证产物一致性"
            return self._signature_valid, self._signature_msg

        metadata_sig = metadata.get("training_signature", "")
        summary_sig = summary.get("training_signature", "")

        valid, msg = validate_training_signature(metadata_sig, summary_sig)
        self._signature_valid = valid
        self._signature_msg = msg
        logging.info(msg)
        return valid, msg

    def _load_feature_metadata(self) -> Optional[Dict[str, Any]]:
        if self._feature_metadata is not None:
            return self._feature_metadata
        if os.path.exists(self.feature_metadata_path):
            try:
                self._feature_metadata = load_feature_metadata(self.feature_metadata_path)
            except Exception as e:
                logging.warning(f"Failed to load feature_metadata.json: {e}")
                self._feature_metadata = None
        return self._feature_metadata

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

    def predict_with_explanation(self, features, top_n: int = 5) -> Dict[str, Any]:
        try:
            model, preprocessor = self._load_artifacts()

            feature_metadata = self._load_feature_metadata()
            sig_valid, sig_msg = self._validate_signatures()

            if feature_metadata is None:
                data_scaled = preprocessor.transform(features)
                predictions = model.predict(data_scaled)
                probas = model.predict_proba(data_scaled)

                n_rows = features.shape[0]
                results = []
                for i in range(n_rows):
                    results.append({
                        "row_index": i,
                        "prediction": int(predictions[i]),
                        "prediction_label": "欺诈交易" if predictions[i] == 1 else "正常交易",
                        "fraud_probability": round(float(probas[i, 1]), 4),
                        "top_factors": [],
                    })

                fraud_count = int(sum(1 for r in results if r["prediction"] == 1))
                return {
                    "contract_version": INFERENCE_CONTRACT_VERSION,
                    "training_signature": "",
                    "signature_valid": False,
                    "signature_message": "feature_metadata.json 缺失，请重新训练模型以生成风险解释所需的元数据",
                    "is_batch": n_rows > 1,
                    "count": n_rows,
                    "fraud_count": fraud_count,
                    "normal_count": n_rows - fraud_count,
                    "fraud_rate": round(fraud_count / n_rows * 100, 2) if n_rows > 0 else 0.0,
                    "results": results,
                }

            explanation = explain_risk(
                features_df=features,
                model=model,
                preprocessor=preprocessor,
                feature_metadata=feature_metadata,
                top_n=top_n,
            )

            explanation["signature_valid"] = sig_valid
            explanation["signature_message"] = sig_msg

            if not sig_valid:
                logging.warning(
                    f"Training signature validation failed: {sig_msg}. "
                    "Risk explanation may be inconsistent with model predictions."
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
