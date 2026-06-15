    
import sys
import os
import pandas as pd
from typing import Dict, List, Any

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object


REQUIRED_FIELDS: Dict[str, type] = {
    "from_bank": int,
    "account": str,
    "to_bank": int,
    "account_1": str,
    "amount_received": float,
    "receiving_currency": str,
    "payment_currency": str,
    "payment_format": str,
    "day": str,
}


class PredictionPipeline:
    def __init__(self):
        self._model = None
        self._preprocessor = None

    def _load_models(self):
        if self._model is None or self._preprocessor is None:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            self._model = load_object(file_path=model_path)
            self._preprocessor = load_object(file_path=preprocessor_path)

    def predict(self, features):
        try: 
            self._load_models()
            data_scaled = self._preprocessor.transform(features)
            predictions = self._model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomerException(e, sys)
        
    def predict_proba(self, features):
        try: 
            self._load_models()
            data_scaled = self._preprocessor.transform(features)
            predictions_prob = self._model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            raise CustomerException(e, sys)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
        return df

    def _validate_row(self, row: pd.Series, row_idx: int) -> List[str]:
        errors = []
        actual_fields = set(row.index)
        required_fields_set = set(REQUIRED_FIELDS.keys())

        missing_fields = required_fields_set - actual_fields
        if missing_fields:
            errors.append(f"缺少字段: {', '.join(sorted(missing_fields))}")

        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in row:
                continue
            value = row[field]
            if pd.isna(value):
                errors.append(f"字段 '{field}' 值为空")
                continue
            try:
                if expected_type == int:
                    if isinstance(value, bool):
                        errors.append(f"字段 '{field}' 类型错误: 布尔值不能转换为整数")
                    else:
                        int(str(value))
                elif expected_type == float:
                    if isinstance(value, bool):
                        errors.append(f"字段 '{field}' 类型错误: 布尔值不能转换为浮点数")
                    else:
                        float(str(value))
                elif expected_type == str:
                    str_val = str(value)
                    if not str_val.strip():
                        errors.append(f"字段 '{field}' 为空字符串")
            except (ValueError, TypeError) as e:
                errors.append(f"字段 '{field}' 格式错误: 预期 {expected_type.__name__}, 实际值 '{value}'")

        return errors

    def _get_warnings(self, row: pd.Series) -> List[str]:
        warnings = []
        actual_fields = set(row.index)
        required_fields_set = set(REQUIRED_FIELDS.keys())
        extra_fields = actual_fields - required_fields_set
        if extra_fields:
            warnings.append(f"额外字段（将被保留）: {', '.join(sorted(extra_fields))}")
        return warnings

    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for field in REQUIRED_FIELDS.keys():
            if field not in df.columns:
                df[field] = None
        df = df[list(REQUIRED_FIELDS.keys())]
        
        if "from_bank" in df.columns:
            df["from_bank"] = df["from_bank"].astype("object")
        if "to_bank" in df.columns:
            df["to_bank"] = df["to_bank"].astype("object")
        
        return df

    def predict_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info(f"Starting batch prediction for {len(input_df)} rows")
            self._load_models()

            input_df = self._normalize_columns(input_df)
            results = []

            for idx, (row_idx, row) in enumerate(input_df.iterrows()):
                result_row = row.to_dict()
                errors = self._validate_row(row, idx)
                warnings = self._get_warnings(row)
                all_messages = errors + warnings

                if errors:
                    result_row["prediction_label"] = None
                    result_row["fraud_probability"] = None
                    result_row["error_reason"] = "; ".join(all_messages)
                    results.append(result_row)
                    continue

                try:
                    row_df = pd.DataFrame([row])
                    aligned_df = self._align_features(row_df)
                    
                    prediction = self._model.predict(self._preprocessor.transform(aligned_df))
                    prediction_proba = self._model.predict_proba(self._preprocessor.transform(aligned_df))
                    
                    result_row["prediction_label"] = int(prediction[0])
                    result_row["fraud_probability"] = float(prediction_proba[0][1])
                    result_row["error_reason"] = "; ".join(warnings) if warnings else None
                    
                except Exception as e:
                    result_row["prediction_label"] = None
                    result_row["fraud_probability"] = None
                    result_row["error_reason"] = f"预测失败: {str(e)}" + (f"; {'; '.join(warnings)}" if warnings else "")
                
                results.append(result_row)

            result_df = pd.DataFrame(results)
            output_columns = list(input_df.columns) + ["prediction_label", "fraud_probability", "error_reason"]
            result_df = result_df[output_columns]
            
            success_count = result_df["prediction_label"].notna().sum()
            fail_count = result_df["prediction_label"].isna().sum()
            logging.info(f"Batch prediction completed: {success_count} success, {fail_count} failed")
            
            return result_df

        except Exception as e:
            logging.error(f"Batch prediction failed: {str(e)}")
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
