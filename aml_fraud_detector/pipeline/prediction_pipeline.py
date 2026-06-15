    
import sys
import os
import json
import pandas as pd
from typing import Dict, List, Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object

_SUMMARY_PATH = os.path.join("artifacts", "training_summary.json")
_MODEL_PATH = os.path.join("artifacts", "model.pkl")
_PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")


class _Schema:
    __slots__ = (
        "feature_columns",
        "numerical_features",
        "categorical_features",
        "drop_columns",
        "target_column",
        "derived_features",
    )

    def __init__(self, summary: Dict):
        self.feature_columns: List[str] = summary.get("feature_columns", [])
        self.numerical_features: List[str] = summary.get("numerical_features", [])
        self.categorical_features: List[str] = summary.get("categorical_features", [])
        self.drop_columns: List[str] = summary.get("drop_columns", [])
        self.target_column: str = summary.get("target_column", "")
        self.derived_features: Dict = summary.get("derived_features", {})

    def to_dict(self) -> Dict:
        num_types = {f: "numerical" for f in self.numerical_features}
        cat_types = {f: "categorical" for f in self.categorical_features}
        return {
            "feature_columns": self.feature_columns,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "drop_columns": self.drop_columns,
            "target_column": self.target_column,
            "derived_features": self.derived_features,
            "feature_types": {**num_types, **cat_types},
        }


class PredictionPipeline:
    def __init__(self):
        self._model = None
        self._preprocessor = None
        self._schema: Optional[_Schema] = None

    def _load_models(self):
        if self._model is None or self._preprocessor is None:
            self._model = load_object(file_path=_MODEL_PATH)
            self._preprocessor = load_object(file_path=_PREPROCESSOR_PATH)

    def _load_schema(self) -> _Schema:
        if self._schema is not None:
            return self._schema
        if not os.path.isfile(_SUMMARY_PATH):
            raise CustomerException(
                FileNotFoundError(
                    f"训练摘要文件不存在: {_SUMMARY_PATH}。"
                    f"请先运行训练流程或确保 artifacts 目录包含 training_summary.json"
                ),
                sys,
            )
        with open(_SUMMARY_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._schema = _Schema(raw)
        logging.info(
            f"Loaded inference schema from training_summary.json: "
            f"features={self._schema.feature_columns}, "
            f"drop={self._schema.drop_columns}, "
            f"derived={self._schema.derived_features}"
        )
        return self._schema

    @property
    def schema(self) -> _Schema:
        return self._load_schema()

    def get_schema_info(self) -> Dict:
        return self._load_schema().to_dict()

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

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        schema = self._load_schema()
        df = df.copy()
        if schema.derived_features.get("day_from_timestamp") and "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                if "day" not in df.columns:
                    df["day"] = df["timestamp"].dt.day_name().astype(object)
                else:
                    mask = df["day"].isna()
                    if mask.any():
                        df.loc[mask, "day"] = df.loc[mask, "timestamp"].dt.day_name().astype(object)
            except Exception as e:
                logging.warning(f"Failed to parse timestamp column: {e}")
        return df

    def _validate_row(self, row: pd.Series, schema: _Schema) -> List[str]:
        errors = []
        feature_set = set(schema.feature_columns)
        row_fields = set(row.index)

        missing = feature_set - row_fields
        if missing:
            errors.append(f"缺少模型必需字段: {', '.join(sorted(missing))}")

        for feat in schema.feature_columns:
            if feat not in row:
                continue
            value = row[feat]
            if pd.isna(value):
                errors.append(f"字段 '{feat}' 值为空")
                continue
            if feat in schema.numerical_features:
                try:
                    if isinstance(value, bool):
                        raise ValueError()
                    float(str(value))
                except (ValueError, TypeError):
                    errors.append(f"字段 '{feat}' 格式错误: 预期数值, 实际值 '{value}'")
            elif feat in schema.categorical_features:
                str_val = str(value).strip()
                if not str_val:
                    errors.append(f"字段 '{feat}' 为空字符串")

        return errors

    def _prepare_features(self, row_df: pd.DataFrame, schema: _Schema) -> pd.DataFrame:
        df = row_df.copy()

        drop_cols = [schema.target_column] + [
            c for c in schema.drop_columns if c != schema.target_column
        ]
        existing_drop = [c for c in drop_cols if c in df.columns]
        if existing_drop:
            df = df.drop(columns=existing_drop, axis=1)

        for feat in schema.feature_columns:
            if feat not in df.columns:
                df[feat] = None
            if feat in schema.numerical_features and feat in df.columns:
                df[feat] = pd.to_numeric(df[feat], errors="coerce")
            if feat in schema.categorical_features and feat in df.columns:
                df[feat] = df[feat].astype(object)

        df = df[schema.feature_columns]
        return df

    def predict_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info(f"Starting batch prediction for {len(input_df)} rows")
            self._load_models()
            schema = self._load_schema()

            input_df = self._normalize_columns(input_df)
            input_df = self._engineer_features(input_df)

            results = []

            for idx, (row_idx, row) in enumerate(input_df.iterrows()):
                result_row = row.to_dict()
                errors = self._validate_row(row, schema)

                if errors:
                    result_row["prediction_label"] = None
                    result_row["fraud_probability"] = None
                    result_row["error_reason"] = "; ".join(errors)
                    results.append(result_row)
                    continue

                try:
                    row_df = pd.DataFrame([row])
                    features_df = self._prepare_features(row_df, schema)

                    data_scaled = self._preprocessor.transform(features_df)
                    prediction = self._model.predict(data_scaled)
                    prediction_proba = self._model.predict_proba(data_scaled)

                    result_row["prediction_label"] = int(prediction[0])
                    result_row["fraud_probability"] = float(prediction_proba[0][1])
                    result_row["error_reason"] = None

                except Exception as e:
                    result_row["prediction_label"] = None
                    result_row["fraud_probability"] = None
                    result_row["error_reason"] = f"预测失败: {str(e)}"

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
