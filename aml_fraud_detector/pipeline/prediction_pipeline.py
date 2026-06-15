    
import sys
import os
import pandas as pd
from typing import Dict, Any, Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.entity.artifact_entity import FeatureSchema


class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.feature_schema_path = os.path.join("artifacts", "feature_schema.pkl")
        self._model = None
        self._preprocessor = None
        self._feature_schema = None
        self._loaded = False

    def _load_artifacts(self):
        if self._loaded:
            return
        try:
            logging.info("Loading model, preprocessor, and feature schema...")
            self._model = load_object(file_path=self.model_path)
            self._preprocessor = load_object(file_path=self.preprocessor_path)
            self._feature_schema = load_object(file_path=self.feature_schema_path)
            self._loaded = True
            logging.info(f"Loaded feature schema with {len(self._feature_schema.model_input_columns)} input columns")
            logging.info(f"Model input columns: {self._feature_schema.model_input_columns}")
        except Exception as e:
            raise CustomerException(e, sys)

    @property
    def feature_schema(self) -> FeatureSchema:
        self._load_artifacts()
        return self._feature_schema

    def _validate_and_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        self._load_artifacts()
        logging.info(f"Input features columns: {features.columns.tolist()}")
        aligned_df = self._feature_schema.align_features(features)
        logging.info(f"Aligned features columns: {aligned_df.columns.tolist()}")
        logging.info(f"Aligned features dtypes:\n{aligned_df.dtypes}")
        return aligned_df

    def predict(self, features):
        try: 
            self._load_artifacts()
            data_aligned = self._validate_and_transform(features)
            data_scaled = self._preprocessor.transform(data_aligned)
            if hasattr(data_scaled, 'toarray'):
                data_scaled = data_scaled.toarray()
            predictions = self._model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomerException(e, sys)
        
    def predict_proba(self, features):
        try: 
            self._load_artifacts()
            data_aligned = self._validate_and_transform(features)
            data_scaled = self._preprocessor.transform(data_aligned)
            if hasattr(data_scaled, 'toarray'):
                data_scaled = data_scaled.toarray()
            predictions_prob = self._model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            raise CustomerException(e, sys)

    def get_required_input_fields(self):
        self._load_artifacts()
        schema = self._feature_schema
        derived_source_cols = [
            config['source'] for config in schema.derived_features.values()
            if 'source' in config
        ]
        required_cols = []
        for col in schema.original_raw_columns:
            cleaned = schema.clean_column_name(col)
            if cleaned in schema.dropped_columns and cleaned not in derived_source_cols:
                continue
            if cleaned in schema.model_input_columns or cleaned in derived_source_cols:
                required_cols.append(col)
        return required_cols


class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs
        self._prediction_pipeline = PredictionPipeline()

    def get_data_as_DataFrame(self):
        try:
            custom_data_input_dict = {k: [v] for k, v in self.data.items()}
            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"CustomData input DataFrame columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            raise CustomerException(e, sys)

    @classmethod
    def from_flask_request(cls, request_form):
        data = {}
        for key, value in request_form.items():
            if value is not None and value != '':
                data[key] = value
        return cls(**data)

    def get_required_fields(self):
        return self._prediction_pipeline.get_required_input_fields()
