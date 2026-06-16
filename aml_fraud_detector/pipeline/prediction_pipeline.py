    
import sys
import os
from typing import Dict, List, Optional

import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.utils import (
    get_prediction_features,
    normalize_column_name,
    clean_categorical_fields,
    NormalizationReport,
)
from aml_fraud_detector.entity.artifact_entity import REQUIRED_INPUT_FIELDS


class PredictionPipeline:
    def __init__(self, artifacts_dir: Optional[str] = None):
        self.artifacts_dir = artifacts_dir or "artifacts"
        self.model_path = os.path.join(self.artifacts_dir, "model.pkl")
        self.preprocessor_path = os.path.join(self.artifacts_dir, "preprocessor.pkl")
        self.feature_metadata_path = os.path.join(
            self.artifacts_dir, "feature_metadata.json"
        )
        self._model = None
        self._preprocessor = None
        self._expected_features: Optional[List[str]] = None
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            logging.info(f"Loading model from: {self.model_path}")
            self._model = load_object(file_path=self.model_path)
            
            logging.info(f"Loading preprocessor from: {self.preprocessor_path}")
            self._preprocessor = load_object(file_path=self.preprocessor_path)
            
            self._expected_features = self._get_expected_features()
            logging.info(
                f"PredictionPipeline initialized with "
                f"{len(self._expected_features) if self._expected_features else 'unknown'} "
                f"expected features"
            )
        except Exception as e:
            logging.error(f"Failed to load prediction artifacts: {e}", exc_info=True)
            raise CustomerException(e, sys)

    def _get_expected_features(self) -> Optional[List[str]]:
        import json
        try:
            if os.path.exists(self.feature_metadata_path):
                with open(self.feature_metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                original_features = metadata.get("original_features", [])
                return [normalize_column_name(f) for f in original_features]
        except Exception as e:
            logging.warning(f"Failed to load feature metadata: {e}")
        return REQUIRED_INPUT_FIELDS

    def _prepare_features(
        self,
        features: pd.DataFrame,
    ) -> tuple[pd.DataFrame, NormalizationReport]:
        feature_cols = self._expected_features or REQUIRED_INPUT_FIELDS
        
        logging.info(
            f"Preparing prediction features using unified normalization, "
            f"expected columns: {feature_cols}"
        )
        
        normalized_df, norm_report = get_prediction_features(
            features,
            feature_columns=feature_cols,
            derive_temporal=True,
            clean_categorical=True,
        )
        
        if norm_report.renamed_columns:
            logging.info(f"Renamed input columns: {norm_report.renamed_columns}")
        if norm_report.derived_columns:
            logging.info(f"Derived temporal columns: {norm_report.derived_columns}")
        
        return normalized_df, norm_report

    def predict(self, features: pd.DataFrame):
        try:
            if self._preprocessor is None or self._model is None:
                self._load_artifacts()

            normalized_features, norm_report = self._prepare_features(features)
            
            logging.info(f"Applying preprocessor to normalized features")
            data_scaled = self._preprocessor.transform(normalized_features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()
            
            logging.info("Generating predictions")
            predictions = self._model.predict(data_scaled)
            return predictions
        except Exception as e:
            logging.error("Prediction failed", exc_info=True)
            raise CustomerException(e, sys)
        
    def predict_proba(self, features: pd.DataFrame):
        try:
            if self._preprocessor is None or self._model is None:
                self._load_artifacts()

            normalized_features, norm_report = self._prepare_features(features)
            
            logging.info(f"Applying preprocessor to normalized features")
            data_scaled = self._preprocessor.transform(normalized_features)
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()
            
            logging.info("Generating prediction probabilities")
            predictions_prob = self._model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            logging.error("Prediction (proba) failed", exc_info=True)
            raise CustomerException(e, sys)


class CustomData:
    def __init__(self, **kwargs):
        for field in REQUIRED_INPUT_FIELDS:
            if field in kwargs:
                setattr(self, field, kwargs[field])
            else:
                setattr(self, field, None)
        
        for key, value in kwargs.items():
            normalized_key = normalize_column_name(key)
            if normalized_key in REQUIRED_INPUT_FIELDS and not hasattr(self, normalized_key):
                setattr(self, normalized_key, value)
    
    def get_data_as_DataFrame(self):
        try:
            custom_data_input_dict = {}
            for field in REQUIRED_INPUT_FIELDS:
                value = getattr(self, field, None)
                custom_data_input_dict[field] = [value]
            
            df = pd.DataFrame(custom_data_input_dict)
            
            df, _ = clean_categorical_fields(df)
            
            return df

        except Exception as e:
            logging.error("Failed to create DataFrame from custom data", exc_info=True)
            raise CustomerException(e, sys)
