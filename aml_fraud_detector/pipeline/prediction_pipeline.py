    
import sys
import os
import pandas as pd
from typing import Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import (
    load_object,
    align_features_with_schema,
    load_feature_schema,
)
from aml_fraud_detector.entity.artifact_entity import DataTransformationArtifact


class PredictionPipeline:
    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        feature_schema_path: Optional[str] = None,
    ):
        self.model_path = model_path or os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = preprocessor_path or os.path.join("artifacts", "preprocessor.pkl")
        self.feature_schema_path = feature_schema_path or os.path.join("artifacts", "feature_schema.pkl")
        self._model = None
        self._preprocessor = None
        self._feature_schema = None
        self._loaded = False

    @classmethod
    def from_data_transformation_artifact(
        cls,
        data_transformation_artifact: DataTransformationArtifact,
        model_path: Optional[str] = None,
    ) -> 'PredictionPipeline':
        return cls(
            model_path=model_path or os.path.join("artifacts", "model.pkl"),
            preprocessor_path=data_transformation_artifact.preprocessor_object_file_path,
            feature_schema_path=data_transformation_artifact.feature_schema_file_path,
        )

    def _load_artifacts(self):
        if self._loaded:
            return
        try:
            logging.info("Loading model, preprocessor, and feature schema...")
            self._model = load_object(file_path=self.model_path)
            self._preprocessor = load_object(file_path=self.preprocessor_path)
            self._feature_schema = load_feature_schema(self.feature_schema_path)
            self._loaded = True
            logging.info(
                f"Artifact paths - model: {self.model_path}, "
                f"preprocessor: {self.preprocessor_path}, "
                f"schema: {self.feature_schema_path}"
            )
        except Exception as e:
            raise CustomerException(e, sys)

    @property
    def feature_schema(self):
        self._load_artifacts()
        return self._feature_schema

    def predict(self, features):
        try:
            self._load_artifacts()
            data_aligned = align_features_with_schema(features, self._feature_schema, log_context="PredictionPipeline.predict")
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
            data_aligned = align_features_with_schema(features, self._feature_schema, log_context="PredictionPipeline.predict_proba")
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

    def get_aligned_DataFrame(self):
        try:
            raw_df = self.get_data_as_DataFrame()
            schema = self._prediction_pipeline.feature_schema
            aligned_df = align_features_with_schema(raw_df, schema, log_context="CustomData")
            return aligned_df

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
