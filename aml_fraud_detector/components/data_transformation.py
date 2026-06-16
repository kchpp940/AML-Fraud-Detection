import sys
import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder, CountEncoder

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import save_object
from aml_fraud_detector.utils.feature_normalization import (
    normalize_dataframe,
    NormalizationReport,
)
from aml_fraud_detector.configuration import TrainingConfig


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str


@dataclass
class DataTransformationArtifact:
    train_array: np.ndarray
    test_array: np.ndarray
    feature_columns: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    target_column: str = ""
    preprocessor_path: str = ""


_ENCODING_TYPE_MAP = {
    "account": "frequency",
    "account_1": "frequency",
    "payment_format": "onehot",
    "day": "onehot",
}

_FEATURE_LABELS = {
    "amount_received": "交易金额",
    "account": "发起账户",
    "account_1": "接收账户",
    "payment_format": "支付方式",
    "day": "交易星期",
}


class DataTransformation:
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config or TrainingConfig()
        self._resolved = self.training_config.to_resolved_dict()
        tc = self.training_config
        self.data_transformation_config = DataTransformationConfig(
            preprocessor_obj_file_path=tc.artifacts_subpath("preprocessor.pkl")
        )
        self.feature_metadata_path = tc.artifacts_subpath("feature_metadata.json")
        logging.info(
            f"DataTransformation initialized with resolved config: "
            f"target={self._resolved['features']['target_column']}, "
            f"drop={self._resolved['features']['drop_columns']}, "
            f"preprocessor_out={self._resolved['output']['preprocessor_pkl']}"
        )

    def get_data_transformer_object(self, numerical_columns, categorical_columns):
        """
        This function is responsible for data transformation
        """
        try:
            num_transformer = make_pipeline(
                SimpleImputer(strategy='median'),
                RobustScaler()
            )

            freq_encoder = CountEncoder(normalize=True)
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

            high_card_cols = [c for c in ['account', 'account_1'] if c in categorical_columns]
            low_card_cols = [c for c in ['payment_format', 'day'] if c in categorical_columns]

            cat_tf_steps = []
            if high_card_cols:
                cat_tf_steps.append((freq_encoder, high_card_cols))
            if low_card_cols:
                cat_tf_steps.append((one_hot_encoder, low_card_cols))

            if cat_tf_steps:
                cat_transformer = make_column_transformer(*cat_tf_steps, remainder="drop")
                preprocessor = make_column_transformer(
                    (num_transformer, numerical_columns),
                    (cat_transformer, categorical_columns),
                    remainder="drop"
                )
            else:
                preprocessor = make_column_transformer(
                    (num_transformer, numerical_columns),
                    remainder="drop"
                )

            logging.info("Preprocessed both numerical and categorical columns")
            return preprocessor

        except Exception as e:
            raise CustomerException(e, sys)

    def _save_feature_metadata(
        self,
        norm_report: NormalizationReport,
        feature_columns: List[str],
        numerical_features: List[str],
        categorical_features: List[str],
    ) -> str:
        encoding_info = {}
        for col in feature_columns:
            if col in numerical_features:
                encoding_info[col] = {"type": "numerical"}
            elif col in _ENCODING_TYPE_MAP:
                encoding_info[col] = {"type": _ENCODING_TYPE_MAP[col]}
            else:
                encoding_info[col] = {"type": "frequency"}

        feature_labels = {col: _FEATURE_LABELS.get(col, col) for col in feature_columns}

        metadata = {
            "contract_version": "1.0",
            "original_features": feature_columns,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
            "encoding_info": encoding_info,
            "feature_labels": feature_labels,
            "normalization_report": norm_report.to_dict(),
            "baseline_values": {},
            "training_stats": {},
            "training_signature": self._resolved.get("run_info", {}).get(
                "created_at", ""
            )[:16].replace("-", "").replace(":", "").replace("T", ""),
            "generated_at": pd.Timestamp.now().isoformat(),
        }

        os.makedirs(os.path.dirname(self.feature_metadata_path), exist_ok=True)
        with open(self.feature_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logging.info(f"Feature metadata saved to: {self.feature_metadata_path}")
        return os.path.abspath(self.feature_metadata_path)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> DataTransformationArtifact:
        logging.info("\nEntered the 'data transformation' method or component")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data completed")

            target_column_name = self.training_config.features.target_column
            drop_columns = self.training_config.features.drop_columns

            logging.info("Applying unified feature normalization to training data")
            train_df_normalized, train_norm_report = normalize_dataframe(
                train_df,
                target_column_name=target_column_name,
                drop_columns=drop_columns,
                derive_temporal=True,
                clean_categorical=True,
            )

            logging.info("Applying unified feature normalization to test data")
            test_df_normalized, test_norm_report = normalize_dataframe(
                test_df,
                target_column_name=target_column_name,
                drop_columns=drop_columns,
                derive_temporal=True,
                clean_categorical=True,
            )

            if train_norm_report.renamed_columns:
                logging.info(
                    f"Renamed columns: {train_norm_report.renamed_columns}"
                )
            if train_norm_report.derived_columns:
                logging.info(
                    f"Derived temporal columns: {train_norm_report.derived_columns}"
                )

            logging.info(f"Train Dataframe Head : \n{train_df_normalized.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df_normalized.head().to_string()}")

            extra_drop = [target_column_name]
            configured_drop = [
                c for c in drop_columns
                if c != target_column_name
            ]
            all_drop_columns = extra_drop + configured_drop
            normalized_drop = [
                c for c in all_drop_columns
                if c in train_df_normalized.columns
            ]

            input_features_train_df = train_df_normalized.drop(columns=normalized_drop, axis=1)
            target_feature_train_df = train_df_normalized[target_column_name]

            input_features_test_df = test_df_normalized.drop(columns=normalized_drop, axis=1)
            target_feature_test_df = test_df_normalized[target_column_name]

            numerical_features = train_norm_report.numerical_columns
            logging.info(f"Columns name of numerical features: {numerical_features}")
            categorical_features = train_norm_report.categorical_columns
            logging.info(f"Columns name of categorical features: {categorical_features}")
            feature_columns = numerical_features + categorical_features

            logging.info("Saving feature metadata with normalization report")
            self._save_feature_metadata(
                train_norm_report,
                feature_columns,
                numerical_features,
                categorical_features,
            )

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(
                numerical_features, categorical_features
            )

            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()
            else:
                input_feature_train_arr = np.asarray(input_feature_train_arr)
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()
            else:
                input_feature_test_arr = np.asarray(input_feature_test_arr)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Saved data preprocessing object")

            return DataTransformationArtifact(
                train_array=train_arr,
                test_array=test_arr,
                feature_columns=feature_columns,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                target_column=target_column_name,
                preprocessor_path=os.path.abspath(
                    self.data_transformation_config.preprocessor_obj_file_path
                ),
            )

        except Exception as e:
            raise CustomerException(e, sys)
