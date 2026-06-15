import sys
import os
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


class DataTransformation:
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config or TrainingConfig()
        tc = self.training_config
        self.data_transformation_config = DataTransformationConfig(
            preprocessor_obj_file_path=tc.artifacts_subpath("preprocessor.pkl")
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

            cat_transformer = make_column_transformer(
                (freq_encoder, high_card_cols) if high_card_cols else ("drop", []),
                (one_hot_encoder, low_card_cols) if low_card_cols else ("drop", []),
                remainder="drop"
            )

            preprocessor = make_column_transformer(
                (num_transformer, numerical_columns),
                (cat_transformer, categorical_columns),
                remainder="drop"
            )

            logging.info("Preprocessed both numerical and categorical columns")
            return preprocessor

        except Exception as e:
            raise CustomerException(e, sys)

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> DataTransformationArtifact:
        logging.info("\nEntered the 'data transformation' method or component")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data completed")

            train_df.columns = train_df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
            test_df.columns = test_df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
            logging.info("Train and Test dataframe columns name renamed")

            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

            if "from_bank" in train_df.columns:
                train_df["from_bank"] = train_df["from_bank"].astype("object")
            if "to_bank" in train_df.columns:
                train_df["to_bank"] = train_df["to_bank"].astype("object")
            if "from_bank" in test_df.columns:
                test_df["from_bank"] = test_df["from_bank"].astype("object")
            if "to_bank" in test_df.columns:
                test_df["to_bank"] = test_df["to_bank"].astype("object")

            if "timestamp" in train_df.columns:
                train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
                train_df["date"] = train_df["timestamp"].dt.date
                train_df["day"] = train_df["timestamp"].dt.day_name()
                train_df["time"] = train_df["timestamp"].dt.time
            if "timestamp" in test_df.columns:
                test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
                test_df["date"] = test_df["timestamp"].dt.date
                test_df["day"] = test_df["timestamp"].dt.day_name()
                test_df["time"] = test_df["timestamp"].dt.time

            target_column_name = self.training_config.features.target_column
            extra_drop = [target_column_name]
            configured_drop = [
                c for c in self.training_config.features.drop_columns
                if c != target_column_name
            ]
            drop_columns = extra_drop + configured_drop
            existing_drop = [c for c in drop_columns if c in train_df.columns]

            input_features_train_df = train_df.drop(columns=existing_drop, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=existing_drop, axis=1)
            target_feature_test_df = test_df[target_column_name]

            numerical_features = input_features_train_df.select_dtypes(include=np.number).columns.tolist()
            logging.info(f"Columns name of numerical features: {numerical_features}")
            categorical_features = input_features_train_df.select_dtypes(include=object).columns.tolist()
            logging.info(f"Columns name of categorical features: {categorical_features}")
            feature_columns = numerical_features + categorical_features

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(
                numerical_features, categorical_features
            )

            logging.info("Applying preprocessing object on training and testing datasets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_train_arr = input_feature_train_arr.toarray()
            input_feature_test_arr = preprocessing_obj.transform(input_features_test_df)
            input_feature_test_arr = input_feature_test_arr.toarray()

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
