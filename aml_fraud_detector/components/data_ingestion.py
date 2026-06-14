import os
import sys
from typing import Optional, Tuple

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.configuration import Configuration
from aml_fraud_detector.entity.config_entity import DataIngestionConfig
from aml_fraud_detector.components.data_validation import DataValidation, DataValidationResult

import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(
        self,
        ingestion_config: Optional[DataIngestionConfig] = None,
        config_manager: Optional[Configuration] = None,
    ):
        self.config_manager = config_manager or Configuration()
        self.ingestion_config = ingestion_config or self.config_manager.get_data_ingestion_config()
        self.data_validator = DataValidation(
            config=self.config_manager.get_data_validation_config(),
            schema=self.config_manager.get_schema(),
        )

    def _read_source_data(self) -> pd.DataFrame:
        source_path = self.ingestion_config.source_data_path
        logging.info(f"Attempting to read source data from: {source_path}")
        if not os.path.exists(source_path):
            available_paths = [
                f"ENV[{self.config_manager._resolve_source_data_path.__code__.co_varnames[0]}]",
            ]
            raise CustomerException(
                FileNotFoundError(
                    f"Source data file not found at: {source_path}. "
                    f"Please set the AML_DATA_PATH environment variable, "
                    f"configure source_data_path in config/model.yaml, "
                    f"or place the data CSV at the default location (artifacts/data.csv)."
                ),
                sys,
            )
        try:
            df = pd.read_csv(source_path)
            logging.info(f"Successfully read {len(df)} rows from {source_path}")
            return df
        except pd.errors.EmptyDataError as e:
            raise CustomerException(
                ValueError(f"Source data file is empty: {source_path}"),
                sys,
            )
        except Exception as e:
            raise CustomerException(
                IOError(f"Failed to read CSV from {source_path}: {str(e)}"),
                sys,
            )

    def _validate_before_split(self, df: pd.DataFrame) -> DataValidationResult:
        logging.info("Running data validation BEFORE train/test split")
        result = self.data_validator.validate_source_data(
            self.ingestion_config.source_data_path,
            df,
        )
        if not result.is_valid:
            error_details = "\n  - ".join([""] + result.errors)
            raise CustomerException(
                ValueError(
                    f"Cannot proceed to train/test split — data validation failed. "
                    f"Errors:{error_details}"
                ),
                sys,
            )
        for warning in result.warnings:
            logging.warning(f"[Validation Warning] {warning}")
        return result

    def _sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        sample_size = self.ingestion_config.sample_size
        total_rows = len(df)
        if sample_size and sample_size < total_rows:
            logging.info(
                f"Sampling {sample_size} rows out of {total_rows} "
                f"(random_state={self.ingestion_config.random_state})"
            )
            df_sample = df.sample(n=sample_size, random_state=self.ingestion_config.random_state)
            logging.info(f"Sampled data shape: {df_sample.shape}")
            return df_sample
        logging.info(
            f"No sampling needed: requested sample_size={sample_size}, "
            f"available rows={total_rows}"
        )
        return df

    def _save_artifacts(
        self,
        df_raw: pd.DataFrame,
        train_set: pd.DataFrame,
        test_set: pd.DataFrame,
    ) -> None:
        for path in [
            self.ingestion_config.raw_data_path,
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path,
        ]:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Saving raw data ({len(df_raw)} rows) to {self.ingestion_config.raw_data_path}")
        df_raw.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
        logging.info(f"Saving train data ({len(train_set)} rows) to {self.ingestion_config.train_data_path}")
        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        logging.info(f"Saving test data ({len(test_set)} rows) to {self.ingestion_config.test_data_path}")
        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

    def initiate_data_ingestion(self) -> Tuple[str, str]:
        logging.info("Entered the 'data ingestion' method or component")
        try:
            logging.info("=" * 70)
            logging.info(f"Data Ingestion Configuration:")
            logging.info(f"  source_data_path        = {self.ingestion_config.source_data_path}")
            logging.info(f"  raw_data_path           = {self.ingestion_config.raw_data_path}")
            logging.info(f"  train_data_path         = {self.ingestion_config.train_data_path}")
            logging.info(f"  test_data_path          = {self.ingestion_config.test_data_path}")
            logging.info(f"  train_test_split_ratio  = {self.ingestion_config.train_test_split_ratio}")
            logging.info(f"  random_state            = {self.ingestion_config.random_state}")
            logging.info(f"  sample_size             = {self.ingestion_config.sample_size}")
            logging.info("=" * 70)

            df = self._read_source_data()
            self._validate_before_split(df)

            df_sample = self._sample_data(df)

            split_ratio = self.ingestion_config.train_test_split_ratio
            random_state = self.ingestion_config.random_state
            logging.info(
                f"Initiating train/test split with test_size={split_ratio}, "
                f"random_state={random_state}"
            )
            train_set, test_set = train_test_split(
                df_sample,
                test_size=split_ratio,
                random_state=random_state,
            )
            logging.info(
                f"Train/test split complete: train={len(train_set)}, test={len(test_set)}"
            )

            self._save_artifacts(df_sample, train_set, test_set)
            logging.info("Ingestion of the data completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except CustomerException:
            raise
        except Exception as e:
            raise CustomerException(e, sys)
