import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.configuration import TrainingConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str
    test_data_path: str
    raw_data_path: str


class DataIngestion:
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config or TrainingConfig()
        self._resolved = self.training_config.to_resolved_dict()
        tc = self.training_config
        self.ingestion_config = DataIngestionConfig(
            train_data_path=tc.artifacts_subpath("train.csv"),
            test_data_path=tc.artifacts_subpath("test.csv"),
            raw_data_path=tc.artifacts_subpath("data.csv"),
        )
        logging.info(
            f"DataIngestion initialized with resolved config: "
            f"source={self._resolved['data']['source_path']}, "
            f"sample_size={self._resolved['data']['sample_size']}, "
            f"test_size={self._resolved['data']['test_size']}"
        )

    def initiate_data_ingestion(self) -> Tuple[str, str, pd.DataFrame]:
        logging.info("Entered the 'data ingestion' method or component")
        try:
            csv_file = self.training_config.resolve_source_path()
            logging.info(f"CSV File Path: {csv_file}")

            if os.path.exists(csv_file):
                logging.info(f"File found: {csv_file}")
            else:
                raise CustomerException(
                    FileNotFoundError(f"Data source file not found: {csv_file}"), sys
                )

            df = pd.read_csv(csv_file)
            logging.info(f"Read the dataset as DataFrame, shape={df.shape}")

            sample_size = self.training_config.data.sample_size
            if sample_size and isinstance(sample_size, int) and sample_size < len(df):
                df_sample = df.sample(
                    n=sample_size,
                    random_state=self.training_config.data.random_state,
                )
                logging.info(f"Sampled {sample_size} rows from dataset of {len(df)} rows")
            else:
                df_sample = df
                logging.info("Using full dataset (no sampling applied)")

            os.makedirs(
                os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),
                exist_ok=True,
            )
            df_sample.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(
                df_sample,
                test_size=self.training_config.data.test_size,
                random_state=self.training_config.data.random_state,
            )
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(
                f"Ingestion completed: train={len(train_set)}, test={len(test_set)}"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                df_sample,
            )
        except Exception as e:
            raise CustomerException(e, sys)
