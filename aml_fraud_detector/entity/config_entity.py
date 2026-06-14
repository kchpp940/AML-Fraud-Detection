import os
from dataclasses import dataclass
from aml_fraud_detector.constants import (
    ARTIFACTS_DIR,
    DEFAULT_DATA_FILE,
    TRAIN_TEST_SPLIT_RATIO,
    RANDOM_STATE,
    DATA_SAMPLE_SIZE,
)


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(ARTIFACTS_DIR, "train.csv")
    test_data_path: str = os.path.join(ARTIFACTS_DIR, "test.csv")
    raw_data_path: str = os.path.join(ARTIFACTS_DIR, "data.csv")
    source_data_path: str = DEFAULT_DATA_FILE
    train_test_split_ratio: float = TRAIN_TEST_SPLIT_RATIO
    random_state: int = RANDOM_STATE
    sample_size: int = DATA_SAMPLE_SIZE


@dataclass
class DataValidationConfig:
    schema_file_path: str = None
