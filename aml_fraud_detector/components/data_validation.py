import os
import sys
from dataclasses import dataclass

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


@dataclass
class DataValidationArtifact:
    validated_train_path: str
    validated_test_path: str


class DataValidation:
    def __init__(self):
        logging.info("DataValidation initialized")

    def initiate_data_validation(
        self, train_path: str, test_path: str
    ) -> DataValidationArtifact:
        logging.info("Entered the 'data validation' method or component")
        try:
            for path in (train_path, test_path):
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Data validation input not found: {path}")
            logging.info("Data validation done")
            return DataValidationArtifact(
                validated_train_path=train_path,
                validated_test_path=test_path,
            )
        except Exception as e:
            raise CustomerException(e, sys)
