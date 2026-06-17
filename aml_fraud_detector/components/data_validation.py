import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from aml_fraud_detector.exception import (
    CustomerException,
    DataQualityException,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.logger import logging
from aml_fraud_detector.configuration import TrainingConfig


@dataclass
class DataValidationArtifact:
    validation_status: bool
    validated_train_path: str
    validated_test_path: str
    missing_columns: List[str] = field(default_factory=list)
    null_report: Dict[str, int] = field(default_factory=dict)


class DataValidation:
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config or TrainingConfig()
        logging.info("DataValidation initialized")

    def initiate_data_validation(
        self, train_path: str, test_path: str
    ) -> DataValidationArtifact:
        logging.info("Entered the 'data validation' method or component")
        try:
            missing_columns = []
            null_report: Dict[str, int] = {}
            validation_status = True

            for label, path in [("train", train_path), ("test", test_path)]:
                if not os.path.exists(path):
                    raise DataQualityException(
                        ErrorCode.DATA_SOURCE_NOT_FOUND,
                        error_details=sys,
                        path=path,
                    )
                df = pd.read_csv(path)
                if len(df) == 0:
                    raise DataQualityException(
                        ErrorCode.DATA_EMPTY,
                        error_details=sys,
                        rows=0,
                    )
                df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('.', '_')
                target = self.training_config.features.target_column.lower()
                if target not in df.columns:
                    missing_columns.append(target)
                    validation_status = False
                null_counts = df.isnull().sum()
                null_cols = null_counts[null_counts > 0]
                for col, cnt in null_cols.items():
                    null_report[f"{label}.{col}"] = int(cnt)

            if missing_columns:
                raise DataQualityException(
                    ErrorCode.DATA_MISSING_COLUMNS,
                    error_details=sys,
                    missing=", ".join(missing_columns),
                )

            logging.info(
                f"Data validation {'passed' if validation_status else 'completed with warnings'}: "
                f"missing_columns={missing_columns}, null_report={null_report}"
            )

            return DataValidationArtifact(
                validation_status=validation_status,
                validated_train_path=train_path,
                validated_test_path=test_path,
                missing_columns=missing_columns,
                null_report=null_report,
            )
        except Exception as e:
            raise CustomerException(e, sys)
