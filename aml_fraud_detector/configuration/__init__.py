import os
import yaml
from typing import Optional
from aml_fraud_detector.logger import logging
from aml_fraud_detector.constants import (
    ENV_DATA_PATH,
    ENV_CONFIG_PATH,
    DEFAULT_DATA_FILE,
    DEFAULT_CONFIG_FILE,
    DEFAULT_SCHEMA_FILE,
    SCHEMA_REQUIRED_COLUMNS_KEY,
    SCHEMA_TARGET_COLUMN_KEY,
    AML_REQUIRED_COLUMNS,
    AML_TARGET_COLUMN,
    AML_NUMERICAL_COLUMNS,
    AML_CATEGORICAL_COLUMNS,
)
from aml_fraud_detector.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)


class Configuration:
    def __init__(
        self,
        config_file_path: Optional[str] = None,
        schema_file_path: Optional[str] = None,
    ):
        self.config_file_path = self._resolve_config_file_path(config_file_path)
        self.schema_file_path = self._resolve_schema_file_path(schema_file_path)
        self.config_info = self._load_yaml_file(self.config_file_path)
        self.schema_info = self._load_yaml_file(self.schema_file_path)

    def _resolve_config_file_path(self, config_file_path: Optional[str]) -> str:
        if config_file_path and os.path.exists(config_file_path):
            logging.info(f"Using provided config file: {config_file_path}")
            return config_file_path
        env_config = os.getenv(ENV_CONFIG_PATH)
        if env_config and os.path.exists(env_config):
            logging.info(f"Using config file from {ENV_CONFIG_PATH}: {env_config}")
            return env_config
        if os.path.exists(DEFAULT_CONFIG_FILE):
            logging.info(f"Using default config file: {DEFAULT_CONFIG_FILE}")
            return DEFAULT_CONFIG_FILE
        logging.info(f"No config file found, will use defaults")
        return None

    def _resolve_schema_file_path(self, schema_file_path: Optional[str]) -> str:
        if schema_file_path and os.path.exists(schema_file_path):
            logging.info(f"Using provided schema file: {schema_file_path}")
            return schema_file_path
        if os.path.exists(DEFAULT_SCHEMA_FILE):
            logging.info(f"Using default schema file: {DEFAULT_SCHEMA_FILE}")
            return DEFAULT_SCHEMA_FILE
        logging.info(f"No schema file found, will use built-in schema defaults")
        return None

    def _load_yaml_file(self, file_path: Optional[str]) -> dict:
        if not file_path or not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, "r") as f:
                content = yaml.safe_load(f) or {}
            logging.info(f"Loaded YAML config from {file_path}")
            return content
        except Exception as e:
            logging.warning(f"Failed to load YAML from {file_path}: {e}")
            return {}

    def _resolve_source_data_path(self) -> str:
        env_data_path = os.getenv(ENV_DATA_PATH)
        if env_data_path and os.path.exists(env_data_path):
            logging.info(f"Using data path from {ENV_DATA_PATH}: {env_data_path}")
            return env_data_path
        if self.config_info and "data_ingestion" in self.config_info:
            config_data_path = self.config_info["data_ingestion"].get("source_data_path")
            if config_data_path and os.path.exists(config_data_path):
                logging.info(f"Using data path from config file: {config_data_path}")
                return config_data_path
        if os.path.exists(DEFAULT_DATA_FILE):
            logging.info(f"Using default data file: {DEFAULT_DATA_FILE}")
            return DEFAULT_DATA_FILE
        logging.warning(f"No valid data source found. Tried: {ENV_DATA_PATH} env var, config file, and {DEFAULT_DATA_FILE}")
        return DEFAULT_DATA_FILE

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        source_data_path = self._resolve_source_data_path()
        config = DataIngestionConfig(source_data_path=source_data_path)
        if self.config_info and "data_ingestion" in self.config_info:
            di_config = self.config_info["data_ingestion"]
            if "train_data_path" in di_config:
                config.train_data_path = di_config["train_data_path"]
            if "test_data_path" in di_config:
                config.test_data_path = di_config["test_data_path"]
            if "raw_data_path" in di_config:
                config.raw_data_path = di_config["raw_data_path"]
            if "train_test_split_ratio" in di_config:
                config.train_test_split_ratio = float(di_config["train_test_split_ratio"])
            if "random_state" in di_config:
                config.random_state = int(di_config["random_state"])
            if "sample_size" in di_config:
                config.sample_size = int(di_config["sample_size"])
        logging.info(f"DataIngestionConfig resolved: source={config.source_data_path}")
        return config

    def get_data_validation_config(self) -> DataValidationConfig:
        return DataValidationConfig(schema_file_path=self.schema_file_path)

    def get_schema(self) -> dict:
        if self.schema_info and SCHEMA_REQUIRED_COLUMNS_KEY in self.schema_info:
            required_columns = self.schema_info[SCHEMA_REQUIRED_COLUMNS_KEY]
            target_column = self.schema_info.get(SCHEMA_TARGET_COLUMN_KEY, AML_TARGET_COLUMN)
            numerical_columns = self.schema_info.get("numerical_columns", AML_NUMERICAL_COLUMNS)
            categorical_columns = self.schema_info.get("categorical_columns", AML_CATEGORICAL_COLUMNS)
            logging.info(f"Using schema from schema file: {len(required_columns)} required columns")
        else:
            required_columns = AML_REQUIRED_COLUMNS
            target_column = AML_TARGET_COLUMN
            numerical_columns = AML_NUMERICAL_COLUMNS
            categorical_columns = AML_CATEGORICAL_COLUMNS
            logging.info(f"Using built-in schema defaults: {len(required_columns)} required columns")
        return {
            SCHEMA_REQUIRED_COLUMNS_KEY: required_columns,
            SCHEMA_TARGET_COLUMN_KEY: target_column,
            "numerical_columns": numerical_columns,
            "categorical_columns": categorical_columns,
        }

    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig()

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig()
