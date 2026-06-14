import os
import yaml
from typing import Optional, Dict, Any
from aml_fraud_detector.logger import logging
from aml_fraud_detector.constants import (
    ENV_DATA_PATH,
    ENV_CONFIG_PATH,
    ENV_DATA_CONFIG_PATH,
    DEFAULT_DATA_FILE,
    DEFAULT_CONFIG_FILE,
    DEFAULT_DATA_CONFIG_FILE,
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
        data_config_file_path: Optional[str] = None,
        schema_file_path: Optional[str] = None,
    ):
        self.model_config_file_path = self._resolve_model_config_file_path(config_file_path)
        self.data_config_file_path = self._resolve_data_config_file_path(data_config_file_path)
        self.schema_file_path = self._resolve_schema_file_path(schema_file_path)

        self.model_config_info = self._load_yaml_file(self.model_config_file_path)
        self.data_config_info = self._load_yaml_file(self.data_config_file_path)
        self.schema_info = self._load_yaml_file(self.schema_file_path)

        if self.model_config_info:
            keys = list(self.model_config_info.keys())
            logging.info(
                f"[Model Config] Loaded {len(keys)} top-level key(s) from model.yaml: {keys}. "
                f"Note: Only 'data_ingestion' key (if present) is used for backward compatibility — "
                f"all other keys are left untouched for user-defined training parameters."
            )

    def _resolve_model_config_file_path(self, config_file_path: Optional[str]) -> Optional[str]:
        if config_file_path and os.path.exists(config_file_path):
            logging.info(f"[Model Config] Using provided model config: {config_file_path}")
            return config_file_path
        env_config = os.getenv(ENV_CONFIG_PATH)
        if env_config and os.path.exists(env_config):
            logging.info(f"[Model Config] Using {ENV_CONFIG_PATH} env var: {env_config}")
            return env_config
        if os.path.exists(DEFAULT_CONFIG_FILE):
            logging.info(f"[Model Config] Using default model config: {DEFAULT_CONFIG_FILE}")
            return DEFAULT_CONFIG_FILE
        logging.info("[Model Config] No model config file found")
        return None

    def _resolve_data_config_file_path(self, data_config_file_path: Optional[str]) -> Optional[str]:
        if data_config_file_path and os.path.exists(data_config_file_path):
            logging.info(f"[Data Config] Using provided data config: {data_config_file_path}")
            return data_config_file_path
        env_data_config = os.getenv(ENV_DATA_CONFIG_PATH)
        if env_data_config and os.path.exists(env_data_config):
            logging.info(f"[Data Config] Using {ENV_DATA_CONFIG_PATH} env var: {env_data_config}")
            return env_data_config
        if os.path.exists(DEFAULT_DATA_CONFIG_FILE):
            logging.info(f"[Data Config] Using dedicated data config: {DEFAULT_DATA_CONFIG_FILE}")
            return DEFAULT_DATA_CONFIG_FILE
        logging.info("[Data Config] No dedicated data config found — will use model.yaml data_ingestion + built-in defaults")
        return None

    def _resolve_schema_file_path(self, schema_file_path: Optional[str]) -> Optional[str]:
        if schema_file_path and os.path.exists(schema_file_path):
            logging.info(f"[Schema] Using provided schema: {schema_file_path}")
            return schema_file_path
        if os.path.exists(DEFAULT_SCHEMA_FILE):
            logging.info(f"[Schema] Using default schema: {DEFAULT_SCHEMA_FILE}")
            return DEFAULT_SCHEMA_FILE
        logging.info("[Schema] No schema file found, will use built-in schema defaults")
        return None

    def _load_yaml_file(self, file_path: Optional[str]) -> Dict[str, Any]:
        if not file_path or not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, "r") as f:
                content = yaml.safe_load(f) or {}
            logging.info(f"Loaded YAML: {file_path}")
            return content
        except Exception as e:
            logging.warning(f"Failed to load YAML {file_path}: {e}")
            return {}

    def _get_data_ingestion_block(self) -> Dict[str, Any]:
        primary = self.data_config_info.get("data_ingestion", {}) if self.data_config_info else {}
        fallback = self.model_config_info.get("data_ingestion", {}) if self.model_config_info else {}
        if primary and fallback:
            merged = dict(fallback)
            merged.update(primary)
            logging.info(
                f"[Data Ingestion] Merged data_config.yaml ({len(primary)} keys) "
                f"over model.yaml data_ingestion fallback ({len(fallback)} keys)"
            )
            return merged
        if primary:
            logging.info(f"[Data Ingestion] Using data_config.yaml data_ingestion block ({len(primary)} keys)")
            return primary
        if fallback:
            logging.info(
                f"[Data Ingestion] BACKWARD COMPAT: Using model.yaml data_ingestion block "
                f"({len(fallback)} keys) as fallback"
            )
            return fallback
        logging.info("[Data Ingestion] No data_ingestion block in any config — using built-in defaults")
        return {}

    def _resolve_source_data_path(self) -> str:
        env_data_path = os.getenv(ENV_DATA_PATH)
        if env_data_path and os.path.exists(env_data_path):
            logging.info(f"[Source Data] Priority 1 — {ENV_DATA_PATH} env var: {env_data_path}")
            return env_data_path

        di_block = self._get_data_ingestion_block()

        from_data_config = self.data_config_info.get("data_ingestion", {}).get("source_data_path") if self.data_config_info else None
        if from_data_config and os.path.exists(str(from_data_config)):
            logging.info(f"[Source Data] Priority 2 — data_config.yaml: {from_data_config}")
            return str(from_data_config)

        from_model_yaml = self.model_config_info.get("data_ingestion", {}).get("source_data_path") if self.model_config_info else None
        if from_model_yaml and os.path.exists(str(from_model_yaml)):
            logging.info(f"[Source Data] Priority 3 — model.yaml (compat): {from_model_yaml}")
            return str(from_model_yaml)

        if os.path.exists(DEFAULT_DATA_FILE):
            logging.info(f"[Source Data] Priority 4 — default bundled data: {DEFAULT_DATA_FILE}")
            return DEFAULT_DATA_FILE

        logging.warning(
            f"[Source Data] No valid source after trying all layers: "
            f"{ENV_DATA_PATH} env → data_config.yaml → model.yaml → {DEFAULT_DATA_FILE}. "
            f"Returning default path as a hint."
        )
        return DEFAULT_DATA_FILE

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        source_data_path = self._resolve_source_data_path()
        config = DataIngestionConfig(source_data_path=source_data_path)

        di_block = self._get_data_ingestion_block()
        if di_block:
            if "train_data_path" in di_block:
                config.train_data_path = str(di_block["train_data_path"])
            if "test_data_path" in di_block:
                config.test_data_path = str(di_block["test_data_path"])
            if "raw_data_path" in di_block:
                config.raw_data_path = str(di_block["raw_data_path"])
            if "train_test_split_ratio" in di_block:
                config.train_test_split_ratio = float(di_block["train_test_split_ratio"])
            if "random_state" in di_block:
                config.random_state = int(di_block["random_state"])
            if "sample_size" in di_block:
                config.sample_size = int(di_block["sample_size"])

        logging.info(
            f"[DataIngestionConfig] source={config.source_data_path}, "
            f"sample_size={config.sample_size}, "
            f"split_ratio={config.train_test_split_ratio}"
        )
        return config

    def get_data_validation_config(self) -> DataValidationConfig:
        return DataValidationConfig(schema_file_path=self.schema_file_path)

    def get_schema(self) -> dict:
        if self.schema_info and SCHEMA_REQUIRED_COLUMNS_KEY in self.schema_info:
            required_columns = self.schema_info[SCHEMA_REQUIRED_COLUMNS_KEY]
            target_column = self.schema_info.get(SCHEMA_TARGET_COLUMN_KEY, AML_TARGET_COLUMN)
            numerical_columns = self.schema_info.get("numerical_columns", AML_NUMERICAL_COLUMNS)
            categorical_columns = self.schema_info.get("categorical_columns", AML_CATEGORICAL_COLUMNS)
            logging.info(f"[Schema] Using {self.schema_file_path} — {len(required_columns)} required columns")
        else:
            required_columns = AML_REQUIRED_COLUMNS
            target_column = AML_TARGET_COLUMN
            numerical_columns = AML_NUMERICAL_COLUMNS
            categorical_columns = AML_CATEGORICAL_COLUMNS
            logging.info("[Schema] Using built-in defaults — 11 required columns")
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
