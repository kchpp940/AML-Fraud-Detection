import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


DEFAULT_CONFIG_PATH = os.path.join("config", "training_config.yaml")

ENV_PREFIX = "AML_TRAIN_"

ENV_MAPPING = {
    "data.source_path": f"{ENV_PREFIX}DATA_SOURCE_PATH",
    "data.sample_size": f"{ENV_PREFIX}DATA_SAMPLE_SIZE",
    "data.test_size": f"{ENV_PREFIX}DATA_TEST_SIZE",
    "data.random_state": f"{ENV_PREFIX}DATA_RANDOM_STATE",
    "features.target_column": f"{ENV_PREFIX}FEATURES_TARGET",
    "models.selection_metric": f"{ENV_PREFIX}MODEL_METRIC",
    "models.enabled.RandomForest": f"{ENV_PREFIX}MODEL_RF_ENABLED",
    "models.enabled.AdaBoost": f"{ENV_PREFIX}MODEL_AB_ENABLED",
    "models.enabled.GradientBoosting": f"{ENV_PREFIX}MODEL_GB_ENABLED",
    "models.enabled.XGBoost": f"{ENV_PREFIX}MODEL_XGB_ENABLED",
    "output.artifacts_dir": f"{ENV_PREFIX}OUTPUT_DIR",
    "config.path": f"{ENV_PREFIX}CONFIG_PATH",
}


def _get_nested(d: Dict[str, Any], path: str) -> Any:
    keys = path.split(".")
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def _set_nested(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_env_value(raw: str, hint_type: Optional[type] = None) -> Any:
    raw = raw.strip()
    if hint_type is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    if hint_type is int:
        return int(raw)
    if hint_type is float:
        return float(raw)
    if raw.lower() in ("null", "none", ""):
        return None
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    for path, env_name in ENV_MAPPING.items():
        if path == "config.path":
            continue
        value = os.environ.get(env_name)
        if value is not None:
            hint_type = None
            current = _get_nested(config, path)
            if current is not None:
                hint_type = type(current)
            parsed = _parse_env_value(value, hint_type)
            _set_nested(config, path, parsed)
            logging.info(f"Env override: {path} = {parsed} (from {env_name})")
    return config


@dataclass
class DataConfig:
    source_path: str
    sample_size: Optional[int]
    test_size: float
    random_state: int


@dataclass
class FeaturesConfig:
    target_column: str
    drop_columns: List[str]


@dataclass
class ModelsConfig:
    selection_metric: str
    enabled: Dict[str, bool]
    param_grid: Dict[str, Dict[str, Any]]


@dataclass
class OutputConfig:
    artifacts_dir: str


@dataclass
class TrainingSummary:
    data_source: str = ""
    data_rows: int = 0
    train_rows: int = 0
    test_rows: int = 0
    feature_columns: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    target_column: str = ""
    candidate_models: List[str] = field(default_factory=list)
    selection_metric: str = ""
    best_model_name: str = ""
    best_model_params: Dict[str, Any] = field(default_factory=dict)
    best_metric_value: float = 0.0
    all_model_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    preprocessor_path: str = ""
    model_path: str = ""
    artifacts_dir: str = ""
    summary_path: str = ""


class TrainingConfig:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = self._resolve_config_path(config_path)
        logging.info(f"Loading training config from: {self.config_path}")
        raw = self._load_yaml(self.config_path)
        raw = _apply_env_overrides(raw)
        self._raw = raw

        self.data: DataConfig = DataConfig(
            source_path=raw["data"]["source_path"],
            sample_size=raw["data"].get("sample_size"),
            test_size=float(raw["data"]["test_size"]),
            random_state=int(raw["data"]["random_state"]),
        )
        self.features: FeaturesConfig = FeaturesConfig(
            target_column=raw["features"]["target_column"],
            drop_columns=list(raw["features"].get("drop_columns", [])),
        )
        self.models: ModelsConfig = ModelsConfig(
            selection_metric=raw["models"]["selection_metric"],
            enabled=dict(raw["models"].get("enabled", {})),
            param_grid=dict(raw["models"].get("param_grid", {})),
        )
        self.output: OutputConfig = OutputConfig(
            artifacts_dir=raw["output"]["artifacts_dir"],
        )
        self._validate()
        logging.info("Training config loaded and validated successfully")

    @staticmethod
    def _resolve_config_path(explicit: Optional[str]) -> str:
        env_path = os.environ.get(ENV_MAPPING["config.path"])
        for candidate in (explicit, env_path, DEFAULT_CONFIG_PATH):
            if candidate and os.path.isfile(candidate):
                return os.path.abspath(candidate)
        default_abs = os.path.abspath(DEFAULT_CONFIG_PATH)
        logging.warning(
            f"Config file not found in any candidate location; using defaults. "
            f"Expected: {default_abs}"
        )
        return default_abs

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        defaults = {
            "data": {
                "source_path": "notebook/data/HI-Small_Trans.csv",
                "sample_size": 50000,
                "test_size": 0.2,
                "random_state": 42,
            },
            "features": {
                "target_column": "is_laundering",
                "drop_columns": [
                    "timestamp", "date", "time", "amount_paid",
                    "receiving_currency", "payment_currency",
                    "from_bank", "to_bank",
                ],
            },
            "models": {
                "selection_metric": "Recall",
                "enabled": {
                    "RandomForest": True,
                    "AdaBoost": True,
                    "GradientBoosting": False,
                    "XGBoost": True,
                },
                "param_grid": {},
            },
            "output": {"artifacts_dir": "artifacts"},
        }
        if not os.path.isfile(path):
            return defaults
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
        except Exception as e:
            raise CustomerException(
                RuntimeError(f"Failed to parse YAML config '{path}': {e}"), sys
            )

        def _deep_merge(base: Dict, overlay: Dict) -> Dict:
            for k, v in overlay.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _deep_merge(base[k], v)
                else:
                    base[k] = v
            return base

        return _deep_merge(defaults, loaded)

    def _validate(self) -> None:
        if not isinstance(self.data.test_size, float) or not (0.0 < self.data.test_size < 1.0):
            raise CustomerException(
                ValueError(f"data.test_size must be in (0, 1), got {self.data.test_size}"), sys
            )
        valid_metrics = {"Precision", "Recall", "F1 score", "F1"}
        if self.models.selection_metric not in valid_metrics:
            raise CustomerException(
                ValueError(
                    f"models.selection_metric must be one of {sorted(valid_metrics)}, "
                    f"got '{self.models.selection_metric}'"
                ), sys
            )
        if self.models.selection_metric == "F1":
            self.models.selection_metric = "F1 score"
        if not any(self.models.enabled.values()):
            raise CustomerException(
                ValueError("At least one model must be enabled in models.enabled"), sys
            )

    def resolve_source_path(self) -> str:
        if os.path.isabs(self.data.source_path):
            return self.data.source_path
        for base in (os.getcwd(), os.path.dirname(os.path.dirname(self.config_path))):
            candidate = os.path.join(base, self.data.source_path)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        return os.path.abspath(self.data.source_path)

    def artifacts_subpath(self, *parts: str) -> str:
        return os.path.join(self.output.artifacts_dir, *parts)
