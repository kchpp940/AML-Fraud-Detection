import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml

from aml_fraud_detector.exception import (
    CustomerException,
    ConfigException,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.logger import logging
from aml_fraud_detector.runtime.workspace import WorkspaceContext, WorkspaceMode


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


def _apply_env_overrides(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    overrides: List[Dict[str, Any]] = []
    for path, env_name in ENV_MAPPING.items():
        if path == "config.path":
            continue
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        hint_type = None
        original = _get_nested(config, path)
        if original is not None:
            hint_type = type(original)
        parsed = _parse_env_value(raw_value, hint_type)
        _set_nested(config, path, parsed)
        overrides.append({
            "path": path,
            "env_name": env_name,
            "env_value": raw_value,
            "original_value": original,
            "resolved_value": parsed,
        })
        logging.info(
            f"Env override: {path} = {original!r} -> {parsed!r} (from {env_name}='{raw_value}')"
        )
    return config, overrides


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
    resolved_config: Dict[str, Any] = field(default_factory=dict)


class TrainingConfig:
    def __init__(self, config_path: Optional[str] = None):
        self.created_at = datetime.now().isoformat()
        self.config_path, self.config_path_source = self._resolve_config_path(config_path)
        logging.info(
            f"Loading training config (source={self.config_path_source}): {self.config_path}"
        )
        raw, self.yaml_config_used = self._load_yaml(self.config_path)
        raw, self.env_overrides = _apply_env_overrides(raw)
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

        self.workspace: WorkspaceContext = WorkspaceContext.from_config_dict(raw)
        self.workspace.ensure_directories()

        self._resolved_cache: Optional[Dict[str, Any]] = None
        logging.info("Training config loaded and validated successfully")
        logging.info(
            f"Resolved config signature: data_source={self.resolve_source_path()}, "
            f"target={self.features.target_column}, metric={self.models.selection_metric}, "
            f"output={self.workspace.artifacts_dir}"
        )

    @staticmethod
    def _resolve_config_path(explicit: Optional[str]) -> Tuple[str, str]:
        env_path = os.environ.get(ENV_MAPPING["config.path"])
        if explicit:
            if os.path.isfile(explicit):
                return os.path.abspath(explicit), "explicit"
            raise ConfigException(
                ErrorCode.CONFIG_FILE_NOT_FOUND,
                error_details=sys,
                path=explicit,
            )
        if env_path:
            if os.path.isfile(env_path):
                return os.path.abspath(env_path), "env"
            raise ConfigException(
                ErrorCode.CONFIG_FILE_NOT_FOUND,
                error_details=sys,
                path=env_path,
                env_var=ENV_MAPPING["config.path"],
            )
        if os.path.isfile(DEFAULT_CONFIG_PATH):
            return os.path.abspath(DEFAULT_CONFIG_PATH), "default_file"
        default_abs = os.path.abspath(DEFAULT_CONFIG_PATH)
        logging.warning(
            f"Config file not found in any candidate location; using defaults only. "
            f"Expected: {default_abs}"
        )
        return default_abs, "defaults_only"

    @staticmethod
    def _load_yaml(path: str) -> Tuple[Dict[str, Any], bool]:
        defaults: Dict[str, Any] = {
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
            "output": {
                "artifacts_dir": "artifacts",
                "workspace": {
                    "mode": "flat",
                    "root": None,
                    "run_name": None,
                },
            },
        }
        if not os.path.isfile(path):
            return defaults, False
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigException(
                ErrorCode.CONFIG_PARSE_FAILED,
                error_details=sys,
                path=path,
                detail=str(e),
            )

        def _deep_merge(base: Dict, overlay: Dict) -> Dict:
            for k, v in overlay.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _deep_merge(base[k], v)
                else:
                    base[k] = v
            return base

        return _deep_merge(defaults, loaded), True

    def _validate(self) -> None:
        if not isinstance(self.data.test_size, float) or not (0.0 < self.data.test_size < 1.0):
            raise ConfigException(
                ErrorCode.CONFIG_INVALID_VALUE,
                error_details=sys,
                key="data.test_size",
                value=str(self.data.test_size),
                detail="must be in (0, 1)",
            )
        valid_metrics = {"Precision", "Recall", "F1 score", "F1"}
        if self.models.selection_metric not in valid_metrics:
            raise ConfigException(
                ErrorCode.CONFIG_INVALID_VALUE,
                error_details=sys,
                key="models.selection_metric",
                value=self.models.selection_metric,
                detail=f"must be one of {sorted(valid_metrics)}",
            )
        if self.models.selection_metric == "F1":
            self.models.selection_metric = "F1 score"
            for ov in self.env_overrides:
                if ov["path"] == "models.selection_metric":
                    ov["resolved_value"] = "F1 score"
            self._raw["models"]["selection_metric"] = "F1 score"
        if not any(self.models.enabled.values()):
            raise ConfigException(
                ErrorCode.CONFIG_INVALID_VALUE,
                error_details=sys,
                key="models.enabled",
                value=str(self.models.enabled),
                detail="at least one model must be enabled",
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
        return self.workspace.artifacts_subpath(*parts)

    def to_resolved_dict(self) -> Dict[str, Any]:
        """
        Return a complete, serializable snapshot of the *finally resolved* training
        configuration. Components and the training summary all reference this dict
        so that the exact configuration used for a training run can always be
        reproduced from the summary.

        Fields included:
          - run_info: created_at, config_path, config_path_source, yaml_config_used
          - env_overrides: list of {path, env_name, env_value, original_value, resolved_value}
          - data: source_path (resolved absolute), sample_size, test_size, random_state
          - features: target_column, drop_columns
          - models: selection_metric, enabled, param_grid
          - output: artifacts_dir (absolute), preprocessor_path, model_path, summary_path
          - workspace: workspace context info
        """
        if self._resolved_cache is not None:
            return self._resolved_cache

        ws = self.workspace
        snapshot: Dict[str, Any] = {
            "run_info": {
                "created_at": self.created_at,
                "config_path": self.config_path,
                "config_path_source": self.config_path_source,
                "yaml_config_used": self.yaml_config_used,
            },
            "env_overrides": list(self.env_overrides),
            "data": {
                "source_path": self.resolve_source_path(),
                "source_path_input": self.data.source_path,
                "sample_size": self.data.sample_size,
                "test_size": self.data.test_size,
                "random_state": self.data.random_state,
            },
            "features": {
                "target_column": self.features.target_column,
                "drop_columns": list(self.features.drop_columns),
            },
            "models": {
                "selection_metric": self.models.selection_metric,
                "enabled": dict(self.models.enabled),
                "enabled_display_names": [
                    display
                    for key, display in _MODEL_REGISTRY_ITEMS
                    if self.models.enabled.get(key)
                ],
                "param_grid": dict(self.models.param_grid),
            },
            "output": {
                "artifacts_dir": os.path.abspath(ws.artifacts_dir),
                "artifacts_dir_input": self.output.artifacts_dir,
                "train_csv": os.path.abspath(ws.get_artifact_path("train_csv")),
                "test_csv": os.path.abspath(ws.get_artifact_path("test_csv")),
                "raw_csv": os.path.abspath(ws.get_artifact_path("raw_csv")),
                "preprocessor_pkl": os.path.abspath(ws.get_artifact_path("preprocessor_pkl")),
                "model_pkl": os.path.abspath(ws.get_artifact_path("model_pkl")),
                "summary_json": os.path.abspath(ws.get_artifact_path("summary_json")),
            },
            "workspace": ws.to_dict(),
        }
        self._resolved_cache = snapshot
        return snapshot


_MODEL_REGISTRY_ITEMS = [
    ("RandomForest", "Random Forest"),
    ("AdaBoost", "AdaBoost"),
    ("GradientBoosting", "Gradient Boosting"),
    ("XGBoost", "XGBoost"),
]
