import os
import sys
import copy
import json
import yaml
from enum import Enum
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from aml_fraud_detector.config.config_schema import (
    AppConfig,
    TrainingConfigSection,
    PredictionConfigSection,
    ServerConfigSection,
    DataConfigSection,
    FeaturesConfigSection,
    ModelsConfigSection,
    OutputConfigSection,
    LoggingConfigSection,
    SecurityConfigSection,
)
from aml_fraud_detector.exception import ConfigException, wrap_exception
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.logger import logging


class ConfigProfile(str, Enum):
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"
    SMOKE = "smoke"


T = TypeVar("T")

ENV_PREFIX = "AML_"

DEFAULT_CONFIG_DIR = "config"
DEFAULT_CONFIG_FILE = "app_config.yaml"

ENV_CONFIG_MAPPING: Dict[str, str] = {
    "env": f"{ENV_PREFIX}ENV",
    "project_root": f"{ENV_PREFIX}PROJECT_ROOT",
    "app_name": f"{ENV_PREFIX}APP_NAME",
    "app_version": f"{ENV_PREFIX}APP_VERSION",
    "training.data.source_path": f"{ENV_PREFIX}TRAIN_DATA_SOURCE",
    "training.data.sample_size": f"{ENV_PREFIX}TRAIN_DATA_SAMPLE_SIZE",
    "training.data.test_size": f"{ENV_PREFIX}TRAIN_DATA_TEST_SIZE",
    "training.data.random_state": f"{ENV_PREFIX}TRAIN_DATA_RANDOM_STATE",
    "training.features.target_column": f"{ENV_PREFIX}TRAIN_FEATURE_TARGET",
    "training.models.selection_metric": f"{ENV_PREFIX}TRAIN_MODEL_METRIC",
    "training.output.artifacts_dir": f"{ENV_PREFIX}TRAIN_OUTPUT_DIR",
    "prediction.default_artifacts_dir": f"{ENV_PREFIX}PREDICT_ARTIFACTS_DIR",
    "prediction.model_file_name": f"{ENV_PREFIX}PREDICT_MODEL_FILE",
    "prediction.enable_lazy_load": f"{ENV_PREFIX}PREDICT_LAZY_LOAD",
    "prediction.risk_thresholds.critical": f"{ENV_PREFIX}PREDICT_RISK_THRESHOLD_CRITICAL",
    "prediction.risk_thresholds.high": f"{ENV_PREFIX}PREDICT_RISK_THRESHOLD_HIGH",
    "prediction.risk_thresholds.medium": f"{ENV_PREFIX}PREDICT_RISK_THRESHOLD_MEDIUM",
    "server.flask_host": f"{ENV_PREFIX}SERVER_HOST",
    "server.flask_port": f"{ENV_PREFIX}SERVER_PORT",
    "server.flask_debug": f"{ENV_PREFIX}SERVER_DEBUG",
    "server.streamlit_port": f"{ENV_PREFIX}SERVER_STREAMLIT_PORT",
    "server.enable_cors": f"{ENV_PREFIX}SERVER_ENABLE_CORS",
    "server.request_timeout": f"{ENV_PREFIX}SERVER_REQUEST_TIMEOUT",
    "server.max_content_length": f"{ENV_PREFIX}SERVER_MAX_CONTENT_LENGTH",
    "server.cors_origins": f"{ENV_PREFIX}SERVER_CORS_ORIGINS",
    "logging.level": f"{ENV_PREFIX}LOG_LEVEL",
    "logging.log_dir": f"{ENV_PREFIX}LOG_DIR",
    "logging.log_to_file": f"{ENV_PREFIX}LOG_TO_FILE",
    "logging.log_to_console": f"{ENV_PREFIX}LOG_TO_CONSOLE",
    "security.mask_secrets_in_logs": f"{ENV_PREFIX}SECURITY_MASK_SECRETS",
    "security.enable_input_sanitization": f"{ENV_PREFIX}SECURITY_SANITIZE_INPUT",
    "config.path": f"{ENV_PREFIX}CONFIG_PATH",
    "config.profile": f"{ENV_PREFIX}CONFIG_PROFILE",
}

REQUIRED_FIELDS: List[str] = []

SECRET_FIELD_PATTERNS: List[str] = [
    "api_key", "secret", "token", "password", "credential",
    "access_key", "private_key", "authorization",
]


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


def _parse_env_value(raw: str, hint_type: Optional[Type] = None) -> Any:
    raw = raw.strip()
    if hint_type is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    if hint_type is int:
        return int(raw)
    if hint_type is float:
        return float(raw)
    if hint_type is list:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return [item.strip() for item in raw.split(",") if item.strip()]
    if hint_type is dict:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
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


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _dataclass_from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    if not is_dataclass(cls):
        return data
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data and data[f.name] is not None:
            val = data[f.name]
            if is_dataclass(f.type):
                kwargs[f.name] = _dataclass_from_dict(f.type, val)
            elif f.type == List[str] and isinstance(val, list):
                kwargs[f.name] = [str(x) for x in val]
            elif f.type == Dict[str, bool] and isinstance(val, dict):
                kwargs[f.name] = {str(k): bool(v) for k, v in val.items()}
            elif f.type == Dict[str, Dict[str, Any]] and isinstance(val, dict):
                kwargs[f.name] = {str(k): dict(v) if isinstance(v, dict) else {} for k, v in val.items()}
            elif f.type == Dict[str, float] and isinstance(val, dict):
                kwargs[f.name] = {str(k): float(v) for k, v in val.items()}
            elif f.type == List[str] and isinstance(val, str):
                kwargs[f.name] = [val]
            else:
                kwargs[f.name] = val
    return cls(**kwargs)


def _resolve_absolute_path(
    path: str,
    base_dirs: Optional[List[str]] = None,
    check_existence: bool = False,
) -> str:
    if os.path.isabs(path):
        return path
    base_dirs = base_dirs or [os.getcwd()]
    for base in base_dirs:
        candidate = os.path.join(base, path)
        if check_existence:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        else:
            return os.path.abspath(candidate)
    if check_existence:
        return os.path.abspath(os.path.join(base_dirs[0], path))
    return os.path.abspath(path)


def _mask_value(key: str, value: Any, patterns: Optional[List[str]] = None) -> Any:
    patterns = patterns or SECRET_FIELD_PATTERNS
    key_lower = key.lower()
    is_secret = any(p in key_lower for p in patterns)
    if is_secret and value is not None:
        if isinstance(value, str) and len(value) > 4:
            return value[:2] + "***" + value[-2:]
        return "***MASKED***"
    return value


def _mask_dict_recursive(
    d: Dict[str, Any],
    patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _mask_dict_recursive(v, patterns)
        elif isinstance(v, list):
            result[k] = [
                _mask_dict_recursive(item, patterns) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            result[k] = _mask_value(k, v, patterns)
    return result


class ConfigService:
    def __init__(
        self,
        config_path: Optional[str] = None,
        profile: Optional[Union[str, ConfigProfile]] = None,
        project_root: Optional[str] = None,
    ):
        self._loaded_at = datetime.now().isoformat()
        self._env_overrides: List[Dict[str, Any]] = []
        self._yaml_sources: List[Dict[str, str]] = []
        self._defaults_used: List[str] = []
        self._config_path: Optional[str] = None
        self._config_path_source: str = "defaults"
        self._profile: str = self._resolve_profile(profile)
        self._project_root: str = self._resolve_project_root(project_root)
        self._effective_dict: Dict[str, Any] = {}
        self._sanitized_dict: Optional[Dict[str, Any]] = None

        raw_dict = self._build_raw_config(config_path)
        self._effective_dict = self._resolve_all_paths(raw_dict)
        self._validate_required(self._effective_dict)
        self._app_config: AppConfig = _dataclass_from_dict(AppConfig, self._effective_dict)
        self._sanitized_dict = _mask_dict_recursive(
            copy.deepcopy(self._effective_dict),
            self._app_config.security.secret_keys,
        )
        self._log_summary()

    def _resolve_profile(self, profile: Optional[Union[str, ConfigProfile]]) -> str:
        if profile:
            return profile.value if isinstance(profile, ConfigProfile) else str(profile)
        env_profile = os.environ.get(ENV_CONFIG_MAPPING["config.profile"])
        if env_profile:
            return env_profile
        env_env = os.environ.get(ENV_CONFIG_MAPPING["env"])
        if env_env:
            return env_env
        return ConfigProfile.LOCAL.value

    def _resolve_project_root(self, explicit_root: Optional[str]) -> str:
        if explicit_root:
            return os.path.abspath(explicit_root)
        env_root = os.environ.get(ENV_CONFIG_MAPPING["project_root"])
        if env_root:
            return os.path.abspath(env_root)
        return os.path.abspath(os.getcwd())

    def _resolve_config_path(self, explicit: Optional[str]) -> Tuple[Optional[str], str]:
        env_path = os.environ.get(ENV_CONFIG_MAPPING["config.path"])
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
                env_var=ENV_CONFIG_MAPPING["config.path"],
            )

        candidates = []
        if self._profile:
            profile_file = f"app_config.{self._profile}.yaml"
            candidates.append(os.path.join(DEFAULT_CONFIG_DIR, profile_file))
            candidates.append(os.path.join("config", profile_file))
        candidates.append(os.path.join(DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_FILE))
        candidates.append(os.path.join("config", "training_config.yaml"))
        if self._profile == ConfigProfile.SMOKE.value:
            candidates.append(os.path.join("config", "training_config_smoke.yaml"))

        for candidate in candidates:
            abs_candidate = os.path.join(self._project_root, candidate)
            if os.path.isfile(abs_candidate):
                return abs_candidate, f"profile_file:{os.path.basename(abs_candidate)}"

        return None, "defaults_only"

    def _load_yaml_dict(self, path: Optional[str]) -> Tuple[Dict[str, Any], bool]:
        result: Dict[str, Any] = {}
        if not path:
            return result, False
        if not os.path.isfile(path):
            return result, False
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                logging.warning(f"YAML root is not a dict in {path}, ignoring")
                return result, False
            self._yaml_sources.append({
                "path": path,
                "loaded_at": datetime.now().isoformat(),
            })
            return loaded, True
        except Exception as e:
            raise ConfigException(
                ErrorCode.CONFIG_PARSE_FAILED,
                error_details=sys,
                path=path,
                detail=str(e),
            )

    def _get_defaults_dict(self) -> Dict[str, Any]:
        return AppConfig().to_dict()

    def _build_raw_config(self, explicit_config_path: Optional[str]) -> Dict[str, Any]:
        defaults = self._get_defaults_dict()
        self._defaults_used = list(defaults.keys())

        resolved_path, path_source = self._resolve_config_path(explicit_config_path)
        self._config_path = resolved_path
        self._config_path_source = path_source

        yaml_config, yaml_loaded = self._load_yaml_dict(resolved_path)
        if yaml_loaded:
            defaults = _deep_merge(defaults, yaml_config)

        legacy_mapped = self._map_legacy_training_yaml(yaml_config)
        if legacy_mapped:
            defaults = _deep_merge(defaults, legacy_mapped)

        defaults["env"] = self._profile
        defaults["project_root"] = self._project_root
        if not defaults.get("app_name"):
            defaults["app_name"] = "AML Fraud Detector"
        if not defaults.get("app_version"):
            defaults["app_version"] = "1.0.0"

        defaults = self._apply_env_overrides(defaults)

        return defaults

    def _map_legacy_training_yaml(self, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        if not yaml_data:
            return {}
        legacy_keys = {"data", "features", "models", "output"}
        has_legacy = any(k in yaml_data for k in legacy_keys)
        has_new = any(k in yaml_data for k in {"training", "prediction", "server", "logging", "security"})
        if has_new:
            return {}
        if not has_legacy:
            return {}

        mapped: Dict[str, Any] = {"training": {}}
        for section in ("data", "features", "models", "output"):
            if section in yaml_data:
                mapped["training"][section] = copy.deepcopy(yaml_data[section])

        return mapped

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        overrides: List[Dict[str, Any]] = []
        for path, env_name in ENV_CONFIG_MAPPING.items():
            if path in ("config.path", "config.profile"):
                continue
            raw_value = os.environ.get(env_name)
            if raw_value is None:
                continue
            hint_type: Optional[Type] = None
            original = _get_nested(config, path)
            if original is not None:
                hint_type = type(original)
            try:
                parsed = _parse_env_value(raw_value, hint_type)
            except Exception as e:
                raise ConfigException(
                    ErrorCode.CONFIG_INVALID_VALUE,
                    error_details=sys,
                    key=path,
                    value=raw_value,
                    detail=f"env parse error from {env_name}: {e}",
                )
            _set_nested(config, path, parsed)
            override_record = {
                "path": path,
                "env_name": env_name,
                "env_value": "***MASKED***" if any(
                    p in path.lower() for p in SECRET_FIELD_PATTERNS
                ) else raw_value,
                "original_value": original,
                "resolved_value": _mask_value(path, parsed),
            }
            overrides.append(override_record)
            logging.info(
                f"Env override: {path} = {original!r} -> {_mask_value(path, parsed)!r} "
                f"(from {env_name})"
            )
        self._env_overrides = overrides
        return config

    def _resolve_all_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        base_dirs = [self._project_root]
        if self._config_path:
            base_dirs.append(os.path.dirname(os.path.dirname(self._config_path)))

        path_fields = [
            "training.data.source_path",
            "training.output.artifacts_dir",
            "prediction.default_artifacts_dir",
            "logging.log_dir",
        ]
        for field_path in path_fields:
            current = _get_nested(config, field_path)
            if isinstance(current, str) and current:
                resolved = _resolve_absolute_path(current, base_dirs, check_existence=False)
                _set_nested(config, field_path, resolved)

        if "training" in config and "output" in config["training"]:
            out = config["training"]["output"]
            artifacts_dir = out.get("artifacts_dir", "artifacts")
            path_names = [
                ("train_csv_name", "train_csv_path"),
                ("test_csv_name", "test_csv_path"),
                ("raw_csv_name", "raw_csv_path"),
                ("preprocessor_name", "preprocessor_path"),
                ("model_name", "model_path"),
                ("summary_name", "summary_path"),
                ("feature_metadata_name", "feature_metadata_path"),
                ("model_metadata_name", "model_metadata_path"),
                ("quality_report_name", "quality_report_path"),
                ("manifest_name", "manifest_path"),
            ]
            for name_key, path_key in path_names:
                name_val = out.get(name_key)
                if name_val:
                    out[path_key] = os.path.join(artifacts_dir, name_val)

        if "prediction" in config:
            pred = config["prediction"]
            artifacts_dir = pred.get("default_artifacts_dir", "artifacts")
            file_mappings = [
                ("model_file_name", "model_path"),
                ("preprocessor_file_name", "preprocessor_path"),
                ("feature_metadata_file_name", "feature_metadata_path"),
                ("model_metadata_file_name", "model_metadata_path"),
            ]
            for name_key, path_key in file_mappings:
                name_val = pred.get(name_key)
                if name_val:
                    pred[path_key] = os.path.join(artifacts_dir, name_val)

        return config

    def _validate_required(self, config: Dict[str, Any]) -> None:
        for field_path in REQUIRED_FIELDS:
            val = _get_nested(config, field_path)
            if val is None or (isinstance(val, str) and not val.strip()):
                raise ConfigException(
                    ErrorCode.CONFIG_MISSING_REQUIRED,
                    error_details=sys,
                    key=field_path,
                )

        data_cfg = _get_nested(config, "training.data")
        if isinstance(data_cfg, dict):
            test_size = data_cfg.get("test_size")
            if not isinstance(test_size, (int, float)) or not (0.0 < float(test_size) < 1.0):
                raise ConfigException(
                    ErrorCode.CONFIG_INVALID_VALUE,
                    error_details=sys,
                    key="training.data.test_size",
                    value=str(test_size),
                    detail="must be in (0, 1)",
                )

        models_cfg = _get_nested(config, "training.models")
        if isinstance(models_cfg, dict):
            metric = models_cfg.get("selection_metric")
            valid_metrics = {"Precision", "Recall", "F1 score", "F1"}
            if metric not in valid_metrics:
                raise ConfigException(
                    ErrorCode.CONFIG_INVALID_VALUE,
                    error_details=sys,
                    key="training.models.selection_metric",
                    value=str(metric),
                    detail=f"must be one of {sorted(valid_metrics)}",
                )
            if metric == "F1":
                models_cfg["selection_metric"] = "F1 score"

            enabled = models_cfg.get("enabled", {})
            if isinstance(enabled, dict) and not any(bool(v) for v in enabled.values()):
                raise ConfigException(
                    ErrorCode.CONFIG_INVALID_VALUE,
                    error_details=sys,
                    key="training.models.enabled",
                    value=str(enabled),
                    detail="at least one model must be enabled",
                )

        server_cfg = _get_nested(config, "server")
        if isinstance(server_cfg, dict):
            for port_key in ("flask_port", "streamlit_port"):
                port = server_cfg.get(port_key)
                if not isinstance(port, int) or not (1 <= port <= 65535):
                    raise ConfigException(
                        ErrorCode.CONFIG_INVALID_VALUE,
                        error_details=sys,
                        key=f"server.{port_key}",
                        value=str(port),
                        detail="must be a valid port number (1-65535)",
                    )

    def _log_summary(self) -> None:
        logging.info("=" * 72)
        logging.info("ConfigService - Configuration loaded")
        logging.info("=" * 72)
        logging.info(f"  Profile            : {self._profile}")
        logging.info(f"  Project root       : {self._project_root}")
        logging.info(f"  Config path source : {self._config_path_source}")
        if self._config_path:
            logging.info(f"  Config file        : {self._config_path}")
        logging.info(f"  YAML sources loaded: {len(self._yaml_sources)}")
        logging.info(f"  Env overrides      : {len(self._env_overrides)}")
        for ov in self._env_overrides:
            logging.info(f"    - {ov['path']} (from {ov['env_name']})")
        logging.info("=" * 72)

    @property
    def config(self) -> AppConfig:
        return self._app_config

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def project_root(self) -> str:
        return self._project_root

    @property
    def config_path(self) -> Optional[str]:
        return self._config_path

    @property
    def env_overrides(self) -> List[Dict[str, Any]]:
        return list(self._env_overrides)

    @property
    def yaml_sources(self) -> List[Dict[str, str]]:
        return list(self._yaml_sources)

    def resolve_path(
        self,
        path: str,
        subpath: Optional[str] = None,
        check_existence: bool = False,
    ) -> str:
        if subpath:
            path = os.path.join(path, subpath)
        return _resolve_absolute_path(
            path,
            base_dirs=[self._project_root],
            check_existence=check_existence,
        )

    def training_artifacts_path(self, *parts: str) -> str:
        artifacts_dir = self._app_config.training.output.artifacts_dir
        return os.path.join(artifacts_dir, *parts)

    def prediction_artifacts_path(self, *parts: str) -> str:
        artifacts_dir = self._app_config.prediction.default_artifacts_dir
        return os.path.join(artifacts_dir, *parts)

    def get_effective_config(self, sanitized: bool = True) -> Dict[str, Any]:
        if sanitized and self._sanitized_dict is not None:
            return copy.deepcopy(self._sanitized_dict)
        return copy.deepcopy(self._effective_dict)

    def get_config_metadata(self) -> Dict[str, Any]:
        return {
            "loaded_at": self._loaded_at,
            "profile": self._profile,
            "project_root": self._project_root,
            "config_path": self._config_path,
            "config_path_source": self._config_path_source,
            "yaml_sources": list(self._yaml_sources),
            "env_override_count": len(self._env_overrides),
            "env_overrides": [
                {k: v for k, v in ov.items()}
                for ov in self._env_overrides
            ],
        }

    def export_effective_config(
        self,
        output_path: Optional[str] = None,
        sanitized: bool = True,
        include_metadata: bool = True,
    ) -> str:
        export_data: Dict[str, Any] = {}
        if include_metadata:
            export_data["_config_metadata"] = self.get_config_metadata()
        export_data["config"] = self.get_effective_config(sanitized=sanitized)

        if output_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_sanitized" if sanitized else ""
            output_path = os.path.join(
                self._app_config.logging.log_dir,
                f"effective_config_{self._profile}_{stamp}{suffix}.json",
            )

        output_path = _resolve_absolute_path(output_path, [self._project_root])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            logging.info(
                f"Effective config exported ({'sanitized' if sanitized else 'raw'}) "
                f"to: {output_path}"
            )
            return output_path
        except Exception as e:
            raise wrap_exception(e, error_details=sys)

    def print_summary(self) -> None:
        cfg = self.get_effective_config(sanitized=True)
        print("\n" + "=" * 72)
        print("CONFIGURATION SUMMARY (SANITIZED)")
        print("=" * 72)
        meta = self.get_config_metadata()
        print(f"  Loaded at          : {meta['loaded_at']}")
        print(f"  Profile            : {meta['profile']}")
        print(f"  Project root       : {meta['project_root']}")
        print(f"  Config path source : {meta['config_path_source']}")
        if meta["config_path"]:
            print(f"  Config file        : {meta['config_path']}")
        print(f"  YAML sources       : {len(meta['yaml_sources'])}")
        print(f"  Env overrides      : {meta['env_override_count']}")
        for ov in meta["env_overrides"]:
            print(f"    - {ov['env_name']} -> {ov['path']}: "
                  f"{ov['original_value']!r} -> {ov['resolved_value']!r}")
        print("-" * 72)
        print("  Training:")
        tcfg = cfg.get("training", {})
        print(f"    Data source      : {tcfg.get('data', {}).get('source_path')}")
        print(f"    Sample size      : {tcfg.get('data', {}).get('sample_size')}")
        print(f"    Test size        : {tcfg.get('data', {}).get('test_size')}")
        print(f"    Target column    : {tcfg.get('features', {}).get('target_column')}")
        print(f"    Drop columns     : {tcfg.get('features', {}).get('drop_columns')}")
        print(f"    Selection metric : {tcfg.get('models', {}).get('selection_metric')}")
        print(f"    Enabled models   : {list(tcfg.get('models', {}).get('enabled', {}).keys())}")
        print(f"    Artifacts dir    : {tcfg.get('output', {}).get('artifacts_dir')}")
        print("-" * 72)
        print("  Prediction:")
        pcfg = cfg.get("prediction", {})
        print(f"    Artifacts dir    : {pcfg.get('default_artifacts_dir')}")
        print(f"    Lazy load        : {pcfg.get('enable_lazy_load')}")
        print(f"    Risk thresholds  : {pcfg.get('risk_thresholds')}")
        print("-" * 72)
        print("  Server:")
        scfg = cfg.get("server", {})
        print(f"    Flask host:port  : {scfg.get('flask_host')}:{scfg.get('flask_port')}")
        print(f"    Flask debug      : {scfg.get('flask_debug')}")
        print(f"    Streamlit port   : {scfg.get('streamlit_port')}")
        print(f"    Enable CORS      : {scfg.get('enable_cors')}")
        print(f"    CORS origins     : {scfg.get('cors_origins')}")
        print("-" * 72)
        print("  Logging:")
        lcfg = cfg.get("logging", {})
        print(f"    Level            : {lcfg.get('level')}")
        print(f"    Log dir          : {lcfg.get('log_dir')}")
        print(f"    Log to file      : {lcfg.get('log_to_file')}")
        print(f"    Log to console   : {lcfg.get('log_to_console')}")
        print("=" * 72 + "\n")


_config_service_singleton: Optional[ConfigService] = None


def get_config_service(
    config_path: Optional[str] = None,
    profile: Optional[Union[str, ConfigProfile]] = None,
    project_root: Optional[str] = None,
    force_reload: bool = False,
) -> ConfigService:
    global _config_service_singleton
    if _config_service_singleton is None or force_reload:
        _config_service_singleton = ConfigService(
            config_path=config_path,
            profile=profile,
            project_root=project_root,
        )
    return _config_service_singleton


def reset_config_service() -> None:
    global _config_service_singleton
    _config_service_singleton = None
