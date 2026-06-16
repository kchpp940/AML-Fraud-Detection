import os
import sys
import json
import hashlib
import time
from dataclasses import asdict
from datetime import datetime
from typing import Optional, Dict, Any

from aml_fraud_detector.exception import (
    AMLException,
    ConfigException,
    MetadataValidationException,
    ModelLoadingException,
    wrap_exception,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.entity import (
    FOUR_ARTIFACT_FILENAMES,
    HealthStatus,
    HealthStatusEnum,
    ComponentHealth,
    ComponentHealthStatus,
    ModelVersionInfo,
)
from aml_fraud_detector.logger import logging
from aml_fraud_detector.configuration import TrainingConfig
from aml_fraud_detector.pipeline.prediction_pipeline import PredictionPipeline


class ApplicationContainer:
    _instance: Optional["ApplicationContainer"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        config_path: Optional[str] = None,
        artifacts_dir: Optional[str] = None,
        eager_load: bool = True,
        mode: str = "inference",
    ):
        if self._initialized:
            return

        self._started_at = datetime.now().isoformat()
        self._config_path = config_path
        self._artifacts_dir_override = artifacts_dir
        self._eager_load = eager_load
        self._mode = mode

        self.config: Optional[TrainingConfig] = None
        self.artifacts_dir: str = ""
        self.prediction_pipeline: Optional[PredictionPipeline] = None

        self.health: HealthStatus = HealthStatus(started_at=self._started_at)

        self._initialized = True
        logging.info(f"ApplicationContainer created in {mode} mode")

        if eager_load:
            self.bootstrap()

    @classmethod
    def get_instance(cls) -> Optional["ApplicationContainer"]:
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def bootstrap(self) -> HealthStatus:
        logging.info("=" * 72)
        logging.info(f"Application bootstrap started in {self._mode} mode")
        logging.info("=" * 72)

        self.health = HealthStatus(started_at=self._started_at)
        self.health.overall = HealthStatusEnum.INITIALIZING

        self._load_config()

        if self._mode == "training":
            self._validate_training_environment()
        else:
            self._validate_artifacts_dir()
            self._validate_artifact_manifest()
            self._validate_metadata()
            self._initialize_prediction_pipeline()

        self._compute_overall_health()

        self.health.checked_at = datetime.now().isoformat()

        if self.health.is_healthy():
            logging.info("=" * 72)
            logging.info("Application bootstrap completed successfully")
            logging.info("=" * 72)
        else:
            logging.error(
                f"Application bootstrap completed with status: {self.health.overall.value}"
            )
            for comp in self.health.components:
                if comp.status == ComponentHealthStatus.ERROR:
                    logging.error(f"  - {comp.name}: ERROR - {comp.message}")
                elif comp.status == ComponentHealthStatus.WARNING:
                    logging.warning(f"  - {comp.name}: WARNING - {comp.message}")

        return self.health

    def _validate_training_environment(self) -> None:
        name = "training_environment"
        if not self.config:
            self._add_component(
                name=name,
                status=ComponentHealthStatus.SKIPPED,
                message="Skipped: config not loaded",
            )
            return

        try:
            def _do_check():
                details = {
                    "mode": self._mode,
                    "artifacts_dir": self.artifacts_dir,
                    "data_source": self.config.resolve_source_path(),
                    "target_column": self.config.features.target_column,
                    "selection_metric": self.config.models.selection_metric,
                }

                data_source = self.config.resolve_source_path()
                if not os.path.isfile(data_source):
                    raise ConfigException(
                        ErrorCode.DATA_SOURCE_NOT_FOUND,
                        error_details=sys,
                        path=data_source,
                    )
                details["data_source_exists"] = True

                os.makedirs(self.artifacts_dir, exist_ok=True)
                details["artifacts_dir_created"] = True

                return details

            details, duration_ms = self._measure_time(_do_check)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.OK,
                message="Training environment validated successfully",
                details=details,
                duration_ms=duration_ms,
            )
        except AMLException as e:
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                details={"error_code": e.error_code.value},
            )
        except Exception as e:
            wrapped = wrap_exception(e, error_details=sys)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(wrapped),
                details={"error_code": wrapped.error_code.value},
            )

    def _measure_time(self, func, *args, **kwargs) -> tuple:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration_ms = (time.perf_counter() - start) * 1000
        return result, duration_ms

    def _add_component(
        self,
        name: str,
        status: ComponentHealthStatus,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
    ) -> None:
        component = ComponentHealth(
            name=name,
            status=status,
            message=message,
            details=details or {},
            duration_ms=duration_ms,
        )
        self.health.components.append(component)

    def _load_config(self) -> None:
        name = "config_load"
        try:
            def _do_load():
                self.config = TrainingConfig(config_path=self._config_path)
                if self._artifacts_dir_override:
                    self.config.output.artifacts_dir = self._artifacts_dir_override
                self.artifacts_dir = os.path.abspath(self.config.output.artifacts_dir)
                return {
                    "config_path": self.config.config_path,
                    "config_source": self.config.config_path_source,
                    "yaml_used": self.config.yaml_config_used,
                    "artifacts_dir": self.artifacts_dir,
                }

            details, duration_ms = self._measure_time(_do_load)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.OK,
                message="Configuration loaded successfully",
                details=details,
                duration_ms=duration_ms,
            )
        except AMLException as e:
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                details={"error_code": e.error_code.value},
            )
            self.health.error_detail = e.error_detail
        except Exception as e:
            wrapped = wrap_exception(e, error_details=sys)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(wrapped),
                details={"error_code": wrapped.error_code.value},
            )
            self.health.error_detail = wrapped.error_detail

    def _validate_artifacts_dir(self) -> None:
        name = "artifacts_dir"
        if not self.config:
            self._add_component(
                name=name,
                status=ComponentHealthStatus.SKIPPED,
                message="Skipped: config not loaded",
            )
            return

        try:
            def _do_check():
                if not os.path.isdir(self.artifacts_dir):
                    raise MetadataValidationException(
                        ErrorCode.ARTIFACT_MISSING,
                        error_details=sys,
                        artifact=f"artifacts directory: {self.artifacts_dir}",
                    )

                missing = []
                for filename in FOUR_ARTIFACT_FILENAMES:
                    path = os.path.join(self.artifacts_dir, filename)
                    if not os.path.isfile(path):
                        missing.append(filename)

                if missing:
                    raise MetadataValidationException(
                        ErrorCode.ARTIFACT_MISSING,
                        error_details=sys,
                        artifact=", ".join(missing),
                    )

                return {
                    "artifacts_dir": self.artifacts_dir,
                    "required_files": FOUR_ARTIFACT_FILENAMES,
                }

            details, duration_ms = self._measure_time(_do_check)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.OK,
                message="Artifacts directory and required files present",
                details=details,
                duration_ms=duration_ms,
            )
        except AMLException as e:
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                details={"error_code": e.error_code.value},
            )
        except Exception as e:
            wrapped = wrap_exception(e, error_details=sys)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(wrapped),
                details={"error_code": wrapped.error_code.value},
            )

    def _validate_artifact_manifest(self) -> None:
        name = "artifact_manifest"
        manifest_path = os.path.join(self.artifacts_dir, "artifact_manifest.json")

        if not os.path.isfile(manifest_path):
            self._add_component(
                name=name,
                status=ComponentHealthStatus.WARNING,
                message="artifact_manifest.json not found, skipping integrity check",
                details={"manifest_path": manifest_path},
            )
            return

        try:
            def _do_validate():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

                details = {
                    "manifest_path": os.path.abspath(manifest_path),
                    "generated_at": manifest.get("generated_at", ""),
                    "artifacts_count": len(manifest.get("artifacts", {})),
                }

                errors = []
                warnings = []
                checked = 0
                for filename, info in manifest.get("artifacts", {}).items():
                    filepath = os.path.join(self.artifacts_dir, filename)
                    if not os.path.isfile(filepath):
                        errors.append(f"{filename}: file missing")
                        continue
                    checked += 1

                    expected_digest = info.get("digest", "")
                    if expected_digest.startswith("sha256:"):
                        actual_digest = self._sha256_file(filepath)
                        if actual_digest != expected_digest:
                            errors.append(
                                f"{filename}: digest mismatch (expected={expected_digest}, actual={actual_digest})"
                            )

                details["files_checked"] = checked
                details["errors"] = errors
                details["warnings"] = warnings

                if errors:
                    raise MetadataValidationException(
                        ErrorCode.MANIFEST_INTEGRITY_FAILED,
                        error_details=sys,
                        detail="; ".join(errors),
                    )

                return details

            details, duration_ms = self._measure_time(_do_validate)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.OK,
                message=f"Manifest integrity verified ({details['files_checked']} files)",
                details=details,
                duration_ms=duration_ms,
            )
        except AMLException as e:
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                details={"error_code": e.error_code.value},
            )
        except Exception as e:
            wrapped = wrap_exception(e, error_details=sys)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(wrapped),
                details={"error_code": wrapped.error_code.value},
            )

    @staticmethod
    def _sha256_file(filepath: str, chunk_size: int = 8192) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    def _validate_metadata(self) -> None:
        name = "metadata"
        feature_meta_path = os.path.join(self.artifacts_dir, "feature_metadata.json")
        model_meta_path = os.path.join(self.artifacts_dir, "model_metadata.json")

        try:
            def _do_validate():
                details = {}

                if not os.path.isfile(feature_meta_path):
                    raise MetadataValidationException(
                        ErrorCode.METADATA_FILE_NOT_FOUND,
                        error_details=sys,
                        path=feature_meta_path,
                    )

                if not os.path.isfile(model_meta_path):
                    raise MetadataValidationException(
                        ErrorCode.METADATA_FILE_NOT_FOUND,
                        error_details=sys,
                        path=model_meta_path,
                    )

                with open(feature_meta_path, "r", encoding="utf-8") as f:
                    feature_meta = json.load(f)

                with open(model_meta_path, "r", encoding="utf-8") as f:
                    model_meta = json.load(f)

                required_model_fields = [
                    "model_version",
                    "training_time",
                    "best_model_name",
                    "selection_metric",
                    "best_metric_value",
                ]
                missing = [f for f in required_model_fields if f not in model_meta]
                if missing:
                    raise MetadataValidationException(
                        ErrorCode.METADATA_SCHEMA_MISMATCH,
                        error_details=sys,
                        missing=", ".join(missing),
                    )

                version_info = ModelVersionInfo(
                    model_version=int(model_meta.get("model_version", 0)),
                    model_name=str(model_meta.get("best_model_name", "")),
                    training_time=str(model_meta.get("training_time", "")),
                    selection_metric=str(model_meta.get("selection_metric", "")),
                    best_metric_value=float(model_meta.get("best_metric_value", 0.0)),
                    feature_schema_version=str(
                        model_meta.get("feature_schema_version",
                                       feature_meta.get("contract_version", ""))
                    ),
                    artifact_path=os.path.abspath(self.artifacts_dir),
                )
                self.health.model_version = version_info

                details["model_version"] = version_info.model_version
                details["model_name"] = version_info.model_name
                details["selection_metric"] = version_info.selection_metric
                details["best_metric_value"] = version_info.best_metric_value
                details["feature_schema_version"] = version_info.feature_schema_version
                details["training_time"] = version_info.training_time

                return details

            details, duration_ms = self._measure_time(_do_validate)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.OK,
                message=f"Metadata validated (model v{details['model_version']})",
                details=details,
                duration_ms=duration_ms,
            )
        except AMLException as e:
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                details={"error_code": e.error_code.value},
            )
        except Exception as e:
            wrapped = wrap_exception(e, error_details=sys)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(wrapped),
                details={"error_code": wrapped.error_code.value},
            )

    def _initialize_prediction_pipeline(self) -> None:
        name = "prediction_pipeline"

        has_error = any(
            c.status == ComponentHealthStatus.ERROR
            for c in self.health.components
        )
        if has_error:
            self._add_component(
                name=name,
                status=ComponentHealthStatus.SKIPPED,
                message="Skipped: previous errors detected",
            )
            return

        try:
            def _do_init():
                self.prediction_pipeline = PredictionPipeline(
                    artifacts_dir=self.artifacts_dir
                )
                self.prediction_pipeline._load_artifacts()

                model_type = type(self.prediction_pipeline._model).__name__
                preprocessor_type = type(self.prediction_pipeline._preprocessor).__name__

                return {
                    "model_type": model_type,
                    "preprocessor_type": preprocessor_type,
                    "artifacts_dir": self.artifacts_dir,
                }

            details, duration_ms = self._measure_time(_do_init)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.OK,
                message=f"PredictionPipeline initialized (model={details['model_type']})",
                details=details,
                duration_ms=duration_ms,
            )
        except AMLException as e:
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                details={"error_code": e.error_code.value},
            )
        except Exception as e:
            wrapped = wrap_exception(e, error_details=sys)
            self._add_component(
                name=name,
                status=ComponentHealthStatus.ERROR,
                message=str(wrapped),
                details={"error_code": wrapped.error_code.value},
            )

    def _compute_overall_health(self) -> None:
        error_count = 0
        warning_count = 0

        for comp in self.health.components:
            if comp.status == ComponentHealthStatus.ERROR:
                error_count += 1
            elif comp.status == ComponentHealthStatus.WARNING:
                warning_count += 1

        if error_count > 0:
            self.health.overall = HealthStatusEnum.UNHEALTHY
        elif warning_count > 0:
            self.health.overall = HealthStatusEnum.DEGRADED
        elif len(self.health.components) == 0:
            self.health.overall = HealthStatusEnum.INITIALIZING
        else:
            self.health.overall = HealthStatusEnum.HEALTHY

    def check_health(self) -> HealthStatus:
        if not self._initialized or self.health.overall == HealthStatusEnum.INITIALIZING:
            self.bootstrap()
        else:
            self.health.checked_at = datetime.now().isoformat()
        return self.health

    def get_model_version(self) -> ModelVersionInfo:
        return self.health.model_version

    def get_health_dict(self) -> Dict[str, Any]:
        return self.check_health().to_dict()

    def get_version_dict(self) -> Dict[str, Any]:
        version_info = self.get_model_version()
        result = asdict(version_info)
        result["healthy"] = self.check_health().is_healthy()
        return result
