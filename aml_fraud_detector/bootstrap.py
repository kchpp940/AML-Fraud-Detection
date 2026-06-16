import os
import sys
import json
import hashlib
import time
import dataclasses
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from aml_fraud_detector.exception import (
    AMLException,
    ConfigException,
    MetadataValidationException,
    ModelLoadingException,
    wrap_exception,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.logger import logging
from aml_fraud_detector.configuration import TrainingConfig
from aml_fraud_detector.pipeline.prediction_pipeline import PredictionPipeline
from aml_fraud_detector.entity import ModelVersionInfo, FOUR_ARTIFACT_FILENAMES


class HealthStatusEnum(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"


class ComponentHealthStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ComponentHealth:
    name: str
    status: ComponentHealthStatus
    message: str = ""
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class HealthStatus:
    overall: HealthStatusEnum = HealthStatusEnum.INITIALIZING
    started_at: str = ""
    checked_at: str = ""
    components: List[ComponentHealth] = dataclasses.field(default_factory=list)
    model_version: ModelVersionInfo = dataclasses.field(default_factory=ModelVersionInfo)
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall.value,
            "started_at": self.started_at,
            "checked_at": self.checked_at,
            "components": [c.to_dict() for c in self.components],
            "model_version": asdict(self.model_version),
            "error_message": self.error_message,
        }

    def is_healthy(self) -> bool:
        return self.overall == HealthStatusEnum.HEALTHY

    def get_component(self, name: str) -> Optional[ComponentHealth]:
        for c in self.components:
            if c.name == name:
                return c
        return None


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
        self._artifacts_dir = artifacts_dir
        self._mode = mode
        self._health = HealthStatus(started_at=self._started_at)

        self._config: Optional[TrainingConfig] = None
        self._prediction_pipeline: Optional[PredictionPipeline] = None

        self._initialized = True

        if eager_load:
            self.bootstrap()

    @classmethod
    def reset(cls):
        cls._instance = None

    @property
    def health(self) -> HealthStatus:
        return self._health

    @property
    def config(self) -> Optional[TrainingConfig]:
        return self._config

    @property
    def prediction_pipeline(self) -> Optional[PredictionPipeline]:
        return self._prediction_pipeline

    def bootstrap(self) -> HealthStatus:
        try:
            self._load_config()

            if self._mode == "training":
                self._validate_training_environment()
            else:
                self._validate_artifacts_dir()
                self._validate_artifact_manifest()
                self._validate_metadata()
                self._initialize_prediction_pipeline()

            self._compute_overall_health()

        except Exception as e:
            logging.error(f"Bootstrap failed: {e}")
            self._health.overall = HealthStatusEnum.UNHEALTHY
            self._health.error_message = str(e)

        self._health.checked_at = datetime.now().isoformat()
        return self._health

    def check_health(self) -> HealthStatus:
        self._health.checked_at = datetime.now().isoformat()
        return self._health

    def get_health_dict(self) -> Dict[str, Any]:
        return self.check_health().to_dict()

    def get_version_dict(self) -> Dict[str, Any]:
        mv = self._health.model_version
        return asdict(mv)

    def get_model_version(self) -> ModelVersionInfo:
        return self._health.model_version

    def _add_component(self, component: ComponentHealth) -> None:
        self._health.components.append(component)

    def _load_config(self) -> None:
        start = time.time()
        try:
            if self._config_path is None:
                self._config_path = os.path.join("config", "training_config.yaml")

            self._config = TrainingConfig(self._config_path)

            if self._artifacts_dir is None and self._config.output.artifacts_dir:
                self._artifacts_dir = self._config.output.artifacts_dir

            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="config_load",
                status=ComponentHealthStatus.OK,
                message="Configuration loaded successfully",
                details={"config_path": self._config_path},
                duration_ms=duration,
            ))

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="config_load",
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                details={"config_path": self._config_path},
                duration_ms=duration,
            ))
            raise

    def _validate_artifacts_dir(self) -> None:
        start = time.time()
        try:
            if self._artifacts_dir is None:
                self._artifacts_dir = os.path.join(os.getcwd(), "artifacts")

            if not os.path.isdir(self._artifacts_dir):
                raise FileNotFoundError(f"Artifacts directory not found: {self._artifacts_dir}")

            missing_files = []
            for filename in FOUR_ARTIFACT_FILENAMES:
                filepath = os.path.join(self._artifacts_dir, filename)
                if not os.path.isfile(filepath):
                    missing_files.append(filename)

            if missing_files:
                raise FileNotFoundError(
                    f"Missing required artifact files: {', '.join(missing_files)}"
                )

            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="artifacts_dir",
                status=ComponentHealthStatus.OK,
                message="Artifacts directory and required files present",
                details={
                    "artifacts_dir": self._artifacts_dir,
                    "required_files": list(FOUR_ARTIFACT_FILENAMES),
                },
                duration_ms=duration,
            ))

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="artifacts_dir",
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                details={"artifacts_dir": self._artifacts_dir},
                duration_ms=duration,
            ))
            raise

    def _validate_artifact_manifest(self) -> None:
        start = time.time()
        try:
            manifest_path = os.path.join(self._artifacts_dir, "artifact_manifest.json")
            if not os.path.isfile(manifest_path):
                self._add_component(ComponentHealth(
                    name="artifact_manifest",
                    status=ComponentHealthStatus.WARNING,
                    message="Manifest file not found, skipping integrity check",
                    details={"manifest_path": manifest_path},
                    duration_ms=(time.time() - start) * 1000,
                ))
                return

            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            artifacts = manifest.get("artifacts", {})
            if not artifacts:
                self._add_component(ComponentHealth(
                    name="artifact_manifest",
                    status=ComponentHealthStatus.WARNING,
                    message="Manifest has no artifact entries",
                    duration_ms=(time.time() - start) * 1000,
                ))
                return

            mismatches = []
            for filename, info in artifacts.items():
                expected_digest = info.get("digest", "")
                if expected_digest.startswith("sha256:"):
                    expected_digest = expected_digest[7:]

                filepath = os.path.join(self._artifacts_dir, filename)
                if not os.path.isfile(filepath):
                    mismatches.append(f"{filename}: file not found")
                    continue

                actual_digest = self._sha256_file(filepath)
                if actual_digest.startswith("sha256:"):
                    actual_digest = actual_digest[7:]

                if expected_digest != actual_digest:
                    mismatches.append(
                        f"{filename}: digest mismatch (expected={expected_digest[:16]}..., actual={actual_digest[:16]}...)"
                    )

            if mismatches:
                raise MetadataValidationException(
                    ErrorCode.MANIFEST_INTEGRITY_FAILED,
                    detail="; ".join(mismatches),
                )

            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="artifact_manifest",
                status=ComponentHealthStatus.OK,
                message=f"All {len(artifacts)} artifacts passed SHA256 integrity check",
                details={"files_checked": list(artifacts.keys())},
                duration_ms=duration,
            ))

        except AMLException as e:
            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="artifact_manifest",
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                details={"error_code": e.error_code.value},
                duration_ms=duration,
            ))
            raise

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="artifact_manifest",
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                duration_ms=duration,
            ))
            raise

    def _validate_metadata(self) -> None:
        start = time.time()
        try:
            model_metadata_path = os.path.join(self._artifacts_dir, "model_metadata.json")
            with open(model_metadata_path, "r") as f:
                model_meta = json.load(f)

            feature_metadata_path = os.path.join(self._artifacts_dir, "feature_metadata.json")
            with open(feature_metadata_path, "r") as f:
                feature_meta = json.load(f)

            model_version = model_meta.get("model_version", 0)
            model_name = model_meta.get("model_name", "")
            training_time = model_meta.get("training_time", "")
            selection_metric = model_meta.get("selection_metric", "")
            best_metric_value = model_meta.get("best_metric_value", 0.0)
            feature_schema_version = feature_meta.get("schema_version", "")

            self._health.model_version = ModelVersionInfo(
                model_version=model_version,
                model_name=model_name,
                training_time=training_time,
                selection_metric=selection_metric,
                best_metric_value=best_metric_value,
                feature_schema_version=feature_schema_version,
                artifact_path=self._artifacts_dir,
            )

            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="metadata",
                status=ComponentHealthStatus.OK,
                message=f"Metadata validated (model v{model_version})",
                details={
                    "model_version": model_version,
                    "model_name": model_name,
                },
                duration_ms=duration,
            ))

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="metadata",
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                duration_ms=duration,
            ))
            raise

    def _initialize_prediction_pipeline(self) -> None:
        previous_errors = any(
            c.status == ComponentHealthStatus.ERROR
            for c in self._health.components
        )
        if previous_errors:
            self._add_component(ComponentHealth(
                name="prediction_pipeline",
                status=ComponentHealthStatus.SKIPPED,
                message="Skipped: previous errors detected",
            ))
            return

        start = time.time()
        try:
            self._prediction_pipeline = PredictionPipeline(
                artifacts_dir=self._artifacts_dir
            )

            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="prediction_pipeline",
                status=ComponentHealthStatus.OK,
                message="Prediction pipeline initialized successfully",
                details={"artifacts_dir": self._artifacts_dir},
                duration_ms=duration,
            ))

        except Exception as e:
            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="prediction_pipeline",
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                duration_ms=duration,
            ))
            raise

    def _validate_training_environment(self) -> None:
        start = time.time()
        try:
            if self._config is None:
                raise ConfigException(ErrorCode.CONFIG_MISSING_REQUIRED, key="config")

            training_data_path = self._config.training_data_path
            if not os.path.isfile(training_data_path):
                raise ConfigException(
                    ErrorCode.DATA_SOURCE_NOT_FOUND,
                    path=training_data_path,
                )

            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="training_env",
                status=ComponentHealthStatus.OK,
                message="Training environment validated",
                details={
                    "config_path": self._config_path,
                    "training_data": training_data_path,
                },
                duration_ms=duration,
            ))

        except AMLException as e:
            duration = (time.time() - start) * 1000
            self._add_component(ComponentHealth(
                name="training_env",
                status=ComponentHealthStatus.ERROR,
                message=str(e),
                details={"error_code": e.error_code.value},
                duration_ms=duration,
            ))
            raise

    def _compute_overall_health(self) -> None:
        components = self._health.components
        if not components:
            self._health.overall = HealthStatusEnum.INITIALIZING
            return

        error_count = sum(1 for c in components if c.status == ComponentHealthStatus.ERROR)
        warning_count = sum(1 for c in components if c.status == ComponentHealthStatus.WARNING)
        skipped_count = sum(1 for c in components if c.status == ComponentHealthStatus.SKIPPED)

        if error_count > 0:
            self._health.overall = HealthStatusEnum.UNHEALTHY
        elif warning_count > 0 or skipped_count > 0:
            self._health.overall = HealthStatusEnum.DEGRADED
        else:
            self._health.overall = HealthStatusEnum.HEALTHY

    @staticmethod
    def _sha256_file(filepath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"
