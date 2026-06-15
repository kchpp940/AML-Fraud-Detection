import json
import os
import sys
import threading
from typing import Any, Dict, Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object
from aml_fraud_detector.inference.contracts import ModelArtifacts, ValidationReport
from aml_fraud_detector.inference.artifact_validator import (
    ArtifactValidator,
    DEFAULT_ARTIFACTS_DIR,
    MANIFEST_FILENAME,
    MODEL_METADATA_FILENAME,
    FEATURE_METADATA_FILENAME,
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
)


class ModelLoader:
    _instance: Optional["ModelLoader"] = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self, artifacts_dir: Optional[str] = None, validate: bool = True):
        self.artifacts_dir = artifacts_dir or DEFAULT_ARTIFACTS_DIR
        self._validate_on_load = validate
        self._lock = threading.RLock()
        self._cached_artifacts: Optional[ModelArtifacts] = None
        self._last_validation: Optional[ValidationReport] = None

    @classmethod
    def get_instance(cls, artifacts_dir: Optional[str] = None, validate: bool = True) -> "ModelLoader":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(artifacts_dir=artifacts_dir, validate=validate)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._instance_lock:
            cls._instance = None

    def _read_json_safe(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"Metadata file not found: {path}, using empty dict")
            return {}
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid JSON in {path}: {e}, using empty dict")
            return {}

    def _do_validate(self) -> ValidationReport:
        validator = ArtifactValidator(artifacts_dir=self.artifacts_dir)
        report = validator.validate(strict=False)
        for w in report.warnings:
            logging.warning(f"[ArtifactValidator] {w}")
        return report

    def _load_artifacts_uncached(self) -> ModelArtifacts:
        logging.info(f"Loading model artifacts from disk: {self.artifacts_dir}")
        if self._validate_on_load:
            report = self._do_validate()
            self._last_validation = report
            report.raise_if_invalid()

        model_path = os.path.join(self.artifacts_dir, MODEL_FILENAME)
        preprocessor_path = os.path.join(self.artifacts_dir, PREPROCESSOR_FILENAME)
        model_meta_path = os.path.join(self.artifacts_dir, MODEL_METADATA_FILENAME)
        feature_meta_path = os.path.join(self.artifacts_dir, FEATURE_METADATA_FILENAME)
        manifest_path = os.path.join(self.artifacts_dir, MANIFEST_FILENAME)

        try:
            model = load_object(file_path=model_path)
        except Exception as e:
            raise CustomerException(RuntimeError(f"Failed to load model from {model_path}: {e}"), sys)
        try:
            preprocessor = load_object(file_path=preprocessor_path)
        except Exception as e:
            raise CustomerException(RuntimeError(f"Failed to load preprocessor from {preprocessor_path}: {e}"), sys)

        model_metadata = self._read_json_safe(model_meta_path)
        feature_metadata = self._read_json_safe(feature_meta_path)
        manifest = self._read_json_safe(manifest_path)

        artifacts = ModelArtifacts(
            model=model,
            preprocessor=preprocessor,
            model_metadata=model_metadata,
            feature_metadata=feature_metadata,
            manifest=manifest,
        )
        logging.info(
            "Artifacts loaded successfully: "
            f"model_type={type(model).__name__}, "
            f"preprocessor_type={type(preprocessor).__name__}, "
            f"model_version={model_metadata.get('model_version', 'unknown')}"
        )
        return artifacts

    def load(self, force_reload: bool = False) -> ModelArtifacts:
        with self._lock:
            if self._cached_artifacts is None or force_reload:
                self._cached_artifacts = self._load_artifacts_uncached()
            return self._cached_artifacts

    def is_loaded(self) -> bool:
        with self._lock:
            return self._cached_artifacts is not None

    def last_validation_report(self) -> Optional[ValidationReport]:
        with self._lock:
            return self._last_validation

    def unload(self) -> None:
        with self._lock:
            self._cached_artifacts = None
            logging.info("Model artifacts cache cleared")
