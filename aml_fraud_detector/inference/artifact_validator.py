import hashlib
import json
import os
import sys
from typing import Dict, List, Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.inference.contracts import ValidationReport


DEFAULT_ARTIFACTS_DIR = os.path.join("artifacts")
MANIFEST_FILENAME = "artifact_manifest.json"
MODEL_METADATA_FILENAME = "model_metadata.json"
FEATURE_METADATA_FILENAME = "feature_metadata.json"
MODEL_FILENAME = "model.pkl"
PREPROCESSOR_FILENAME = "preprocessor.pkl"

FOUR_ARTIFACT_FILENAMES = [
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
    FEATURE_METADATA_FILENAME,
    MODEL_METADATA_FILENAME,
]


class ArtifactValidator:
    def __init__(self, artifacts_dir: Optional[str] = None):
        self.artifacts_dir = artifacts_dir or DEFAULT_ARTIFACTS_DIR
        self.manifest_path = os.path.join(self.artifacts_dir, MANIFEST_FILENAME)
        self.model_metadata_path = os.path.join(self.artifacts_dir, MODEL_METADATA_FILENAME)
        self.feature_metadata_path = os.path.join(self.artifacts_dir, FEATURE_METADATA_FILENAME)
        self.model_path = os.path.join(self.artifacts_dir, MODEL_FILENAME)
        self.preprocessor_path = os.path.join(self.artifacts_dir, PREPROCESSOR_FILENAME)

    def _read_json(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise CustomerException(FileNotFoundError(f"Required file not found: {path}"), sys)
        except json.JSONDecodeError as e:
            raise CustomerException(ValueError(f"Invalid JSON in {path}: {e}"), sys)

    def _file_exists(self, path: str) -> bool:
        return os.path.isfile(path)

    def _file_size(self, path: str) -> int:
        return os.path.getsize(path) if self._file_exists(path) else 0

    def _sha256_digest(self, path: str, chunk_size: int = 8192) -> str:
        h = hashlib.new("sha256")
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"

    def _validate_four_artifacts_exist(self, errors: List[str]) -> None:
        expected = {
            MODEL_FILENAME: self.model_path,
            PREPROCESSOR_FILENAME: self.preprocessor_path,
            FEATURE_METADATA_FILENAME: self.feature_metadata_path,
            MODEL_METADATA_FILENAME: self.model_metadata_path,
        }
        for name, path in expected.items():
            if not self._file_exists(path):
                errors.append(f"Missing required artifact: {name} at {path}")
            elif self._file_size(path) == 0:
                errors.append(f"Artifact file is empty: {name} at {path}")

    def _validate_manifest(
        self, manifest: Dict, errors: List[str], warnings: List[str], details: Dict
    ) -> None:
        if "artifacts" not in manifest:
            errors.append("Manifest missing 'artifacts' section")
            return
        artifacts_section = manifest["artifacts"]
        details["manifest_artifacts"] = list(artifacts_section.keys())

        manifest_checks = {
            "model_pkl": self.model_path,
            "preprocessor_pkl": self.preprocessor_path,
            "feature_metadata_json": self.feature_metadata_path,
            "model_metadata_json": self.model_metadata_path,
        }
        for key, actual_path in manifest_checks.items():
            entry = artifacts_section.get(key)
            if not entry:
                warnings.append(f"Manifest has no entry for '{key}', skipping integrity check")
                continue
            manifest_path = entry.get("path")
            manifest_size = entry.get("size")
            manifest_digest = entry.get("digest")

            if manifest_path and os.path.abspath(manifest_path) != os.path.abspath(actual_path):
                warnings.append(
                    f"Manifest path mismatch for {key}: manifest={manifest_path}, actual={actual_path}"
                )

            if manifest_size is not None and self._file_exists(actual_path):
                actual_size = self._file_size(actual_path)
                if actual_size != manifest_size:
                    errors.append(
                        f"Size mismatch for {key}: manifest={manifest_size}, actual={actual_size}"
                    )

            if manifest_digest and self._file_exists(actual_path):
                try:
                    actual_digest = self._sha256_digest(actual_path)
                    details[f"digest_{key}"] = {
                        "manifest": manifest_digest,
                        "actual": actual_digest,
                    }
                    if actual_digest != manifest_digest:
                        errors.append(
                            f"Digest mismatch for {key}: manifest={manifest_digest}, actual={actual_digest}"
                        )
                except Exception as e:
                    warnings.append(f"Could not compute digest for {key}: {e}")

    def _validate_model_metadata(
        self, model_meta: Dict, errors: List[str], details: Dict
    ) -> None:
        required_keys = ["model_version", "best_model_name", "artifact_path"]
        for k in required_keys:
            if k not in model_meta:
                errors.append(f"model_metadata.json missing required key: '{k}'")
        details["model_version"] = model_meta.get("model_version")
        details["best_model_name"] = model_meta.get("best_model_name")
        details["data_file_digest"] = model_meta.get("data_file_digest")
        details["feature_schema_version"] = model_meta.get("feature_schema_version")

    def _validate_feature_metadata(
        self, feature_meta: Dict, errors: List[str], warnings: List[str], details: Dict
    ) -> None:
        if "contract_version" not in feature_meta:
            warnings.append("feature_metadata.json missing 'contract_version'")
        for section in ["numerical_features", "categorical_features", "encoding_info"]:
            if section not in feature_meta:
                errors.append(f"feature_metadata.json missing required section: '{section}'")
        details["feature_contract_version"] = feature_meta.get("contract_version")
        details["numerical_features"] = feature_meta.get("numerical_features", [])
        details["categorical_features"] = feature_meta.get("categorical_features", [])

    def validate(self, strict: bool = True) -> ValidationReport:
        errors: List[str] = []
        warnings: List[str] = []
        details: Dict[str, object] = {
            "artifacts_dir": os.path.abspath(self.artifacts_dir),
            "checked_artifacts": list(FOUR_ARTIFACT_FILENAMES),
        }

        logging.info(f"Validating artifacts in: {self.artifacts_dir}")

        self._validate_four_artifacts_exist(errors)
        if errors:
            return ValidationReport(
                is_valid=False, errors=errors, warnings=warnings, details=details
            )

        try:
            model_meta = self._read_json(self.model_metadata_path)
            feature_meta = self._read_json(self.feature_metadata_path)
            self._validate_model_metadata(model_meta, errors, details)
            self._validate_feature_metadata(feature_meta, errors, warnings, details)
        except CustomerException:
            raise
        except Exception as e:
            errors.append(f"Failed to parse metadata files: {e}")

        if self._file_exists(self.manifest_path):
            try:
                manifest = self._read_json(self.manifest_path)
                self._validate_manifest(manifest, errors, warnings, details)
                details["manifest_present"] = True
            except CustomerException:
                raise
            except Exception as e:
                warnings.append(f"Failed to parse manifest, skipping integrity checks: {e}")
                details["manifest_present"] = False
        else:
            if strict:
                warnings.append(
                    "artifact_manifest.json not found; running without integrity verification"
                )
            details["manifest_present"] = False

        is_valid = len(errors) == 0
        if is_valid:
            logging.info("Artifact validation passed")
        else:
            logging.warning(f"Artifact validation failed with {len(errors)} error(s)")
        return ValidationReport(
            is_valid=is_valid, errors=errors, warnings=warnings, details=details
        )
