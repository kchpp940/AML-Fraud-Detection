import json
import os
import sys
from typing import Dict, List, Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.inference.contracts import ValidationReport


DEFAULT_ARTIFACTS_DIR = os.path.join("artifacts")
MODEL_FILENAME = "model.pkl"
PREPROCESSOR_FILENAME = "preprocessor.pkl"
MODEL_METADATA_FILENAME = "model_metadata.json"
FEATURE_METADATA_FILENAME = "feature_metadata.json"

FOUR_ARTIFACT_FILENAMES = [
    MODEL_FILENAME,
    PREPROCESSOR_FILENAME,
    FEATURE_METADATA_FILENAME,
    MODEL_METADATA_FILENAME,
]


class ArtifactValidator:
    def __init__(self, artifacts_dir: Optional[str] = None):
        self.artifacts_dir = artifacts_dir or DEFAULT_ARTIFACTS_DIR
        self.model_path = os.path.join(self.artifacts_dir, MODEL_FILENAME)
        self.preprocessor_path = os.path.join(self.artifacts_dir, PREPROCESSOR_FILENAME)
        self.model_metadata_path = os.path.join(self.artifacts_dir, MODEL_METADATA_FILENAME)
        self.feature_metadata_path = os.path.join(self.artifacts_dir, FEATURE_METADATA_FILENAME)

    def _read_json(self, path: str) -> Dict:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise CustomerException(FileNotFoundError(f"Required file not found: {path}"), sys)
        except json.JSONDecodeError as e:
            raise CustomerException(ValueError(f"Invalid JSON in {path}: {e}"), sys)

    def _validate_four_artifacts_exist(self, errors: List[str]) -> None:
        expected = {
            MODEL_FILENAME: self.model_path,
            PREPROCESSOR_FILENAME: self.preprocessor_path,
            FEATURE_METADATA_FILENAME: self.feature_metadata_path,
            MODEL_METADATA_FILENAME: self.model_metadata_path,
        }
        for name, path in expected.items():
            if not os.path.isfile(path):
                errors.append(f"Missing required artifact: {name} at {path}")
            elif os.path.getsize(path) == 0:
                errors.append(f"Artifact file is empty: {name} at {path}")

    def _validate_model_metadata(self, model_meta: Dict, errors: List[str], details: Dict) -> None:
        required_keys = ["model_version", "best_model_name", "artifact_path"]
        for k in required_keys:
            if k not in model_meta:
                errors.append(f"model_metadata.json missing required key: '{k}'")
        details["model_version"] = model_meta.get("model_version")
        details["best_model_name"] = model_meta.get("best_model_name")

    def _validate_feature_metadata(self, feature_meta: Dict, errors: List[str], details: Dict) -> None:
        required_sections = ["numerical_features", "categorical_features", "encoding_info"]
        for section in required_sections:
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

        logging.info(f"Validating artifacts (4 required files) in: {self.artifacts_dir}")

        self._validate_four_artifacts_exist(errors)
        if errors:
            return ValidationReport(
                is_valid=False, errors=errors, warnings=warnings, details=details
            )

        try:
            model_meta = self._read_json(self.model_metadata_path)
            feature_meta = self._read_json(self.feature_metadata_path)
            self._validate_model_metadata(model_meta, errors, details)
            self._validate_feature_metadata(feature_meta, errors, details)
        except CustomerException:
            raise
        except Exception as e:
            errors.append(f"Failed to parse metadata files: {e}")

        is_valid = len(errors) == 0
        if is_valid:
            logging.info("Artifact validation passed (4 files checked)")
        else:
            logging.warning(f"Artifact validation failed with {len(errors)} error(s)")
        return ValidationReport(
            is_valid=is_valid, errors=errors, warnings=warnings, details=details
        )
