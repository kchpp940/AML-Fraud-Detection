import os
import sys
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from aml_fraud_detector.exception import (
    AMLException,
    MetadataValidationException,
    wrap_exception,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.logger import logging


REQUIRED_MANIFEST_ARTIFACTS = [
    "model.pkl",
    "preprocessor.pkl",
    "feature_metadata.json",
    "model_metadata.json",
]

MANIFEST_SCHEMA_VERSION = "1.0"


def _sha256_file(file_path: str, chunk_size: int = 65536) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


@dataclass
class ArtifactValidationResult:
    is_valid: bool = True
    checked_artifacts: List[str] = field(default_factory=list)
    missing_artifacts: List[str] = field(default_factory=list)
    mismatched_digests: List[Dict[str, Any]] = field(default_factory=list)
    manifest_missing: bool = False
    manifest_corrupted: bool = False
    schema_version_mismatch: Optional[Dict[str, Any]] = None
    error_code: Optional[ErrorCode] = None
    error_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.error_code:
            result["error_code"] = self.error_code.value
        return result

    def first_error_code(self) -> ErrorCode:
        if self.error_code:
            return self.error_code
        if self.manifest_missing:
            return ErrorCode.METADATA_FILE_NOT_FOUND
        if self.manifest_corrupted:
            return ErrorCode.METADATA_CORRUPTED
        if self.schema_version_mismatch:
            return ErrorCode.METADATA_VERSION_MISMATCH
        if self.missing_artifacts:
            return ErrorCode.ARTIFACT_MISSING
        if self.mismatched_digests:
            return ErrorCode.MANIFEST_INTEGRITY_FAILED
        return ErrorCode.MANIFEST_INTEGRITY_FAILED

    def first_error_message(self) -> str:
        if self.error_reason:
            return self.error_reason
        if self.manifest_missing:
            return f"Manifest 文件未找到，无法进行产物校验"
        if self.manifest_corrupted:
            return f"Manifest 文件损坏，无法解析 JSON"
        if self.schema_version_mismatch:
            return (
                f"Manifest schema 版本不匹配，期望 {self.schema_version_mismatch['expected']}，"
                f"实际 {self.schema_version_mismatch['actual']}"
            )
        if self.missing_artifacts:
            return f"缺少必需产物文件: {', '.join(self.missing_artifacts)}"
        if self.mismatched_digests:
            names = [m["name"] for m in self.mismatched_digests]
            return f"产物完整性校验失败（哈希不匹配）: {', '.join(names)}"
        return "产物校验未通过"


def build_manifest(artifacts_dir: str) -> Dict[str, Any]:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifacts_dir": os.path.abspath(artifacts_dir),
        "artifacts": {},
        "generated_at": _now_iso(),
    }
    for fname in sorted(os.listdir(artifacts_dir)):
        fpath = os.path.join(artifacts_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            stat = os.stat(fpath)
            digest = "sha256:" + _sha256_file(fpath)
            manifest["artifacts"][fname] = {
                "path": os.path.abspath(fpath),
                "size": stat.st_size,
                "mtime_iso": _iso_from_timestamp(stat.st_mtime),
                "digest": digest,
            }
        except Exception as e:
            logging.warning(f"Failed to build manifest entry for {fname}: {e}")
    return manifest


def save_manifest(manifest_path: str, artifacts_dir: str) -> Dict[str, Any]:
    manifest = build_manifest(artifacts_dir)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logging.info(f"Manifest written to {manifest_path}")
    return manifest


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat()


def _iso_from_timestamp(ts: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(ts).isoformat()


class ArtifactValidator:
    def __init__(self, artifacts_dir: str, manifest_filename: str = "artifact_manifest.json"):
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        self.manifest_path = os.path.join(self.artifacts_dir, manifest_filename)

    def _load_manifest(self) -> Tuple[Optional[Dict[str, Any]], ArtifactValidationResult]:
        result = ArtifactValidationResult()
        if not os.path.exists(self.manifest_path):
            result.manifest_missing = True
            result.is_valid = False
            result.error_code = ErrorCode.METADATA_FILE_NOT_FOUND
            result.error_reason = f"Manifest 文件未找到: {self.manifest_path}"
            return None, result
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            result.manifest_corrupted = True
            result.is_valid = False
            result.error_code = ErrorCode.METADATA_CORRUPTED
            result.error_reason = f"Manifest 文件损坏: {self.manifest_path}, 详情: {e}"
            return None, result

        actual_schema = manifest.get("schema_version", "0.0")
        if actual_schema != MANIFEST_SCHEMA_VERSION:
            result.schema_version_mismatch = {
                "expected": MANIFEST_SCHEMA_VERSION,
                "actual": actual_schema,
            }
            result.is_valid = False
            result.error_code = ErrorCode.METADATA_VERSION_MISMATCH

        return manifest, result

    def validate(
        self,
        required_artifacts: Optional[List[str]] = None,
        verify_digests: bool = True,
    ) -> ArtifactValidationResult:
        required = required_artifacts or REQUIRED_MANIFEST_ARTIFACTS

        manifest, result = self._load_manifest()
        if not result.is_valid:
            return result

        manifest_artifacts = manifest.get("artifacts", {}) if manifest else {}

        for artifact_name in required:
            fpath = os.path.join(self.artifacts_dir, artifact_name)
            result.checked_artifacts.append(artifact_name)

            if not os.path.exists(fpath):
                result.missing_artifacts.append(artifact_name)
                result.is_valid = False
                result.error_code = ErrorCode.ARTIFACT_MISSING
                result.error_reason = f"缺少必需产物文件: {artifact_name} ({fpath})"
                continue

            if verify_digests and artifact_name in manifest_artifacts:
                expected_digest = manifest_artifacts[artifact_name].get("digest", "")
                try:
                    actual_digest = "sha256:" + _sha256_file(fpath)
                except Exception as e:
                    raise wrap_exception(e, error_details=sys)

                if expected_digest and actual_digest != expected_digest:
                    result.mismatched_digests.append({
                        "name": artifact_name,
                        "expected": expected_digest,
                        "actual": actual_digest,
                    })
                    result.is_valid = False
                    result.error_code = ErrorCode.MANIFEST_INTEGRITY_FAILED
                    result.error_reason = (
                        f"产物 {artifact_name} 哈希校验失败，"
                        f"期望 {expected_digest}，实际 {actual_digest}"
                    )

        if not result.is_valid:
            logging.warning(
                f"Artifact validation failed for {self.artifacts_dir}: "
                f"missing={result.missing_artifacts}, "
                f"digest_mismatch={[m['name'] for m in result.mismatched_digests]}"
            )
        else:
            logging.info(
                f"Artifact validation passed for {self.artifacts_dir}, "
                f"checked={len(result.checked_artifacts)} artifacts"
            )
        return result

    def raise_if_invalid(
        self,
        required_artifacts: Optional[List[str]] = None,
        verify_digests: bool = True,
    ) -> None:
        result = self.validate(
            required_artifacts=required_artifacts,
            verify_digests=verify_digests,
        )
        if result.is_valid:
            return
        raise MetadataValidationException(
            result.first_error_code(),
            error_details=sys,
            detail=result.first_error_message(),
        )
