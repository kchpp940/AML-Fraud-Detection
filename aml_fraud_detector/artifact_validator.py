from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional, Tuple

from aml_fraud_detector.entity import (
    FOUR_ARTIFACT_FILENAMES,
    ModelVersionInfo,
    ValidationStatus,
)


def _sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


class ArtifactValidator:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        self.required_files = list(FOUR_ARTIFACT_FILENAMES)

    def _resolve(self, filename: str) -> str:
        return os.path.join(self.artifacts_dir, filename)

    def _read_model_metadata(self) -> Optional[Dict]:
        path = self._resolve("model_metadata.json")
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _read_manifest_digests(self) -> Dict[str, str]:
        meta = self._read_model_metadata()
        if not meta:
            return {}
        digests = meta.get("artifact_digests") or {}
        if isinstance(digests, dict):
            return {str(k): str(v) for k, v in digests.items()}
        return {}

    def load_model_version_info(self) -> Optional[ModelVersionInfo]:
        meta = self._read_model_metadata()
        if not meta:
            return None
        try:
            return ModelVersionInfo(
                model_version=int(meta.get("model_version") or 0),
                model_name=str(meta.get("model_name") or ""),
                training_time=str(meta.get("training_time") or ""),
                selection_metric=str(meta.get("selection_metric") or ""),
                best_metric_value=float(meta.get("best_metric_value") or 0.0),
                feature_schema_version=str(meta.get("feature_schema_version") or ""),
                artifact_path=self.artifacts_dir,
            )
        except Exception:
            return None

    def validate_artifacts(self, verify_digests: bool = False) -> ValidationStatus:
        status = ValidationStatus()
        checked: list[str] = []

        for filename in self.required_files:
            path = self._resolve(filename)
            if not os.path.isfile(path):
                status.is_valid = False
                status.errors.append(f"缺少必需产物: {filename}")
            else:
                checked.append(filename)

        status.checked_artifacts = checked

        if not status.is_valid:
            return status

        if verify_digests:
            expected = self._read_manifest_digests()
            if not expected:
                status.warnings.append("未找到 artifact_digests，跳过完整性校验")
                return status

            for filename in self.required_files:
                path = self._resolve(filename)
                actual = _sha256_of_file(path)
                digest_key = filename
                expected_val = expected.get(digest_key) or expected.get(
                    os.path.basename(filename)
                )
                if expected_val is None:
                    status.warnings.append(f"{filename} 无 manifest digest 记录")
                    continue
                if actual.lower() != str(expected_val).lower():
                    status.is_valid = False
                    status.errors.append(
                        f"{filename} digest 不匹配，期望 {expected_val}，实际 {actual}"
                    )

        return status
