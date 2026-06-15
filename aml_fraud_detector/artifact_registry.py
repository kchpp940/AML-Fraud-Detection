import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import dill

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


ARTIFACT_NAMES = {
    "raw_csv": "data.csv",
    "train_csv": "train.csv",
    "test_csv": "test.csv",
    "preprocessor_pkl": "preprocessor.pkl",
    "model_pkl": "model.pkl",
    "training_summary_json": "training_summary.json",
    "data_quality_report_json": "data_quality_report.json",
    "feature_metadata_json": "feature_metadata.json",
    "model_metadata_json": "model_metadata.json",
    "artifact_manifest_json": "artifact_manifest.json",
    "feature_schema_pkl": "feature_schema.pkl",
    "transformed_train_npy": "transformed_train.npy",
    "transformed_test_npy": "transformed_test.npy",
}


@dataclass
class ArtifactEntry:
    name: str
    path: str
    size: int
    mtime_iso: str
    digest: str


class ArtifactRegistry:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = os.path.abspath(artifacts_dir)
        self._entries: Dict[str, ArtifactEntry] = {}
        os.makedirs(self.artifacts_dir, exist_ok=True)
        logging.info(f"ArtifactRegistry initialized: artifacts_dir={self.artifacts_dir}")

    def path(self, key: str) -> str:
        if key not in ARTIFACT_NAMES:
            raise CustomerException(
                KeyError(f"Unknown artifact key: {key!r}. Valid keys: {sorted(ARTIFACT_NAMES)}"),
                sys,
            )
        return os.path.join(self.artifacts_dir, ARTIFACT_NAMES[key])

    def relative_path(self, key: str) -> str:
        if key not in ARTIFACT_NAMES:
            raise CustomerException(
                KeyError(f"Unknown artifact key: {key!r}. Valid keys: {sorted(ARTIFACT_NAMES)}"),
                sys,
            )
        return ARTIFACT_NAMES[key]

    @staticmethod
    def compute_hash(file_path: str, algorithm: str = "sha256") -> str:
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return f"{algorithm}:{h.hexdigest()}"

    def _record(self, key: str) -> None:
        file_path = self.path(key)
        if not os.path.isfile(file_path):
            return
        stat = os.stat(file_path)
        self._entries[key] = ArtifactEntry(
            name=key,
            path=os.path.abspath(file_path),
            size=stat.st_size,
            mtime_iso=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            digest=self.compute_hash(file_path),
        )

    def save_json(self, key: str, data: Dict[str, Any]) -> str:
        try:
            file_path = self.path(key)
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            if "generated_at" not in data:
                data["generated_at"] = datetime.now().isoformat()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            self._record(key)
            logging.info(f"Artifact saved: {file_path}")
            return os.path.abspath(file_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def save_object(self, key: str, obj: Any) -> str:
        try:
            file_path = self.path(key)
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "wb") as f:
                dill.dump(obj, f)
            self._record(key)
            logging.info(f"Artifact saved: {file_path}")
            return os.path.abspath(file_path)
        except Exception as e:
            raise CustomerException(e, sys)

    def record_existing(self, key: str) -> str:
        file_path = self.path(key)
        if not os.path.isfile(file_path):
            raise CustomerException(
                FileNotFoundError(f"Expected artifact not found: {file_path}"), sys
            )
        self._record(key)
        return os.path.abspath(file_path)

    def has_entry(self, key: str) -> bool:
        return key in self._entries

    def get_entry(self, key: str) -> Optional[ArtifactEntry]:
        return self._entries.get(key)

    def build_manifest(self) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {
            "artifacts_dir": self.artifacts_dir,
            "artifacts": {},
        }
        for key, entry in self._entries.items():
            filename = ARTIFACT_NAMES[key]
            manifest["artifacts"][filename] = {
                "path": entry.path,
                "size": entry.size,
                "mtime_iso": entry.mtime_iso,
                "digest": entry.digest,
            }
        return manifest

    def save_manifest(self) -> str:
        return self.save_json("artifact_manifest_json", self.build_manifest())

    def save_training_summary(self, summary_obj: Any) -> str:
        try:
            data = asdict(summary_obj) if hasattr(summary_obj, "__dataclass_fields__") else dict(summary_obj)
            data["generated_at"] = datetime.now().isoformat()
            return self.save_json("training_summary_json", data)
        except Exception as e:
            raise CustomerException(e, sys)

    def save_model_metadata(self, metadata: Dict[str, Any]) -> str:
        return self.save_json("model_metadata_json", metadata)

    def save_data_quality_report(self, report: Dict[str, Any]) -> str:
        return self.save_json("data_quality_report_json", report)

    def save_feature_metadata(self, metadata: Dict[str, Any]) -> str:
        return self.save_json("feature_metadata_json", metadata)

    def all_keys(self) -> List[str]:
        return list(self._entries.keys())

    def all_entries(self) -> Dict[str, ArtifactEntry]:
        return dict(self._entries)
