import os
import sys
import shutil
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List

from aml_fraud_detector.logger import logging
from aml_fraud_detector.exception import ConfigException
from aml_fraud_detector.constants import ErrorCode


class WorkspaceMode(str, Enum):
    FLAT = "flat"
    BATCHED = "batched"


DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_TMP_DIR = "tmp"
DEFAULT_RUNS_DIR = "runs"
LATEST_LINK_NAME = "latest"


class WorkspaceContext:
    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        workspace_root: Optional[str] = None,
        mode: WorkspaceMode = WorkspaceMode.FLAT,
        run_name: Optional[str] = None,
    ):
        self._mode = mode
        self._workspace_root = os.path.abspath(workspace_root or os.getcwd())
        self._artifacts_dir_input = artifacts_dir or DEFAULT_ARTIFACTS_DIR

        if mode == WorkspaceMode.BATCHED:
            self._run_name = run_name or self._generate_run_name()
            self._run_dir = os.path.join(
                self._workspace_root, DEFAULT_RUNS_DIR, self._run_name
            )
            self._artifacts_dir = os.path.join(self._run_dir, "artifacts")
            self._logs_dir = os.path.join(self._run_dir, "logs")
            self._tmp_dir = os.path.join(self._run_dir, "tmp")
        else:
            self._run_name = run_name or "default"
            self._run_dir = self._workspace_root
            if os.path.isabs(self._artifacts_dir_input):
                self._artifacts_dir = self._artifacts_dir_input
            else:
                self._artifacts_dir = os.path.join(
                    self._workspace_root, self._artifacts_dir_input
                )
            self._logs_dir = os.path.join(self._workspace_root, DEFAULT_LOGS_DIR)
            self._tmp_dir = os.path.join(self._workspace_root, DEFAULT_TMP_DIR)

        self._artifact_paths: Dict[str, str] = {}
        self._init_artifact_paths()
        logging.info(
            f"WorkspaceContext initialized (mode={mode.value}, "
            f"root={self._workspace_root}, artifacts={self._artifacts_dir})"
        )

    @staticmethod
    def _generate_run_name() -> str:
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")

    def _init_artifact_paths(self) -> None:
        self._artifact_paths = {
            "raw_csv": os.path.join(self._artifacts_dir, "data.csv"),
            "train_csv": os.path.join(self._artifacts_dir, "train.csv"),
            "test_csv": os.path.join(self._artifacts_dir, "test.csv"),
            "preprocessor_pkl": os.path.join(self._artifacts_dir, "preprocessor.pkl"),
            "model_pkl": os.path.join(self._artifacts_dir, "model.pkl"),
            "summary_json": os.path.join(self._artifacts_dir, "training_summary.json"),
            "feature_metadata_json": os.path.join(
                self._artifacts_dir, "feature_metadata.json"
            ),
            "model_metadata_json": os.path.join(
                self._artifacts_dir, "model_metadata.json"
            ),
            "data_quality_report_json": os.path.join(
                self._artifacts_dir, "data_quality_report.json"
            ),
            "artifact_manifest_json": os.path.join(
                self._artifacts_dir, "artifact_manifest.json"
            ),
        }

    def ensure_directories(self) -> None:
        os.makedirs(self._artifacts_dir, exist_ok=True)
        os.makedirs(self._logs_dir, exist_ok=True)
        os.makedirs(self._tmp_dir, exist_ok=True)
        logging.debug(
            f"Workspace directories ensured: artifacts={self._artifacts_dir}, "
            f"logs={self._logs_dir}, tmp={self._tmp_dir}"
        )

    @property
    def mode(self) -> WorkspaceMode:
        return self._mode

    @property
    def workspace_root(self) -> str:
        return self._workspace_root

    @property
    def run_name(self) -> str:
        return self._run_name

    @property
    def run_dir(self) -> str:
        return self._run_dir

    @property
    def artifacts_dir(self) -> str:
        return self._artifacts_dir

    @property
    def logs_dir(self) -> str:
        return self._logs_dir

    @property
    def tmp_dir(self) -> str:
        return self._tmp_dir

    def get_artifact_path(self, name: str) -> str:
        if name not in self._artifact_paths:
            raise ConfigException(
                ErrorCode.CONFIG_INVALID_VALUE,
                error_details=sys,
                key="artifact_name",
                value=name,
                detail=f"Unknown artifact name: {name}",
            )
        return self._artifact_paths[name]

    def artifacts_subpath(self, *parts: str) -> str:
        return os.path.join(self._artifacts_dir, *parts)

    def tmp_subpath(self, *parts: str) -> str:
        return os.path.join(self._tmp_dir, *parts)

    def logs_subpath(self, *parts: str) -> str:
        return os.path.join(self._logs_dir, *parts)

    def list_artifacts(self) -> List[str]:
        if not os.path.isdir(self._artifacts_dir):
            return []
        return sorted(os.listdir(self._artifacts_dir))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self._mode.value,
            "workspace_root": self._workspace_root,
            "run_name": self._run_name,
            "run_dir": self._run_dir,
            "artifacts_dir": self._artifacts_dir,
            "logs_dir": self._logs_dir,
            "tmp_dir": self._tmp_dir,
            "artifacts": dict(self._artifact_paths),
        }

    def update_latest_link(self) -> Optional[str]:
        if self._mode != WorkspaceMode.BATCHED:
            return None
        runs_dir = os.path.join(self._workspace_root, DEFAULT_RUNS_DIR)
        latest_link = os.path.join(runs_dir, LATEST_LINK_NAME)
        try:
            if os.path.islink(latest_link) or os.path.exists(latest_link):
                if os.path.islink(latest_link):
                    os.unlink(latest_link)
                else:
                    shutil.rmtree(latest_link)
            os.symlink(self._run_dir, latest_link, target_is_directory=True)
            logging.info(f"Updated latest run link: {latest_link} -> {self._run_dir}")
            return latest_link
        except Exception as e:
            logging.warning(f"Failed to update latest run link: {e}")
            return None

    def get_latest_run_dir(self) -> Optional[str]:
        runs_dir = os.path.join(self._workspace_root, DEFAULT_RUNS_DIR)
        latest_link = os.path.join(runs_dir, LATEST_LINK_NAME)
        if os.path.islink(latest_link):
            target = os.readlink(latest_link)
            if os.path.isabs(target):
                return target
            return os.path.abspath(os.path.join(runs_dir, target))
        if os.path.isdir(latest_link):
            return os.path.abspath(latest_link)
        return None

    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]) -> "WorkspaceContext":
        output_cfg = config.get("output", {}) if isinstance(config, dict) else {}
        workspace_cfg = output_cfg.get("workspace", {}) if isinstance(output_cfg, dict) else {}

        mode_str = workspace_cfg.get("mode", WorkspaceMode.FLAT.value)
        try:
            mode = WorkspaceMode(mode_str)
        except ValueError:
            mode = WorkspaceMode.FLAT

        return cls(
            artifacts_dir=output_cfg.get("artifacts_dir"),
            workspace_root=workspace_cfg.get("root"),
            mode=mode,
            run_name=workspace_cfg.get("run_name"),
        )
