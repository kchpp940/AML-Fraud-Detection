from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from aml_fraud_detector.logger import logging


@dataclass
class StageRecord:
    stage_name: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    input_paths: List[str] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)
    status: str = "pending"
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StageTracker:
    def __init__(self):
        self._records: List[StageRecord] = []
        self._current: Optional[StageRecord] = None

    def start_stage(self, stage_name: str, input_paths: Optional[List[str]] = None) -> StageRecord:
        record = StageRecord(
            stage_name=stage_name,
            start_time=datetime.now().isoformat(),
            input_paths=list(input_paths or []),
            status="running",
        )
        self._records.append(record)
        self._current = record
        logging.info(f"[StageTracker] Stage '{stage_name}' started at {record.start_time}")
        return record

    def complete_stage(self, output_paths: Optional[List[str]] = None) -> StageRecord:
        if self._current is None:
            raise RuntimeError("No stage is currently running")
        now = datetime.now().isoformat()
        self._current.end_time = now
        if output_paths is not None:
            self._current.output_paths = list(output_paths)
        self._current.status = "completed"
        start_dt = datetime.fromisoformat(self._current.start_time)
        self._current.duration_seconds = (datetime.fromisoformat(now) - start_dt).total_seconds()
        logging.info(
            f"[StageTracker] Stage '{self._current.stage_name}' completed "
            f"in {self._current.duration_seconds:.3f}s"
        )
        record = self._current
        self._current = None
        return record

    def set_output_paths(self, output_paths: List[str]) -> None:
        if self._current is None:
            raise RuntimeError("No stage is currently running")
        self._current.output_paths = list(output_paths)

    def fail_stage(self, error: Exception) -> StageRecord:
        if self._current is None:
            raise RuntimeError("No stage is currently running")
        now = datetime.now().isoformat()
        self._current.end_time = now
        self._current.status = "failed"
        start_dt = datetime.fromisoformat(self._current.start_time)
        self._current.duration_seconds = (datetime.fromisoformat(now) - start_dt).total_seconds()
        error_info: Dict[str, Any] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        if hasattr(error, "error_detail") and hasattr(error.error_detail, "to_dict"):
            error_info["error_detail"] = error.error_detail.to_dict()
        elif hasattr(error, "error_code"):
            error_info["error_code"] = str(error.error_code)
        self._current.error = error_info
        logging.error(
            f"[StageTracker] Stage '{self._current.stage_name}' failed "
            f"after {self._current.duration_seconds:.3f}s: {error_info['error_message']}"
        )
        record = self._current
        self._current = None
        return record

    @contextmanager
    def track_stage(
        self,
        stage_name: str,
        input_paths: Optional[List[str]] = None,
        output_paths: Optional[List[str]] = None,
    ):
        self.start_stage(stage_name, input_paths)
        try:
            yield self._current
            self.complete_stage(output_paths)
        except Exception as e:
            self.fail_stage(e)
            raise

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [record.to_dict() for record in self._records]

    @property
    def records(self) -> List[StageRecord]:
        return list(self._records)
