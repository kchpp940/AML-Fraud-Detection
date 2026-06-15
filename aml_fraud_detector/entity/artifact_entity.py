from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataValidationArtifact:
    report_path: str = ""
    report: Dict[str, Any] = field(default_factory=dict)
    is_valid: bool = True
    critical_issues: List[str] = field(default_factory=list)
    warning_issues: List[str] = field(default_factory=list)
