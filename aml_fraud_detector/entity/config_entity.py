from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DataValidationConfig:
    report_file_name: str = "data_quality_report.json"
    missing_value_critical_ratio: float = 0.3
    missing_value_warning_ratio: float = 0.1
    unknown_category_critical_ratio: float = 0.05
    unknown_category_warning_ratio: float = 0.01
    duplicate_row_warning_ratio: float = 0.1
    target_missing_critical_ratio: float = 0.0
    target_expected_classes: int = 2
    target_imbalance_critical_ratio: float = 0.01
    target_imbalance_warning_ratio: float = 0.05
    categorical_columns: List[str] = field(default_factory=list)
    categorical_whitelist: Dict[str, List[str]] = field(default_factory=dict)
    amount_columns: List[str] = field(default_factory=lambda: ["amount_received", "amount_paid"])
    timestamp_columns: List[str] = field(default_factory=lambda: ["timestamp"])
    critical_feature_columns: List[str] = field(default_factory=list)
