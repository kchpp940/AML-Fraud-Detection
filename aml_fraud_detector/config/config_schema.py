from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class DataConfigSection:
    source_path: str = "notebook/data/HI-Small_Trans.csv"
    sample_size: Optional[int] = 50000
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class FeaturesConfigSection:
    target_column: str = "is_laundering"
    drop_columns: List[str] = field(default_factory=lambda: [
        "timestamp", "date", "time", "amount_paid",
        "receiving_currency", "payment_currency",
        "from_bank", "to_bank",
    ])


@dataclass
class ModelsConfigSection:
    selection_metric: str = "Recall"
    enabled: Dict[str, bool] = field(default_factory=lambda: {
        "RandomForest": True,
        "AdaBoost": True,
        "GradientBoosting": False,
        "XGBoost": True,
    })
    param_grid: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class OutputConfigSection:
    artifacts_dir: str = "artifacts"
    train_csv_name: str = "train.csv"
    test_csv_name: str = "test.csv"
    raw_csv_name: str = "data.csv"
    preprocessor_name: str = "preprocessor.pkl"
    model_name: str = "model.pkl"
    summary_name: str = "training_summary.json"
    feature_metadata_name: str = "feature_metadata.json"
    model_metadata_name: str = "model_metadata.json"
    quality_report_name: str = "data_quality_report.json"
    manifest_name: str = "artifact_manifest.json"


@dataclass
class TrainingConfigSection:
    data: DataConfigSection = field(default_factory=DataConfigSection)
    features: FeaturesConfigSection = field(default_factory=FeaturesConfigSection)
    models: ModelsConfigSection = field(default_factory=ModelsConfigSection)
    output: OutputConfigSection = field(default_factory=OutputConfigSection)


@dataclass
class PredictionConfigSection:
    default_artifacts_dir: str = "artifacts"
    model_file_name: str = "model.pkl"
    preprocessor_file_name: str = "preprocessor.pkl"
    feature_metadata_file_name: str = "feature_metadata.json"
    model_metadata_file_name: str = "model_metadata.json"
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "critical": 0.9,
        "high": 0.7,
        "medium": 0.5,
    })
    enable_lazy_load: bool = True


@dataclass
class ServerConfigSection:
    flask_host: str = "0.0.0.0"
    flask_port: int = 8080
    flask_debug: bool = False
    streamlit_port: int = 8501
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    request_timeout: int = 30
    max_content_length: int = 16 * 1024 * 1024


@dataclass
class LoggingConfigSection:
    level: str = "INFO"
    log_dir: str = "logs"
    log_to_file: bool = True
    log_to_console: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class SecurityConfigSection:
    mask_secrets_in_logs: bool = True
    secret_keys: List[str] = field(default_factory=lambda: [
        "api_key", "secret", "token", "password", "credential",
    ])
    enable_input_sanitization: bool = True


@dataclass
class AppConfig:
    app_name: str = "AML Fraud Detector"
    app_version: str = "1.0.0"
    env: str = "local"
    project_root: str = ""
    training: TrainingConfigSection = field(default_factory=TrainingConfigSection)
    prediction: PredictionConfigSection = field(default_factory=PredictionConfigSection)
    server: ServerConfigSection = field(default_factory=ServerConfigSection)
    logging: LoggingConfigSection = field(default_factory=LoggingConfigSection)
    security: SecurityConfigSection = field(default_factory=SecurityConfigSection)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
