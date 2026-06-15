import os
import sys
from datetime import datetime
from typing import Optional

from aml_fraud_detector.artifact_registry import ArtifactRegistry
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging

from aml_fraud_detector.components.data_ingestion import DataIngestion
from aml_fraud_detector.components.data_validation import DataValidation, DataValidationConfig
from aml_fraud_detector.components.data_transformation import DataTransformation
from aml_fraud_detector.components.model_trainer import ModelTrainer
from aml_fraud_detector.components.model_evaluation import ModelEvaluation

from aml_fraud_detector.configuration import TrainingConfig, TrainingSummary


def _next_model_version(artifacts_dir: str) -> int:
    metadata_path = os.path.join(artifacts_dir, "model_metadata.json")
    if os.path.isfile(metadata_path):
        import json
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            return prev.get("model_version", 0) + 1
        except Exception:
            pass
    return 1


def run_training_pipeline(config_path: Optional[str] = None) -> TrainingSummary:
    logging.info("=" * 72)
    logging.info("AML Fraud Detection - Training pipeline started")
    logging.info("=" * 72)

    training_config = TrainingConfig(config_path=config_path)
    resolved = training_config.to_resolved_dict()
    logging.info(
        f"Training config loaded (source={resolved['run_info']['config_path_source']}): "
        f"{resolved['run_info']['config_path']}"
    )
    if resolved["env_overrides"]:
        logging.info(
            f"Environment variable overrides applied: "
            f"{[e['path'] for e in resolved['env_overrides']]}"
        )

    registry = ArtifactRegistry(artifacts_dir=training_config.output.artifacts_dir)

    summary = TrainingSummary(
        data_source=resolved["data"]["source_path"],
        target_column=resolved["features"]["target_column"],
        selection_metric=resolved["models"]["selection_metric"],
        artifacts_dir=registry.artifacts_dir,
        resolved_config=resolved,
    )

    try:
        data_ingestion = DataIngestion(training_config=training_config, registry=registry)
        train_data_path, test_data_path, df_sample = data_ingestion.initiate_data_ingestion()

        summary.data_rows = len(df_sample)
        summary.train_rows = int(len(df_sample) * (1 - training_config.data.test_size))
        summary.test_rows = int(len(df_sample) * training_config.data.test_size)
        logging.info(
            f"Data ingestion done: total={summary.data_rows}, "
            f"train≈{summary.train_rows}, test≈{summary.test_rows}"
        )

        validation_config = DataValidationConfig(
            critical_feature_columns=["account", "account_1", "amount_received", "payment_format"],
            amount_columns=["amount_received", "amount_paid"],
            timestamp_columns=["timestamp"],
        )
        data_validation = DataValidation(
            registry=registry,
            target_column=training_config.features.target_column,
            config=validation_config,
        )
        validation_artifact = data_validation.validate(df_sample)
        if not validation_artifact.is_valid:
            logging.warning(f"Data validation issues: {validation_artifact.errors}")
        else:
            logging.info("Data validation passed")
        logging.info(
            f"Data quality report saved: {validation_artifact.report_path}"
        )

        data_transformation = DataTransformation(training_config=training_config, registry=registry)
        transform_artifact = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        train_arr = transform_artifact.train_array
        test_arr = transform_artifact.test_array

        summary.feature_columns = transform_artifact.feature_columns
        summary.numerical_features = transform_artifact.numerical_features
        summary.categorical_features = transform_artifact.categorical_features
        summary.preprocessor_path = transform_artifact.preprocessor_path
        summary.target_column = transform_artifact.target_column or summary.target_column
        logging.info(
            f"Data transformation done: {len(summary.feature_columns)} features "
            f"({len(summary.numerical_features)} num, "
            f"{len(summary.categorical_features)} cat)"
        )

        feature_metadata_path = data_validation.save_feature_metadata(
            df=df_sample,
            numerical_features=transform_artifact.numerical_features,
            categorical_features=transform_artifact.categorical_features,
            encoding_info=_build_encoding_info(transform_artifact),
            feature_labels=_build_feature_labels(transform_artifact),
        )
        validation_artifact.feature_metadata_path = feature_metadata_path

        model_trainer = ModelTrainer(training_config=training_config, registry=registry)
        trainer_artifact = model_trainer.initiate_model_trainer(train_arr, test_arr)

        summary.candidate_models = trainer_artifact.candidate_models
        summary.selection_metric = trainer_artifact.selection_metric
        summary.best_model_name = (
            f"{trainer_artifact.best_model_name} "
            f"({trainer_artifact.best_model_display_name})"
        )
        summary.best_model_params = trainer_artifact.best_model_params
        summary.best_metric_value = trainer_artifact.best_metric_value
        summary.all_model_metrics = trainer_artifact.all_model_metrics
        summary.model_path = trainer_artifact.model_path
        logging.info(
            f"Model training done: best='{summary.best_model_name}', "
            f"{summary.selection_metric}={summary.best_metric_value:.4f}"
        )

        model_version = _next_model_version(registry.artifacts_dir)
        model_metadata = _build_model_metadata(
            model_version=model_version,
            trainer_artifact=trainer_artifact,
            data_source=summary.data_source,
            artifacts_dir=registry.artifacts_dir,
        )
        model_metadata_path = registry.save_model_metadata(model_metadata)
        logging.info(
            f"Model metadata saved to: {registry.relative_path('model_metadata_json')} "
            f"(version={model_version})"
        )

        summary.model_metadata_path = model_metadata_path

        summary.summary_path = registry.path("training_summary_json")
        summary.artifact_manifest_path = registry.path("artifact_manifest_json")

        registry.save_training_summary(summary)

        registry.save_manifest()
        logging.info(f"Artifact manifest saved to: {registry.relative_path('artifact_manifest_json')}")

        logging.info("=" * 72)
        logging.info("Training pipeline completed successfully")
        logging.info("=" * 72)

        print("\n" + "=" * 72)
        print("TRAINING SUMMARY")
        print("=" * 72)
        print(f"Config file        : {resolved['run_info']['config_path']}")
        print(f"Config source      : {resolved['run_info']['config_path_source']}")
        print(f"YAML loaded        : {resolved['run_info']['config_path_source'] != 'defaults_only'}")
        if resolved['env_overrides']:
            print(f"Env overrides ({len(resolved['env_overrides'])}):")
            for ov in resolved['env_overrides']:
                print(f"  - {ov['env_name']} -> {ov['path']}: "
                      f"{ov['original_value']!r} -> {ov['resolved_value']!r}")
        else:
            print("Env overrides      : (none)")
        print(f"Data source        : {summary.data_source}")
        print(f"Total rows         : {summary.data_rows}")
        print(f"Train / Test rows  : {summary.train_rows} / {summary.test_rows}")
        print(f"Target column      : {summary.target_column}")
        print(f"Drop columns       : {resolved['features']['drop_columns']}")
        print(f"Feature columns    : {len(summary.feature_columns)}")
        print(f"  - Numerical      : {summary.numerical_features}")
        print(f"  - Categorical    : {summary.categorical_features}")
        print(f"Enabled models     : {resolved['models']['enabled_display_names']}")
        print(f"Selection metric   : {summary.selection_metric}")
        print(f"Best model         : {summary.best_model_name}")
        print(f"Best metric value  : {summary.best_metric_value:.6f}")
        print(f"Artifacts dir      : {summary.artifacts_dir}")
        print(f"Preprocessor saved : {summary.preprocessor_path}")
        print(f"Model saved        : {summary.model_path}")
        print(f"Summary saved      : {summary.summary_path}")
        print(f"Model metadata     : {summary.model_metadata_path}")
        print(f"Artifact manifest  : {summary.artifact_manifest_path}")
        print(f"Registry entries   : {registry.all_keys()}")
        print("=" * 72 + "\n")

        return summary

    except Exception as e:
        logging.error("Training pipeline failed", exc_info=True)
        raise CustomerException(e, sys)


def _build_encoding_info(transform_artifact) -> dict:
    encoding_info = {}
    for col in transform_artifact.numerical_features:
        encoding_info[col] = {"type": "numerical"}
    high_card = [c for c in ["account", "account_1"] if c in transform_artifact.categorical_features]
    low_card = [c for c in ["payment_format", "day"] if c in transform_artifact.categorical_features]
    for col in high_card:
        encoding_info[col] = {"type": "frequency"}
    for col in low_card:
        encoding_info[col] = {"type": "onehot"}
    return encoding_info


def _build_feature_labels(transform_artifact) -> dict:
    labels = {}
    label_map = {
        "amount_received": "交易金额",
        "account": "发起账户",
        "account_1": "接收账户",
        "payment_format": "支付方式",
        "day": "交易星期",
    }
    for col in transform_artifact.numerical_features + transform_artifact.categorical_features:
        if col in label_map:
            labels[col] = label_map[col]
    return labels


def _build_model_metadata(
    model_version: int,
    trainer_artifact,
    data_source: str,
    artifacts_dir: str,
) -> dict:
    raw_csv_path = os.path.join(artifacts_dir, "data.csv")
    data_digest = ""
    if os.path.isfile(raw_csv_path):
        data_digest = ArtifactRegistry.compute_hash(raw_csv_path)
    return {
        "model_version": model_version,
        "training_time": datetime.now().isoformat(),
        "data_file": os.path.abspath(raw_csv_path),
        "data_file_digest": data_digest,
        "feature_schema_version": "1.0",
        "best_model_name": (
            f"{trainer_artifact.best_model_name} "
            f"({trainer_artifact.best_model_display_name})"
        ),
        "best_model_params": trainer_artifact.best_model_params,
        "selection_metric": trainer_artifact.selection_metric,
        "best_metric_value": trainer_artifact.best_metric_value,
        "all_model_metrics": trainer_artifact.all_model_metrics,
        "artifact_path": trainer_artifact.model_path,
    }


if __name__ == "__main__":
    run_training_pipeline()
