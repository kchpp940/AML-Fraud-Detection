import json
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging

from aml_fraud_detector.components.data_ingestion import DataIngestion
from aml_fraud_detector.components.data_validation import DataValidation
from aml_fraud_detector.components.data_transformation import DataTransformation
from aml_fraud_detector.components.model_trainer import ModelTrainer
from aml_fraud_detector.components.model_evaluation import ModelEvaluation

from aml_fraud_detector.configuration import TrainingConfig, TrainingSummary
from aml_fraud_detector.utils.main_utils import save_training_summary


def _load_data_quality_report(report_path: str) -> Dict[str, Any]:
    if not os.path.isfile(report_path):
        raise FileNotFoundError(
            f"Data quality report not found at: {report_path}. "
            f"Ensure data validation was executed before training."
        )
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _log_data_quality_risks(report: Dict[str, Any]) -> None:
    risk_items = report.get("risk_items", [])
    if not risk_items:
        logging.info("Data quality check: no risk items detected")
        return

    criticals = [r for r in risk_items if r.get("level") == "CRITICAL"]
    warnings = [r for r in risk_items if r.get("level") == "WARNING"]

    logging.warning(
        f"Data quality check completed: "
        f"{len(criticals)} critical, {len(warnings)} warnings"
    )

    for r in criticals:
        logging.error(
            f"[DATA QUALITY CRITICAL] [{r.get('category', 'unknown')}] {r.get('message')}"
        )
    for r in warnings:
        logging.warning(
            f"[DATA QUALITY WARNING] [{r.get('category', 'unknown')}] {r.get('message')}"
        )

    target_dist = report.get("target_distribution", {})
    if target_dist:
        logging.info(
            f"Target column distribution: {target_dist.get('value_counts', {})}, "
            f"positive_ratio={target_dist.get('positive_ratio')}, "
            f"severity={target_dist.get('overall_severity')}"
        )


def _validate_before_training(
    report: Dict[str, Any], training_config: TrainingConfig
) -> None:
    target_col = training_config.features.target_column
    errors: List[str] = []

    critical_missing = report.get("critical_missing_columns", [])
    if critical_missing:
        for cm in critical_missing:
            if isinstance(cm, dict):
                col = cm.get("column", "unknown")
                count = cm.get("missing_count", 0)
                ratio = cm.get("missing_ratio", 0)
                threshold = cm.get("threshold_critical", 0.3)
                errors.append(
                    f"Critical feature column '{col}' has {count} missing values "
                    f"({ratio:.2%}), exceeding critical threshold {threshold:.2%}. "
                    "Training cannot proceed reliably."
                )
            else:
                errors.append(
                    f"Critical feature column '{cm}' has excessive missing values "
                    "(>30%). Training cannot proceed reliably."
                )

    target_dist = report.get("target_distribution", {})
    target_severity = target_dist.get("overall_severity", "NONE")
    if target_severity == "CRITICAL":
        target_issues = target_dist.get("issues", [])
        for issue in target_issues:
            if issue.get("severity") == "CRITICAL":
                code = issue.get("code", "unknown")
                ratio = issue.get("ratio")
                threshold = issue.get("threshold")
                base_msg = issue.get("message", "Unknown issue")
                extra = ""
                if ratio is not None and threshold is not None:
                    extra = f" (ratio={ratio:.4%}, threshold={threshold:.2%})"
                errors.append(
                    f"Target column '{target_col}' [{code}]: {base_msg}{extra}"
                )

    missing_values = report.get("missing_values", [])
    for mv in missing_values:
        if mv.get("column") == target_col and mv.get("severity") == "CRITICAL":
            count = mv.get("missing_count", 0)
            ratio = mv.get("missing_ratio", 0)
            threshold = mv.get("threshold_critical", 0.3)
            errors.append(
                f"Target column '{target_col}' has {count} missing values "
                f"({ratio:.2%}), exceeding critical threshold {threshold:.2%}. "
                "Training cannot proceed."
            )

    if errors:
        joined = "\n  - ".join(errors)
        msg = (
            "Data quality gate FAILED. Training aborted before data transformation. "
            f"Fix the following issues or adjust validation thresholds:\n  - {joined}"
        )
        logging.error(msg)
        raise ValueError(msg)

    logging.info("Data quality gate PASSED: proceeding to data transformation")


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

    summary = TrainingSummary(
        data_source=resolved["data"]["source_path"],
        target_column=resolved["features"]["target_column"],
        selection_metric=resolved["models"]["selection_metric"],
        artifacts_dir=resolved["output"]["artifacts_dir"],
        resolved_config=resolved,
    )

    try:
        data_ingestion = DataIngestion(training_config=training_config)
        train_data_path, test_data_path, df_sample = data_ingestion.initiate_data_ingestion()

        summary.data_rows = len(df_sample)
        summary.train_rows = int(len(df_sample) * (1 - training_config.data.test_size))
        summary.test_rows = int(len(df_sample) * training_config.data.test_size)
        logging.info(
            f"Data ingestion done: total={summary.data_rows}, "
            f"train≈{summary.train_rows}, test≈{summary.test_rows}"
        )

        data_validation = DataValidation(training_config=training_config)
        validation_artifact = data_validation.initiate_data_validation(df=df_sample)
        logging.info(
            f"Data validation done: report_path={validation_artifact.report_path}, "
            f"is_valid={validation_artifact.is_valid}"
        )

        summary.quality_report_path = os.path.abspath(validation_artifact.report_path)
        summary.data_quality_valid = validation_artifact.is_valid
        summary.data_quality_critical_issues = list(validation_artifact.critical_issues)
        summary.data_quality_warning_issues = list(validation_artifact.warning_issues)

        quality_report = _load_data_quality_report(validation_artifact.report_path)
        _log_data_quality_risks(quality_report)
        _validate_before_training(quality_report, training_config)

        data_transformation = DataTransformation(training_config=training_config)
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

        model_trainer = ModelTrainer(training_config=training_config)
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

        summary.summary_path = os.path.abspath(
            training_config.artifacts_subpath("training_summary.json")
        )
        save_training_summary(
            file_path=summary.summary_path,
            summary_obj=summary,
        )

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
        print(f"Quality report     : {summary.quality_report_path}")
        print(f"Data quality valid : {summary.data_quality_valid}")
        if summary.data_quality_critical_issues:
            print(f"  - Critical issues ({len(summary.data_quality_critical_issues)}):")
            for issue in summary.data_quality_critical_issues:
                print(f"    * {issue}")
        if summary.data_quality_warning_issues:
            print(f"  - Warning issues ({len(summary.data_quality_warning_issues)}):")
            for issue in summary.data_quality_warning_issues:
                print(f"    * {issue}")
        print(f"Summary saved      : {summary.summary_path}")
        print("=" * 72 + "\n")

        return summary

    except Exception as e:
        logging.error("Training pipeline failed", exc_info=True)
        raise CustomerException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
