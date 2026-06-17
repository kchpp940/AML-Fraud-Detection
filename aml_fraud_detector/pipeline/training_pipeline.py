import os
import sys
from dataclasses import asdict
from typing import Optional

from aml_fraud_detector.exception import (
    AMLException,
    PipelineException,
    wrap_exception,
    create_error_from_exception,
)
from aml_fraud_detector.logger import logging
from aml_fraud_detector.constants import ErrorCode

from aml_fraud_detector.components.data_ingestion import DataIngestion
from aml_fraud_detector.components.data_validation import DataValidation
from aml_fraud_detector.components.data_transformation import DataTransformation
from aml_fraud_detector.components.model_trainer import ModelTrainer
from aml_fraud_detector.components.model_evaluation import ModelEvaluation

from aml_fraud_detector.configuration import TrainingConfig, TrainingSummary
from aml_fraud_detector.utils.main_utils import save_training_summary
from aml_fraud_detector.utils.stage_tracker import StageTracker
from aml_fraud_detector.entity import TrainingPipelineResult, ProcessStatus


def run_training_pipeline(config_path: Optional[str] = None) -> TrainingPipelineResult:
    logging.info("=" * 72)
    logging.info("AML Fraud Detection - Training pipeline started")
    logging.info("=" * 72)

    tracker = StageTracker()

    try:
        training_config = TrainingConfig(config_path=config_path)
    except AMLException as e:
        logging.error(f"Training config load failed: {e}")
        return TrainingPipelineResult.from_error(e.error_detail)
    except Exception as e:
        logging.error("Training config load failed with unexpected error", exc_info=True)
        err = wrap_exception(e, error_details=sys)
        return TrainingPipelineResult.from_error(err.error_detail)

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
        # ── Stage 1: DataIngestion ──
        with tracker.track_stage(
            "DataIngestion",
            input_paths=[resolved["data"]["source_path"]],
            output_paths=[
                resolved["output"]["raw_csv"],
                resolved["output"]["train_csv"],
                resolved["output"]["test_csv"],
            ],
        ):
            data_ingestion = DataIngestion(training_config=training_config)
            train_data_path, test_data_path, df_sample = data_ingestion.initiate_data_ingestion()

        summary.data_rows = len(df_sample)
        summary.train_rows = int(len(df_sample) * (1 - training_config.data.test_size))
        summary.test_rows = int(len(df_sample) * training_config.data.test_size)
        logging.info(
            f"Data ingestion done: total={summary.data_rows}, "
            f"train≈{summary.train_rows}, test≈{summary.test_rows}"
        )

        # ── Stage 2: DataValidation ──
        with tracker.track_stage(
            "DataValidation",
            input_paths=[
                resolved["output"]["train_csv"],
                resolved["output"]["test_csv"],
            ],
            output_paths=[
                resolved["output"]["train_csv"],
                resolved["output"]["test_csv"],
            ],
        ):
            data_validation = DataValidation(training_config=training_config)
            data_validation.initiate_data_validation(train_data_path, test_data_path)

        logging.info("Data validation done")

        # ── Stage 3: DataTransformation ──
        with tracker.track_stage(
            "DataTransformation",
            input_paths=[
                resolved["output"]["train_csv"],
                resolved["output"]["test_csv"],
            ],
            output_paths=[resolved["output"]["preprocessor_pkl"]],
        ):
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

        # ── Stage 4: ModelTrainer ──
        with tracker.track_stage(
            "ModelTrainer",
            input_paths=[resolved["output"]["preprocessor_pkl"]],
            output_paths=[resolved["output"]["model_pkl"]],
        ):
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

        # ── Stage 5: ModelEvaluation ──
        tracker.start_stage(
            "ModelEvaluation",
            input_paths=[resolved["output"]["model_pkl"]],
        )
        try:
            model_evaluation = ModelEvaluation(training_config=training_config)
            eval_artifact = model_evaluation.initiate_model_evaluation(train_arr, test_arr)
            tracker.complete_stage(output_paths=[eval_artifact.model_path])
            logging.info(
                f"Model evaluation done: precision={eval_artifact.precision:.4f}, "
                f"recall={eval_artifact.recall:.4f}, f1={eval_artifact.f1_score:.4f}"
            )
        except Exception as e:
            tracker.fail_stage(e)
            logging.warning(f"ModelEvaluation stage failed (non-fatal): {e}")

        summary.stage_records = tracker.to_dict_list()
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

        _print_summary(resolved, summary, tracker)

        return TrainingPipelineResult(
            process_status=ProcessStatus.SUCCESS,
            summary=asdict(summary),
            artifacts={
                "train_data": train_data_path,
                "test_data": test_data_path,
                "preprocessor": summary.preprocessor_path,
                "model": summary.model_path,
                "summary": summary.summary_path,
            },
        )

    except AMLException as e:
        logging.error(f"Training pipeline failed: {e}", exc_info=True)
        summary.stage_records = tracker.to_dict_list()
        try:
            summary.summary_path = os.path.abspath(
                training_config.artifacts_subpath("training_summary.json")
            )
            save_training_summary(file_path=summary.summary_path, summary_obj=summary)
        except Exception:
            logging.warning("Failed to save partial training summary after pipeline error")
        return TrainingPipelineResult.from_error(e.error_detail)
    except Exception as e:
        logging.error("Training pipeline failed with unexpected error", exc_info=True)
        wrapped = wrap_exception(e, error_details=sys)
        summary.stage_records = tracker.to_dict_list()
        try:
            summary.summary_path = os.path.abspath(
                training_config.artifacts_subpath("training_summary.json")
            )
            save_training_summary(file_path=summary.summary_path, summary_obj=summary)
        except Exception:
            logging.warning("Failed to save partial training summary after pipeline error")
        return TrainingPipelineResult.from_error(wrapped.error_detail)


def _print_summary(resolved, summary, tracker):
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
    print("-" * 72)
    print("STAGE TRACKER")
    print("-" * 72)
    for rec in tracker.records:
        dur = f"{rec.duration_seconds:.3f}s" if rec.duration_seconds is not None else "N/A"
        status_icon = "✓" if rec.status == "completed" else "✗"
        print(f"  {status_icon} {rec.stage_name:<24s} | {rec.status:<10s} | {dur:>10s}")
        if rec.input_paths:
            print(f"    {'input':>24s} : {rec.input_paths}")
        if rec.output_paths:
            print(f"    {'output':>24s} : {rec.output_paths}")
        if rec.error:
            print(f"    {'error':>24s} : {rec.error.get('error_message', '')}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    result = run_training_pipeline()
    if not result.is_success():
        print(f"\nTraining failed: [{result.error_detail.error_code.value}] {result.error_reason}")
        sys.exit(1)
