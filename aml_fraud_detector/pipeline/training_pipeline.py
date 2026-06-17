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
from aml_fraud_detector.components.data_transformation import DataTransformation
from aml_fraud_detector.components.model_trainer import ModelTrainer
from aml_fraud_detector.components.model_evaluation import ModelEvaluation

from aml_fraud_detector.configuration import TrainingConfig, TrainingSummary
from aml_fraud_detector.config import get_config_service
from aml_fraud_detector.utils.main_utils import save_training_summary
from aml_fraud_detector.entity import TrainingPipelineResult, ProcessStatus


def run_training_pipeline(config_path: Optional[str] = None) -> TrainingPipelineResult:
    logging.info("=" * 72)
    logging.info("AML Fraud Detection - Training pipeline started")
    logging.info("=" * 72)

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
        data_ingestion = DataIngestion(training_config=training_config)
        train_data_path, test_data_path, df_sample = data_ingestion.initiate_data_ingestion()

        summary.data_rows = len(df_sample)
        summary.train_rows = int(len(df_sample) * (1 - training_config.data.test_size))
        summary.test_rows = int(len(df_sample) * training_config.data.test_size)
        logging.info(
            f"Data ingestion done: total={summary.data_rows}, "
            f"train≈{summary.train_rows}, test≈{summary.test_rows}"
        )

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

        try:
            model_evaluation = ModelEvaluation()
            model_evaluation.initiate_model_evaluation(train_arr, test_arr)
            logging.info("Model evaluation done")
        except Exception as e:
            logging.warning(f"ModelEvaluation stage failed (non-fatal): {e}")

        summary.summary_path = os.path.abspath(
            training_config.artifacts_subpath("training_summary.json")
        )
        save_training_summary(
            file_path=summary.summary_path,
            summary_obj=summary,
        )

        try:
            config_svc = get_config_service()
            effective_config_path = config_svc.export_effective_config(
                output_path=os.path.join(
                    training_config.output.artifacts_dir,
                    "effective_config.json",
                ),
                sanitized=True,
                include_metadata=True,
            )
            logging.info(f"Effective (sanitized) config saved: {effective_config_path}")
        except Exception as cfg_err:
            logging.warning(f"Export effective config skipped: {cfg_err}")

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
        print("=" * 72 + "\n")

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
        return TrainingPipelineResult.from_error(e.error_detail)
    except Exception as e:
        logging.error("Training pipeline failed with unexpected error", exc_info=True)
        wrapped = wrap_exception(e, error_details=sys)
        return TrainingPipelineResult.from_error(wrapped.error_detail)


if __name__ == "__main__":
    result = run_training_pipeline()
    if not result.is_success():
        print(f"\nTraining failed: [{result.error_detail.error_code.value}] {result.error_reason}")
        sys.exit(1)
