import os
import sys
from dataclasses import asdict
from typing import Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging

from aml_fraud_detector.components.data_ingestion import DataIngestion
from aml_fraud_detector.components.data_transformation import DataTransformation
from aml_fraud_detector.components.model_trainer import ModelTrainer
from aml_fraud_detector.components.model_evaluation import ModelEvaluation

from aml_fraud_detector.configuration import TrainingConfig, TrainingSummary
from aml_fraud_detector.utils.main_utils import save_training_summary, save_model_metadata


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

        metadata_path = save_model_metadata(
            artifacts_dir=summary.artifacts_dir,
            data_source_path=summary.data_source,
            best_model_name=summary.best_model_name,
            best_model_params=summary.best_model_params,
            selection_metric=summary.selection_metric,
            best_metric_value=summary.best_metric_value,
            all_model_metrics=summary.all_model_metrics,
            model_path=summary.model_path,
        )
        summary.model_metadata_path = metadata_path

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
        print(f"Metadata saved     : {summary.model_metadata_path}")
        print("=" * 72 + "\n")

        return summary

    except Exception as e:
        logging.error("Training pipeline failed", exc_info=True)
        raise CustomerException(e, sys)


if __name__ == "__main__":
    run_training_pipeline()
