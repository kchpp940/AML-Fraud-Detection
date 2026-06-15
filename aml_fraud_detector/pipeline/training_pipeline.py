import os
import sys
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging

from aml_fraud_detector.components.data_ingestion import DataIngestion
from aml_fraud_detector.components.data_transformation import DataTransformation
from aml_fraud_detector.components.model_trainer import ModelTrainer
from aml_fraud_detector.components.model_evaluation import ModelEvaluation
from aml_fraud_detector.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_ingestion_artifact = DataIngestionArtifact(
        train_file_path=train_data,
        test_file_path=test_data,
    )
    logging.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

    data_transformation = DataTransformation()
    data_transformation_artifact = data_transformation.initiate_data_transformation(
        data_ingestion_artifact.train_file_path,
        data_ingestion_artifact.test_file_path,
    )
    logging.info(
        f"Data Transformation Artifact: preprocessor={data_transformation_artifact.preprocessor_object_file_path}, "
        f"schema={data_transformation_artifact.feature_schema_file_path}"
    )

    train_arr, test_arr = data_transformation_artifact

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_arr, test_arr)

    # model_eval_obj = ModelEvaluation()
    # model_eval_obj.initiate_model_evaluation(train_arr, test_arr)