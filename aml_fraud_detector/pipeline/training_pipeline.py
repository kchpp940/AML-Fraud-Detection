import os
import sys
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging

from aml_fraud_detector.components.data_ingestion import DataIngestion
from aml_fraud_detector.components.data_transformation import DataTransformation
from aml_fraud_detector.components.model_trainer import ModelTrainer
from aml_fraud_detector.components.model_evaluation import ModelEvaluation


if __name__ == "__main__":
    try:
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)

        model_trainer = ModelTrainer()
        trainer_result = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info("=" * 80)
        logging.info("MODEL TRAINER RESULT")
        logging.info("=" * 80)
        logging.info(f"Best Model Name: {trainer_result.best_model_name}")
        logging.info(f"Best Parameters: {trainer_result.best_params}")
        logging.info(f"Test Metrics - Precision: {trainer_result.test_metrics.precision}")
        logging.info(f"Test Metrics - Recall: {trainer_result.test_metrics.recall}")
        logging.info(f"Test Metrics - F1 Score: {trainer_result.test_metrics.f1_score}")
        logging.info(f"Test Metrics - Confusion Matrix:\n{trainer_result.test_metrics.confusion_matrix}")
        logging.info(f"Model Path: {trainer_result.model_path}")
        logging.info("=" * 80)

        print("\n" + "=" * 80)
        print("MODEL TRAINER RESULT")
        print("=" * 80)
        print(f"Best Model Name: {trainer_result.best_model_name}")
        print(f"Best Parameters: {trainer_result.best_params}")
        print(f"Test Metrics:")
        print(f"  Precision: {trainer_result.test_metrics.precision}")
        print(f"  Recall: {trainer_result.test_metrics.recall}")
        print(f"  F1 Score: {trainer_result.test_metrics.f1_score}")
        print(f"  Confusion Matrix:\n{trainer_result.test_metrics.confusion_matrix}")
        print(f"Model Path: {trainer_result.model_path}")
        print("=" * 80 + "\n")

        model_eval_obj = ModelEvaluation()
        eval_metrics = model_eval_obj.initiate_model_evaluation(train_arr, test_arr, trainer_result)

        logging.info("=" * 80)
        logging.info("MODEL EVALUATION RESULT (MLflow)")
        logging.info("=" * 80)
        logging.info(f"Evaluation Test Metrics - Precision: {eval_metrics.precision}")
        logging.info(f"Evaluation Test Metrics - Recall: {eval_metrics.recall}")
        logging.info(f"Evaluation Test Metrics - F1 Score: {eval_metrics.f1_score}")
        logging.info(f"Evaluation Test Metrics - Confusion Matrix:\n{eval_metrics.confusion_matrix}")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Exception occurred in training pipeline: {e}")
        raise CustomerException(e, sys)