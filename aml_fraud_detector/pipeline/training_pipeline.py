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
        trainer_dict = trainer_result.to_log_dict()
        test_metrics_log = trainer_dict["test_metrics"]

        logging.info("=" * 80)
        logging.info("MODEL TRAINER RESULT")
        logging.info("=" * 80)
        logging.info(f"Best Model Name: {trainer_dict['best_model_name']}")
        logging.info(f"Best Parameters: {trainer_dict['best_params']}")
        logging.info(f"Test Metrics - Precision: {test_metrics_log['Precision']}")
        logging.info(f"Test Metrics - Recall: {test_metrics_log['Recall']}")
        logging.info(f"Test Metrics - F1 Score: {test_metrics_log['F1 score']}")
        logging.info(f"Test Metrics - Confusion Matrix:\n{test_metrics_log['Confusion Matrix']}")
        logging.info(f"Model Path: {trainer_dict['model_path']}")
        logging.info("=" * 80)

        print("\n" + "=" * 80)
        print("MODEL TRAINER RESULT")
        print("=" * 80)
        print(f"Best Model Name: {trainer_dict['best_model_name']}")
        print(f"Best Parameters: {trainer_dict['best_params']}")
        print(f"Test Metrics:")
        print(f"  Precision: {test_metrics_log['Precision']}")
        print(f"  Recall: {test_metrics_log['Recall']}")
        print(f"  F1 Score: {test_metrics_log['F1 score']}")
        print(f"  Confusion Matrix:\n{test_metrics_log['Confusion Matrix']}")
        print(f"Model Path: {trainer_dict['model_path']}")
        print("=" * 80 + "\n")

        model_eval_obj = ModelEvaluation()
        eval_metrics = model_eval_obj.initiate_model_evaluation(train_arr, test_arr, trainer_result)
        eval_metrics_log = eval_metrics.to_log_dict()

        logging.info("=" * 80)
        logging.info("MODEL EVALUATION RESULT (MLflow)")
        logging.info("=" * 80)
        logging.info(f"Evaluation Test Metrics - Precision: {eval_metrics_log['Precision']}")
        logging.info(f"Evaluation Test Metrics - Recall: {eval_metrics_log['Recall']}")
        logging.info(f"Evaluation Test Metrics - F1 Score: {eval_metrics_log['F1 score']}")
        logging.info(f"Evaluation Test Metrics - Confusion Matrix:\n{eval_metrics_log['Confusion Matrix']}")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Exception occurred in training pipeline: {e}")
        raise CustomerException(e, sys)