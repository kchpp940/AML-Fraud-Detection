import os
import sys
from typing import Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.configuration import Configuration

from aml_fraud_detector.components.data_ingestion import DataIngestion
from aml_fraud_detector.components.data_transformation import DataTransformation
from aml_fraud_detector.components.model_trainer import ModelTrainer
from aml_fraud_detector.components.model_evaluation import ModelEvaluation


class TrainingPipeline:
    def __init__(self, config_manager: Optional[Configuration] = None):
        self.config_manager = config_manager or Configuration()
        logging.info("TrainingPipeline initialized with Configuration")

    def run_data_ingestion(self):
        logging.info("=" * 70)
        logging.info("STAGE 1: DATA INGESTION")
        logging.info("=" * 70)
        data_ingestion = DataIngestion(
            ingestion_config=self.config_manager.get_data_ingestion_config(),
            config_manager=self.config_manager,
        )
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion output: train={train_data_path}, test={test_data_path}")
        return train_data_path, test_data_path

    def run_data_transformation(self, train_data_path, test_data_path):
        logging.info("=" * 70)
        logging.info("STAGE 2: DATA TRANSFORMATION")
        logging.info("=" * 70)
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(
            f"Data transformation output: train_arr shape={train_arr.shape}, "
            f"test_arr shape={test_arr.shape}"
        )
        return train_arr, test_arr

    def run_model_trainer(self, train_arr, test_arr):
        logging.info("=" * 70)
        logging.info("STAGE 3: MODEL TRAINING")
        logging.info("=" * 70)
        model_trainer = ModelTrainer()
        result = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info("Model training completed")
        return result

    def run_model_evaluation(self, train_arr, test_arr):
        logging.info("=" * 70)
        logging.info("STAGE 4: MODEL EVALUATION")
        logging.info("=" * 70)
        try:
            model_eval_obj = ModelEvaluation()
            result = model_eval_obj.initiate_model_evaluation(train_arr, test_arr)
            logging.info("Model evaluation completed")
            return result
        except Exception as e:
            logging.warning(f"Model evaluation skipped due to: {str(e)}")
            return None

    def run_pipeline(self, run_evaluation: bool = False):
        logging.info("\n" + "#" * 70)
        logging.info("STARTING AML FRAUD DETECTION TRAINING PIPELINE")
        logging.info("#" * 70 + "\n")
        try:
            train_data_path, test_data_path = self.run_data_ingestion()
            train_arr, test_arr = self.run_data_transformation(train_data_path, test_data_path)
            self.run_model_trainer(train_arr, test_arr)
            if run_evaluation:
                self.run_model_evaluation(train_arr, test_arr)
            logging.info("\n" + "#" * 70)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info("#" * 70 + "\n")
        except CustomerException as ce:
            logging.error(f"Training pipeline failed with CustomerException: {str(ce)}")
            raise
        except Exception as e:
            logging.error(f"Training pipeline failed with unexpected error: {str(e)}", exc_info=True)
            raise CustomerException(e, sys)


def main():
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline(run_evaluation=False)
    except Exception as e:
        logging.error(f"Fatal error in training pipeline entry point: {str(e)}")
        raise


if __name__ == "__main__":
    main()
