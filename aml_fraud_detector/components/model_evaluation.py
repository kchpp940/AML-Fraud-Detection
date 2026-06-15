import os
import sys
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse
from typing import Optional

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import model_metrics, load_object, ModelMetrics
from aml_fraud_detector.components.model_trainer import ModelTrainerResult


class ModelEvaluation:
    def __init__(self):
        logging.info("Model evaluation started")

    def eval_metrics(self, y_true, y_pred) -> ModelMetrics:
        return model_metrics(y_true, y_pred)
    
    def log_metrics_to_mlflow(self, metrics: ModelMetrics):
        mlflow.log_metric("precision", metrics.precision)
        mlflow.log_metric("recall", metrics.recall)
        mlflow.log_metric("f1", metrics.f1_score)

    def initiate_model_evaluation(self, train_array, test_array, trainer_result: Optional[ModelTrainerResult] = None):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:,-1])

            model_path = trainer_result.model_path if trainer_result else os.path.join("artifacts", "model.pkl")
            model = load_object(file_path=model_path)

            logging.info("model has register")

            with mlflow.start_run():
                if trainer_result:
                    mlflow.log_param("best_model_name", trainer_result.best_model_name)
                    for param_name, param_value in trainer_result.best_params.items():
                        mlflow.log_param(param_name, param_value)
                    mlflow.log_param("model_path", trainer_result.model_path)
                    logging.info(f"Logged trainer result to MLflow: {trainer_result.to_log_dict()}")

                predictions = model.predict(X_test)
                signature = infer_signature(X_test, predictions)
                test_metrics = self.eval_metrics(y_test, predictions)

                print(f"Precision: {test_metrics.precision}")
                print(f"Recall: {test_metrics.recall}")
                print(f"F1 score: {test_metrics.f1_score}")
                print(f"Confusion Matrix:\n{test_metrics.confusion_matrix}")

                self.log_metrics_to_mlflow(test_metrics)

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                print(tracking_url_type_store)

                if tracking_url_type_store != "file":
                    model_name = trainer_result.best_model_name if trainer_result else "BestModel"
                    mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
                else:
                    mlflow.sklearn.log_model(model, "model", signature=signature)
            
            return test_metrics

        except Exception as e:
            raise CustomerException(e, sys)
