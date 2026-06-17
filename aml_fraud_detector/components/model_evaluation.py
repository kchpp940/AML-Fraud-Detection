import os
import sys
from dataclasses import dataclass
from typing import Optional

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import model_metrics, load_object
from aml_fraud_detector.configuration import TrainingConfig


@dataclass
class ModelEvaluationArtifact:
    precision: float
    recall: float
    f1_score: float
    model_path: str
    mlflow_run_id: str = ""


class ModelEvaluation:
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config
        self._model_path = (
            training_config.artifacts_subpath("model.pkl")
            if training_config
            else os.path.join("artifacts", "model.pkl")
        )
        logging.info("Model evaluation started")

    def eval_metrics(self, y_test, y_pred):
        precision, recall, f1, cm = model_metrics(y_test, y_pred)
        return precision, recall, f1, cm

    def initiate_model_evaluation(self, train_array, test_array) -> ModelEvaluationArtifact:
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            model = load_object(file_path=self._model_path)

            logging.info("model has register")

            mlflow_run_id = ""
            with mlflow.start_run() as run:
                mlflow_run_id = run.info.run_id
                predictions = model.predict(X_test)
                signature = infer_signature(X_test, predictions)
                (precision, recall, f1, cm) = self.eval_metrics(y_test, predictions)

                print(f"Precision: {precision}")
                print(f"Recall:{recall}")
                print(f"F1 score: {f1}")
                print(f"Confusion Matrix: {cm}")

                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)

                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                print(tracking_url_type_store)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestBestModel")
                else:
                    mlflow.sklearn.log_model(model, "model", signature=signature)

            logging.info(f"Model evaluation completed, mlflow_run_id={mlflow_run_id}")

            return ModelEvaluationArtifact(
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                model_path=os.path.abspath(self._model_path),
                mlflow_run_id=mlflow_run_id,
            )
        except Exception as e:
            raise CustomerException(e, sys)
