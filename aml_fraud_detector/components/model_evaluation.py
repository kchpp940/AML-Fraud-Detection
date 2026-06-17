import os
import sys

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from urllib.parse import urlparse

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import model_metrics, load_object


class ModelEvaluation:
    def __init__(self):
        self._model_path = os.path.join("artifacts", "model.pkl")
        logging.info("Model evaluation started")

    def eval_metrics(self, y_test, y_pred):
        precision, recall, f1, cm = model_metrics(y_test, y_pred)
        return precision, recall, f1, cm

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            model = load_object(file_path=self._model_path)

            logging.info("model has register")

            with mlflow.start_run():
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
        except Exception as e:
            raise CustomerException(e, sys)
