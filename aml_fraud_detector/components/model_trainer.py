import os
import sys
from dataclasses import dataclass
from typing import Dict, Any

from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import (
    save_object, 
    upsampling_train_data, 
    evaluate_models, 
    model_metrics,
    ModelMetrics,
    _serialize_cm
)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainerResult:
    def __init__(self,
                 best_model_name: str,
                 best_params: Dict[str, Any],
                 test_metrics: ModelMetrics,
                 model_path: str):
        self.best_model_name = best_model_name
        self.best_params = best_params
        self.test_metrics = test_metrics
        self.model_path = model_path

    def __iter__(self):
        yield self.best_model_name
        yield self.best_params
        yield self.test_metrics.to_dict()
        yield self.model_path

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self)[key]
        if isinstance(key, str):
            mapping = {
                "best_model_name": self.best_model_name,
                "best_params": self.best_params,
                "test_metrics": self.test_metrics.to_dict(),
                "model_path": self.model_path,
            }
            return mapping[key]
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return ["best_model_name", "best_params", "test_metrics", "model_path"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_model_name": self.best_model_name,
            "best_params": self.best_params,
            "test_metrics": self.test_metrics.to_dict(),
            "model_path": self.model_path
        }

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "best_model_name": self.best_model_name,
            "best_params": self.best_params,
            "test_metrics": self.test_metrics.to_log_dict(),
            "model_path": self.model_path
        }


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info(f"Get Independent features and Dependent feature from Train and Test datasets")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info(f"Imbalance dataset - upsampling the train data")
            X_train_smp, y_train_smp = upsampling_train_data(X_train, y_train)

            models = {
                "Random Forest": RandomForestClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "XGBoost": XGBClassifier()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                },
                "AdaBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                },
                "XGBoost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.05, 0.001],
                }
            }

            train_report, test_report, best_params_report = evaluate_models(
                X_train=X_train_smp, 
                y_train=y_train_smp, 
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params)
            
            models_recall_score = {
                model: metrics_list[0]["Recall"]
                for model, metrics_list in test_report.items()
            }
            logging.info(f"The models and their corresponding Recall score: \n{models_recall_score}")

            best_model_name, best_recall = max(models_recall_score.items(), key=lambda item: item[1])
            logging.info(f"Best Model: {best_model_name} with Recall score: {best_recall}")
            print(f"Best Model: {best_model_name} with Recall score: {best_recall}")
            
            best_model = models[best_model_name]
            best_params = best_params_report[best_model_name]

            save_object(
                 file_path = self.model_trainer_config.trained_model_file_path,
                 obj = best_model
            )
        
            y_test_pred = best_model.predict(X_test)
            test_metrics = model_metrics(y_test, y_test_pred)

            logging.info(f"Model Training completed")
            logging.info(f"Final test metrics for {best_model_name}: {test_metrics.to_log_dict()}")
            print(f"Final test metrics for the best model i.e. {best_model_name}:")
            print(f"  Precision: {test_metrics.precision}")
            print(f"  Recall: {test_metrics.recall}")
            print(f"  F1 score: {test_metrics.f1_score}")
            print(f"  Confusion Matrix:\n{_serialize_cm(test_metrics.confusion_matrix)}")

            result = ModelTrainerResult(
                best_model_name=best_model_name,
                best_params=best_params,
                test_metrics=test_metrics,
                model_path=self.model_trainer_config.trained_model_file_path
            )
            return result

        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise CustomerException(e, sys)
