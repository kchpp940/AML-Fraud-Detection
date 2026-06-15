import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import precision_score, recall_score, f1_score

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import save_object, upsampling_train_data, evaluate_models
from aml_fraud_detector.configuration import TrainingConfig


MODEL_REGISTRY: Dict[str, Any] = {
    "RandomForest": (RandomForestClassifier, "Random Forest"),
    "AdaBoost": (AdaBoostClassifier, "AdaBoost"),
    "GradientBoosting": (GradientBoostingClassifier, "Gradient Boosting"),
    "XGBoost": (XGBClassifier, "XGBoost"),
}


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str


@dataclass
class ModelTrainerArtifact:
    best_model_name: str
    best_model_display_name: str
    best_model: Any
    best_metric_value: float
    best_model_params: Dict[str, Any]
    selection_metric: str
    candidate_models: List[str]
    all_model_metrics: Dict[str, Dict[str, float]]
    model_path: str


class ModelTrainer:
    def __init__(self, training_config: Optional[TrainingConfig] = None):
        self.training_config = training_config or TrainingConfig()
        tc = self.training_config
        self.model_trainer_config = ModelTrainerConfig(
            trained_model_file_path=tc.artifacts_subpath("model.pkl")
        )

    def _build_candidate_models(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        configured_enabled = self.training_config.models.enabled or {}
        configured_params = self.training_config.models.param_grid or {}
        for key, (cls, display) in MODEL_REGISTRY.items():
            enabled = configured_enabled.get(key, False)
            if not enabled:
                continue
            params = configured_params.get(key, {}) or {}
            result[key] = {
                "class": cls,
                "display_name": display,
                "param_grid": params,
            }
        if not result:
            raise CustomerException(
                ValueError("No models enabled in models.enabled"), sys
            )
        return result

    def initiate_model_trainer(
        self, train_array, test_array
    ) -> ModelTrainerArtifact:
        try:
            logging.info("Get Independent features and Dependent feature from Train and Test datasets")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Imbalance dataset - upsampling the train data")
            X_train_smp, y_train_smp = upsampling_train_data(X_train, y_train)

            candidates = self._build_candidate_models()
            candidate_keys = list(candidates.keys())

            models: Dict[str, Any] = {}
            params: Dict[str, Any] = {}
            display_map: Dict[str, str] = {}
            for key in candidate_keys:
                info = candidates[key]
                display_name = info["display_name"]
                display_map[key] = display_name
                models[display_name] = info["class"]()
                params[display_name] = info["param_grid"]

            logging.info(
                f"Training candidate models: {[display_map[k] for k in candidate_keys]}"
            )

            model_report = evaluate_models(
                X_train=X_train_smp,
                y_train=y_train_smp,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )

            selection_metric = self.training_config.models.selection_metric
            train_report, test_report = model_report

            all_model_metrics: Dict[str, Dict[str, float]] = {}
            for display_name, metric_list in test_report.items():
                m = metric_list[0]
                all_model_metrics[display_name] = {
                    "Precision": float(m["Precision"]),
                    "Recall": float(m["Recall"]),
                    "F1 score": float(m["F1 score"]),
                }

            logging.info(f"Model metrics on Test Data: {all_model_metrics}")
            logging.info(f"Selecting best model using metric: {selection_metric}")

            scored = {
                name: vals[selection_metric]
                for name, vals in all_model_metrics.items()
            }
            logging.info(
                f"Models with '{selection_metric}' score: {scored}"
            )

            best_display_name, best_score = max(scored.items(), key=lambda item: item[1])
            best_key = next(k for k, d in display_map.items() if d == best_display_name)

            logging.info(
                f"Best Model: {best_display_name} with {selection_metric} score: {best_score}"
            )
            print(f"Best Model: {best_display_name} with {selection_metric} score: {best_score}")

            best_model = models[best_display_name]

            best_model_params: Dict[str, Any] = {}
            try:
                best_model_params = best_model.get_params(deep=True)
                best_model_params = {
                    k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                    for k, v in best_model_params.items()
                }
            except Exception:
                pass

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            recall_Score = recall_score(y_test, predicted, average='weighted')

            logging.info("Model Training completed")
            logging.info(
                f"Final {selection_metric} score for {best_display_name}: {best_score}"
            )
            print(
                f"Final {selection_metric} score for the best model i.e. "
                f"{best_display_name}: {best_score}"
            )

            return ModelTrainerArtifact(
                best_model_name=best_key,
                best_model_display_name=best_display_name,
                best_model=best_model,
                best_metric_value=float(best_score),
                best_model_params=best_model_params,
                selection_metric=selection_metric,
                candidate_models=[display_map[k] for k in candidate_keys],
                all_model_metrics=all_model_metrics,
                model_path=os.path.abspath(
                    self.model_trainer_config.trained_model_file_path
                ),
            )

        except Exception as e:
            logging.info("Exception occurred at Model Training")
            raise CustomerException(e, sys)
