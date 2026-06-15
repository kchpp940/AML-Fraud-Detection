import os
import sys
import dill
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Any

from aml_fraud_detector.logger import logging
from aml_fraud_detector.exception import CustomerException

from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def _serialize_cm(cm: Any) -> Any:
    if hasattr(cm, 'tolist'):
        return cm.tolist()
    return cm


@dataclass(frozen=True)
class ModelMetrics:
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Any

    def __iter__(self):
        yield self.precision
        yield self.recall
        yield self.f1_score
        yield _serialize_cm(self.confusion_matrix)

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self)[key]
        return getattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": _serialize_cm(self.confusion_matrix)
        }

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "Precision": self.precision,
            "Recall": self.recall,
            "F1 score": self.f1_score,
            "Confusion Matrix": _serialize_cm(self.confusion_matrix)
        }

    def to_legacy_dict(self) -> Dict[str, Any]:
        return self.to_log_dict()

    def to_legacy_list(self):
        return [self.to_log_dict()]


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.info(f'Exception Occured in save_object function utils')
        raise CustomerException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.info(f'Exception Occured in load_object function utils')
        raise CustomerException(e, sys)


def upsampling_train_data(X, y):
    try: 
        sm = SMOTE(sampling_strategy='auto', random_state=42)
        logging.info(f"Before SMOTE: {Counter(y)}")
        X_sm, y_sm = sm.fit_resample(X, y)   
        logging.info(f"After SMOTE: {Counter(y_sm)}")
        logging.info(f"Upsampling the minority class data completed") 
        return X_sm, y_sm
    except Exception as e:
        logging.info(f"Exception occured during upsampling the minority class")
        raise CustomerException(e, sys)


def model_metrics(y_true, y_pred) -> ModelMetrics:
    try:     
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred) 
        return ModelMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm
        )
    except Exception as e:
        logging.info(f"Exception occured during metrics calculation")
        raise


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        train_report: Dict[str, Any] = {}
        test_report: Dict[str, Any] = {}
        best_params_report: Dict[str, Dict[str, Any]] = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            model_name = list(models.keys())[i]

            skf = StratifiedKFold(n_splits=3)

            logging.info(f"Grid Search started for {model}")
            gs = GridSearchCV(model, param, cv=skf, n_jobs=-1)
            gs.fit(X_train, y_train)
            logging.info(f"Grid Search completed for {model}")

            logging.info(f"Best parameters: {gs.best_params_} for {model}")
            best_params_report[model_name] = gs.best_params_
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            logging.info(f"Obtaining evaluation metrics for {model} by using best hyperparameters")
            train_metrics = model_metrics(y_train, y_train_pred)
            train_report[model_name] = train_metrics.to_legacy_list()
            
            test_metrics = model_metrics(y_test, y_test_pred)
            test_report[model_name] = test_metrics.to_legacy_list()

            logging.info(f"[{model_name}] Train metrics: {train_metrics.to_log_dict()}")
            logging.info(f"[{model_name}] Test metrics: {test_metrics.to_log_dict()}")

        logging.info(f"\n Metrics calculation on Train Data: \n{train_report}")
        logging.info(f"\n Metrics calculation on Test Data: \n{test_report}")
        logging.info(f"\n Best parameters for each model: \n{best_params_report}")
        return train_report, test_report, best_params_report

    except Exception as e:
        logging.info(f"Exception occured during model training")
        raise CustomerException(e, sys)
    
