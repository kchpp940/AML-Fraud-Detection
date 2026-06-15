import os
import sys
import json
import hashlib
import dill
import numpy as np
import pandas as pd
from dataclasses import asdict
from datetime import datetime

from aml_fraud_detector.logger import logging
from aml_fraud_detector.exception import CustomerException

from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def save_training_summary(file_path: str, summary_obj) -> str:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        data = asdict(summary_obj) if hasattr(summary_obj, "__dataclass_fields__") else dict(summary_obj)
        data["generated_at"] = datetime.now().isoformat()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logging.info(f"Training summary saved to: {file_path}")
        return os.path.abspath(file_path)
    except Exception as e:
        logging.info("Exception Occurred in save_training_summary function utils")
        raise CustomerException(e, sys)


def _compute_file_digest(file_path: str, algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return f"{algorithm}:{h.hexdigest()}"


def _read_feature_schema_version(artifacts_dir: str) -> str:
    meta_path = os.path.join(artifacts_dir, "feature_metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return meta.get("contract_version", "unknown")
        except Exception:
            return "unknown"
    return "unknown"


def save_model_metadata(
    artifacts_dir: str,
    data_source_path: str,
    best_model_name: str,
    best_model_params: dict,
    selection_metric: str,
    best_metric_value: float,
    all_model_metrics: dict,
    model_path: str,
) -> str:
    try:
        os.makedirs(artifacts_dir, exist_ok=True)

        data_digest = ""
        if os.path.isfile(data_source_path):
            data_digest = _compute_file_digest(data_source_path)

        feature_schema_version = _read_feature_schema_version(artifacts_dir)

        existing_path = os.path.join(artifacts_dir, "model_metadata.json")
        version = 1
        if os.path.isfile(existing_path):
            try:
                with open(existing_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                version = prev.get("model_version", 0) + 1
            except Exception:
                version = 1

        metadata = {
            "model_version": version,
            "training_time": datetime.now().isoformat(),
            "data_file": os.path.abspath(data_source_path),
            "data_file_digest": data_digest,
            "feature_schema_version": feature_schema_version,
            "best_model_name": best_model_name,
            "best_model_params": best_model_params,
            "selection_metric": selection_metric,
            "best_metric_value": best_metric_value,
            "all_model_metrics": all_model_metrics,
            "artifact_path": os.path.abspath(model_path),
        }

        out_path = os.path.join(artifacts_dir, "model_metadata.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        logging.info(f"Model metadata saved to: {out_path} (version={version})")
        return os.path.abspath(out_path)
    except Exception as e:
        logging.info("Exception occurred in save_model_metadata")
        raise CustomerException(e, sys)


def load_model_metadata(artifacts_dir: str = "artifacts") -> dict:
    try:
        meta_path = os.path.join(artifacts_dir, "model_metadata.json")
        if not os.path.isfile(meta_path):
            logging.warning(f"model_metadata.json not found at: {meta_path}")
            return {}
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        logging.info(
            f"Model metadata loaded: version={metadata.get('model_version')}, "
            f"best_model={metadata.get('best_model_name')}"
        )
        return metadata
    except Exception as e:
        logging.warning(f"Failed to load model_metadata.json: {e}")
        return {}


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


def model_metrics(y_pred, y_test):
    try:     
        precision = precision_score(y_pred, y_test, average='weighted')
        recall = recall_score(y_pred, y_test, average='weighted')
        f1 = f1_score(y_pred, y_test, average='weighted')
        # Compute confusion matrix
        cm = confusion_matrix(y_pred, y_test) 
        return precision, recall, f1, cm
    except Exception as e:
        logging.info(f"Exception occured during metrics calculation")


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        train_report = {}
        test_report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            # Initialize StratifiedKFold with 5 folds
            # Stratified K-Fold ensures that each fold has the same proportion of classes as the entire dataset. 
            skf = StratifiedKFold(n_splits=3)

            # Grid Search
            logging.info(f"Grid Search started for {model}")
            # 
            gs = GridSearchCV(model, param, cv=skf, n_jobs=-1)
            gs.fit(X_train, y_train)
            logging.info(f"Grid Search completed for {model}")

            # Setting model with best hyperparameters
            logging.info(f"Best parameters: {gs.best_params_} for {model}")
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # Predict on Train data
            y_train_pred = model.predict(X_train)
            # Predict Test data
            y_test_pred = model.predict(X_test)

            # Get evaluation metrics for train and test data
            logging.info(f"Obtaining evaluation metrics for {model} by using best hyperparameters")
            precision_train, recall_train, f1_train, cm_train = model_metrics(y_train_pred, y_train)
            train_model_score = []
            train_model_score.append({
                "Precision" : precision_train,
                "Recall" : recall_train,
                "F1 score": f1_train,
                "Confusion Matrix": cm_train
            })
            train_report[list(models.keys())[i]] = train_model_score
            
            precision_test, recall_test, f1_test, cm_test = model_metrics(y_test_pred, y_test)
            test_model_score = []
            test_model_score.append({
                "Precision" : precision_test,
                "Recall" : recall_test,
                "F1 score": f1_test,
                "Confusion Matrix": cm_test
            })
            test_report[list(models.keys())[i]] = test_model_score

        logging.info(f"\n Metrics calculation on Train Data: \n{train_report}")
        
        logging.info(f"\n Metrics calculation on Test Data: \n{test_report}")
        return train_report, test_report

    except Exception as e:
        logging.info(f"Exception occured during model training")
        raise CustomerException(e, sys)
    
