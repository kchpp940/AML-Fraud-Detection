import os
import sys
import json
import dill
import numpy as np
import pandas as pd
from dataclasses import asdict
from datetime import datetime

from aml_fraud_detector.logger import logging
from aml_fraud_detector.exception import (
    CustomerException,
    ModelLoadingException,
    DataQualityException,
    wrap_exception,
)
from aml_fraud_detector.constants import ErrorCode

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
        logging.error("Exception occurred in save_training_summary", exc_info=True)
        raise wrap_exception(e, error_details=sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved to: {file_path}")

    except Exception as e:
        logging.error(f'Exception occurred in save_object: {e}', exc_info=True)
        raise wrap_exception(e, error_details=sys)


def load_object(file_path):
    try:
        if not os.path.exists(file_path):
            raise ModelLoadingException(
                ErrorCode.MODEL_FILE_NOT_FOUND if "model" in file_path.lower() else ErrorCode.PREPROCESSOR_FILE_NOT_FOUND,
                error_details=sys,
                path=file_path,
            )
        with open(file_path, 'rb') as file_obj:
            obj = dill.load(file_obj)
        logging.info(f"Object loaded from: {file_path}")
        return obj
    except ModelLoadingException:
        raise
    except Exception as e:
        logging.error(f'Exception occurred in load_object: {e}', exc_info=True)
        raise ModelLoadingException(
            ErrorCode.MODEL_CORRUPTED if "model" in file_path.lower() else ErrorCode.PREPROCESSOR_CORRUPTED,
            error_details=sys,
            path=file_path,
            detail=str(e),
        )


def upsampling_train_data(X, y):
    try: 
        sm = SMOTE(sampling_strategy='auto', random_state=42)
        logging.info(f"Before SMOTE: {Counter(y)}")
        X_sm, y_sm = sm.fit_resample(X, y)   
        logging.info(f"After SMOTE: {Counter(y_sm)}")
        logging.info(f"Upsampling the minority class data completed") 
        return X_sm, y_sm
    except Exception as e:
        logging.error("Exception occurred during upsampling", exc_info=True)
        raise wrap_exception(e, error_details=sys)


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
        logging.error("Exception occurred during model evaluation", exc_info=True)
        raise wrap_exception(e, error_details=sys)
    
