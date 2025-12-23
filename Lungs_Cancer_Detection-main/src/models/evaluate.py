# src/models/evaluate.py

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from tensorflow.keras.models import load_model
import mlflow
import os
mlflow.set_tracking_uri("http://localhost:5000")

def load_test_data(data_dir):
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    return X_test, y_test

def load_model_from_h5(model_path):
    return load_model(model_path)

def evaluate_model(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "roc_auc": roc_auc_score(y_test, model.predict(X_test), multi_class='ovr'),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred)
    }
    
    return metrics

def log_metrics(metrics):
    mlflow.start_run()
    
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    mlflow.log_metric("f1_score", metrics["f1_score"])
    mlflow.log_metric("roc_auc", metrics["roc_auc"])
    mlflow.log_text(str(metrics["confusion_matrix"]), "confusion_matrix.txt")
    mlflow.log_text(metrics["classification_report"], "classification_report.txt")
    
    mlflow.end_run()
