# src/pipelines/evaluate_pipeline.py

from zenml.pipelines import pipeline
from zenml.steps import step, Output, BaseParameters
import os
from src.models.evaluate import load_test_data, load_model_from_h5, evaluate_model, log_metrics
import numpy as np
import tensorflow as tf
import numpy as np
class EvaluateModelParams(BaseParameters):
    data_dir: str
    model_path: str

@step
def load_test_data_step(params: EvaluateModelParams) -> Output(X_test=np.ndarray, y_test=np.ndarray):
    X_test, y_test = load_test_data(params.data_dir)
    return X_test, y_test

@step
def load_model_step(params: EvaluateModelParams) -> Output(model=tf.keras.Model):
    model = load_model_from_h5(params.model_path)
    return model  # Correctly return the model

@step
def evaluate_test_model_step(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> Output(metrics=dict):
    metrics = evaluate_model(model, X_test, y_test)
    return metrics

@step
def log_metrics_step(metrics: dict) -> None:
    log_metrics(metrics)

@pipeline
def evaluate_pipeline(load_test_data, load_model, evaluate_test_model, log_metrics):
    X_test, y_test = load_test_data()
    model = load_model()
    metrics = evaluate_test_model(model, X_test, y_test)
    log_metrics(metrics)

if __name__ == "__main__":
    data_dir = 'data/split'
    model_path = 'models/lung_cancer_model.h5'
    
    evaluate_pipeline(
        load_test_data=load_test_data_step(params=EvaluateModelParams(data_dir=data_dir, model_path=model_path)),
        load_model=load_model_step(params=EvaluateModelParams(data_dir=data_dir, model_path=model_path)),
        evaluate_test_model=evaluate_test_model_step(),
        log_metrics=log_metrics_step()
    ).run()
