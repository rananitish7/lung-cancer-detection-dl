# src/pipelines/split_data_pipeline.py

from zenml.pipelines import pipeline
from zenml.steps import step, Output, BaseParameters
import numpy as np
from src.data.split_data import load_data, split_data, save_split_data

class SplitDataParams(BaseParameters):
    output_dir: str
    save_dir: str

@step
def load_data_step(params: SplitDataParams) -> Output(images=np.ndarray, labels=np.ndarray):
    return load_data(params.output_dir)

@step
def split_data_step(data: np.ndarray, labels: np.ndarray) -> Output(
    X_train=np.ndarray, y_train=np.ndarray, X_val=np.ndarray, y_val=np.ndarray, X_test=np.ndarray, y_test=np.ndarray
):
    return split_data(data, labels)

@step
def save_data_step(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: SplitDataParams,
) -> None:
    save_split_data(X_train, y_train, X_val, y_val, X_test, y_test, params.save_dir)

@pipeline
def split_data_pipeline(load_step, split_step, save_step):
    data, labels = load_step()
    X_train, y_train, X_val, y_val, X_test, y_test = split_step(data, labels)
    save_step(X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    output_dir = r"data\processed"
    save_dir = r"data\split"

    # Instantiate parameters for the load and save steps
    params = SplitDataParams(output_dir=output_dir, save_dir=save_dir)

    split_data_pipeline(
        load_step=load_data_step(params=params),
        split_step=split_data_step(),
        save_step=save_data_step(params=params)
    ).run()
