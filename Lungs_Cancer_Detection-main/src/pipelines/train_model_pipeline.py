from zenml.pipelines import pipeline
from zenml.steps import step, Output, BaseParameters
import numpy as np
from src.models.train_model import load_split_data, create_cnn_model, evaluate_model, save_model
import tensorflow as tf

class TrainModelParams(BaseParameters):
    data_dir: str
    output_dir: str

@step
def load_data_step(params: TrainModelParams) -> Output(X_train=np.ndarray, y_train=np.ndarray, X_val=np.ndarray, y_val=np.ndarray):
    return load_split_data(params.data_dir)

@step
def train_model_step(X_train: np.ndarray, y_train: np.ndarray) -> Output(model=tf.keras.Model):
    model = create_cnn_model(input_shape=X_train.shape[1:])
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    model.fit(train_dataset, epochs=10)
    return model

@step
def evaluate_model_step(model: tf.keras.Model, X_val: np.ndarray, y_val: np.ndarray) -> Output(accuracy=float, report=str):
    accuracy, report = evaluate_model(model, X_val, y_val)
    return accuracy, report

@step
def save_model_step(model: tf.keras.Model, params: TrainModelParams) -> None:
    save_model(model, params.output_dir)

@pipeline
def train_model_pipeline(load_step, train_step, evaluate_step, save_step):
    X_train, y_train, X_val, y_val = load_step()
    model = train_step(X_train, y_train)
    accuracy, report = evaluate_step(model, X_val, y_val)
    save_step(model)

if __name__ == "__main__":
    data_dir = 'data/split'
    output_dir = "models"  # Path to save the trained model

    # Create an instance of the parameters
    params = TrainModelParams(data_dir=data_dir, output_dir=output_dir)

    # Run the pipeline with parameters
    train_model_pipeline(
        load_step=load_data_step(params=params),
        train_step=train_model_step(),
        evaluate_step=evaluate_model_step(),
        save_step=save_model_step(params=params)
    ).run()
