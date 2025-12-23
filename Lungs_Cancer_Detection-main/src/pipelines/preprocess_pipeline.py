# src/pipelines/preprocess_pipeline.py

from zenml.pipelines import pipeline
from zenml.steps import step, Output, BaseParameters
import numpy as np
from src.data.preprocess import load_jpg_images, save_preprocessed_data

class LoadImagesParams(BaseParameters):
    data_dir: str

class SaveDataParams(BaseParameters):
    output_dir: str

@step
def load_images_step(params: LoadImagesParams) -> Output(images=np.ndarray, labels=np.ndarray):
    images, labels = load_jpg_images(params.data_dir)
    return images, labels

@step
def save_data_step(images: np.ndarray, labels: np.ndarray, params: SaveDataParams) -> None:
    save_preprocessed_data(images, labels, params.output_dir)

@pipeline
def preprocess_pipeline(load_step, save_step):
    images, labels = load_step()
    save_step(images, labels)

if __name__ == "__main__":
    data_dir = r"data\raw"
    output_dir = r"data\processed"

    preprocess_pipeline(
        load_step=load_images_step(params=LoadImagesParams(data_dir=data_dir)),
        save_step=save_data_step(params=SaveDataParams(output_dir=output_dir))
    ).run()
