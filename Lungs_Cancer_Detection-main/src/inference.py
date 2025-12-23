import os
import numpy as np
import argparse
import mlflow
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
mlflow.set_tracking_uri("http://localhost:5000")

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to load the model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function to make a prediction
def make_prediction(model, img_array):
    preds = model.predict(img_array)
    pred_class = np.argmax(preds, axis=1)[0]
    class_mapping = {0: 'Normal cases', 1: 'Bengin cases', 2: 'Malignant cases'}
    return class_mapping[pred_class]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lung Cancer Detection Inference Script")
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')

    args = parser.parse_args()

    # Start MLflow run
    mlflow.start_run()
    # image_path='D:\LLD\data\raw\Malignant cases\Malignant case (1).jpg'
    # model_path='D:\LLD\models\lung_cancer_detection_model.h5'
    # Load and preprocess the image
    img_array = load_and_preprocess_image(args.image_path)

    # Load the trained model
    model = load_model(args.model_path)

    # Make a prediction
    pred_label = make_prediction(model, img_array)
    
    # Log the input image and prediction
    mlflow.log_param("Input Image", args.image_path)
    mlflow.log_param("Predicted Label", pred_label)

    print(f"Predicted Label: {pred_label}")

    # End MLflow run
    mlflow.end_run()
