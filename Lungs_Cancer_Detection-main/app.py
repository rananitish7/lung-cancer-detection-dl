import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Class mapping
class_map = {0: 'Normal cases', 1: 'Bengin cases', 2: 'Malignant cases'}

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
    return class_map[pred_class]

# Streamlit app
st.title("Lung Cancer Detection")

st.write("Upload a CT scan image to classify it as Normal, Benign, or Malignant.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Model path
model_path = r'models\lung_cancer_model.h5'

if uploaded_file is not None:
    # Save uploaded file to a temporary file
    temp_file = "temp_image.jpg"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and preprocess the image
    img_array = load_and_preprocess_image(temp_file)
    
    # Load the model
    model = load_model(model_path)
    
    # Make a prediction
    pred_label = make_prediction(model, img_array)
    
    # Display the image and prediction
    st.image(temp_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Label: {pred_label}")
    
    # Remove the temporary file
    os.remove(temp_file)
