import os
import numpy as np
import cv2
import argparse
import mlflow

# Set the tracking URI for MLflow
mlflow.set_tracking_uri("http://localhost:5000")

def load_jpg_images(data_dir):
    images = []
    labels = []
    class_map = {'Bengin cases': 1, 'Malignant cases': 2, 'Normal cases': 0}

    # Iterate through the class_map to load images and labels
    for class_name, class_label in class_map.items():
        class_dir = os.path.join(data_dir, class_name)
        
        # Check if the class directory exists
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
        
        print(f"Processing class: {class_name} with label: {class_label}")
        
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.endswith(".jpg"):  # Ensure we only process JPG files
                    jpg_path = os.path.join(root, file)
                    try:
                        img_array = cv2.imread(jpg_path)  # Load image
                        img_array = cv2.resize(img_array, (224, 224))  # Resize images
                        img_array = img_array / 255.0  # Normalize pixel values
                        images.append(img_array)
                        labels.append(class_label)
                        print(f"Loaded image: {jpg_path} | Shape: {img_array.shape}")  # Debug print
                    except Exception as e:
                        print(f"Error processing file {jpg_path}: {e}")
    
    print(f"Total images loaded: {len(images)}, Total labels loaded: {len(labels)}")
    
    return np.array(images), np.array(labels)

def save_preprocessed_data(images, labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
    np.save(os.path.join(output_dir, 'images.npy'), images)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess JPG images")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with raw JPG images')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save processed data')
    args = parser.parse_args()
    
    mlflow.start_run()  # Start an MLflow run

    images, labels = load_jpg_images(args.data_dir)  # Load images and labels
    save_preprocessed_data(images, labels, args.output_dir)  # Save processed data

    # Log parameters and artifacts in MLflow
    mlflow.log_param("data_dir", args.data_dir)
    mlflow.log_param("output_dir", args.output_dir)
    mlflow.log_artifacts(args.output_dir)  # Log the output directory containing artifacts

    mlflow.end_run()  # End the MLflow run
