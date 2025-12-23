# src/data/split_data.py

import os
import numpy as np
import argparse
import mlflow
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")

def load_data(output_dir):
    images = np.load(os.path.join(output_dir, 'images.npy'))
    labels = np.load(os.path.join(output_dir, 'labels.npy'))
    return images, labels

def split_data(data: np.ndarray, labels: np.ndarray) -> tuple:
    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def save_split_data(X_train, y_train, X_val, y_val, X_test, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split preprocessed data")
    parser.add_argument('--output-dir', type=str, required=True, help='Directory with processed data')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save split data')
    args = parser.parse_args()

    mlflow.start_run()

    images, labels = load_data(args.output_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(images, labels)
    save_split_data(X_train, y_train, X_val, y_val, X_test, y_test, args.save_dir)

    # Log parameters and artifacts
    mlflow.log_param("output_dir", args.output_dir)
    mlflow.log_param("save_dir", args.save_dir)
    mlflow.log_artifacts(args.save_dir)

    mlflow.end_run()
