import os
import numpy as np
import argparse
import mlflow
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, classification_report

mlflow.set_tracking_uri("http://localhost:5000")

def load_split_data(data_dir):
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

    print(f"Original X_train shape: {X_train.shape}")
    print(f"Original X_val shape: {X_val.shape}")

    if len(X_train.shape) != 4 or len(X_val.shape) != 4:
        raise ValueError("X_train and X_val must be 4D arrays (samples, height, width, channels)")

    return X_train, y_train, X_val, y_val

def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(3, activation='softmax')  # Assuming 3 classes: Normal, Benign, Malignant
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_val, y_val):
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)
    predictions = model.predict(val_dataset)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y_val, predicted_classes)
    report = classification_report(y_val, predicted_classes)
    return accuracy, report

def save_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, 'lung_cancer_model.h5'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the lung cancer detection model")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with split data')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save the trained model')
    args = parser.parse_args()

    mlflow.start_run()

    X_train, y_train, X_val, y_val = load_split_data(args.data_dir)

    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

    model = create_cnn_model(input_shape=X_train.shape[1:])
    model.fit(train_dataset, epochs=5, validation_data=val_dataset)
    
    accuracy, report = evaluate_model(model, X_val, y_val)
    save_model(model, args.output_dir)

    mlflow.log_param("data_dir", args.data_dir)
    mlflow.log_param("output_dir", args.output_dir)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_text(report, "classification_report.txt")
    mlflow.log_artifact(os.path.join(args.output_dir, 'lung_cancer_model.h5'))

    mlflow.end_run()
