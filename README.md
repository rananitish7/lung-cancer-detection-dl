# lung-cancer-detection-dl
Deep Learning–based Lung Cancer Detection system using CT scan images to assist early diagnosis through CNN models.

##  Project Overview

Lung cancer is one of the most life-threatening diseases worldwide. Early detection plays a crucial role in improving patient survival rates.  
This project uses **Deep Learning (Convolutional Neural Networks)** to automatically detect lung cancer from **CT scan images**, helping doctors in faster and more accurate diagnosis. 

## How to Setup and Run the Code
- Python 3.6 
- Pip (Python package installer)
- Virtual environment (recommended)

 - Create a virtual environment to isolate dependencies. Activate it depending on your operating system (Linux/Mac uses source venv/bin/activate, while Windows uses venv\Scripts\activate).
 
- Install all required packages using the provided requirements file.
Dataset Organization

 For the dataset 
                    https://github.com/hamdalla93/The-IQ-OTHNCCD-lung-cancer-dataset.git

The dataset should be arranged in a structured format before training:
data/raw/
├── Normal cases
├── Benign cases
└── Malignant cases


Data Version Control
To ensure reproducibility, the dataset is tracked using DVC. Initialize DVC, configure a remote storage (local or cloud), and push the raw data so that all team members can access the same versioned dataset.

Workflow
- Training
- Run the training script to build the CNN model.
- Experiments are logged using MLflow, capturing parameters, metrics, and artifacts.
- Evaluation
- The trained model is tested on the dataset.
- Metrics such as accuracy, precision, recall, and F1-score are recorded to assess performance.
- Inference
- A separate script allows classification of new CT scan images using the trained model.
- Interactive Predictions
- A Streamlit application provides a user-friendly interface for uploading images and viewing predictions in real time.

Model Performance

The model's performance is evaluated using the following metrics:
- **Accuracy**: 99%
- **Precision**: 
  - Normal cases: 1.00
  - Benign cases: 1.00
  - Malignant cases: 0.99
- **Recall**: 
  - Normal cases: 0.98
  - Benign cases: 1.00
  - Malignant cases: 1.00
- **F1-Score**: 
  - Normal cases: 0.99
  - Benign cases: 1.00
  - Malignant cases: 0.99
    
Confusion Matrix:
[[59, 0, 1],
 [0, 18, 0],
 [0, 0, 87]]



## Technical Challenges

Developing a lung cancer detection model involves several challenges:
- **Data Preprocessing**: Handling and preprocessing large volumes of medical imaging data.
- **Model Training**: Choosing the right architecture and hyperparameters for the CNN.
- **Evaluation**: Ensuring the model's performance is robust and generalizes well to unseen data.


Significance of the Project-

Lung cancer remains one of the leading causes of cancer-related deaths globally. Early detection is critical for improving survival rates. This project demonstrates how machine learning can support medical professionals by providing fast, reliable predictions, ultimately reducing healthcare costs and saving lives.

Tools and Technologies
- MLflow: Tracks experiments, logs metrics, and manages artifacts.
- ZenML: Orchestrates machine learning pipelines for reproducibility and scalability.
- DVC: Handles dataset and model versioning, enabling collaborative workflows.
- GitHub: Provides version control and supports team collaboration on the codebase.

Conclusion
This project showcases the integration of deep learning with MLOps practices to tackle a critical healthcare challenge. By combining robust model training, reproducible workflows, and interactive deployment, it highlights the potential of AI in assisting early detection of lung cancer.


