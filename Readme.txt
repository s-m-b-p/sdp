Fish Disease Classification Project
This project aims to classify fish diseases using machine learning. The pipeline includes preprocessing, feature extraction, model training, and prediction. Below is a detailed explanation of the scripts, their purpose, and how to use them.

Project Overview
The project consists of five Python scripts that handle different stages of the machine learning pipeline:

Image Preprocessing : Resizes, normalizes, and saves images in grayscale format.
Feature Extraction : Extracts Local Binary Pattern (LBP) features from preprocessed images and prepares the dataset.
Model Training with LOOCV : Trains an XGBoost classifier using Leave-One-Out Cross-Validation (LOOCV) and evaluates its performance.
Random Image Prediction : Predicts the disease category for a randomly selected image from a test folder.
LBP Visualization : Visualizes the LBP texture patterns for the same image used in the random prediction step.
Scripts Explanation
1. Image Preprocessing
Resizes all images in the input directory, converts them to grayscale, normalizes pixel values, and saves them to the output directory. This ensures uniformity across the dataset and prepares the images for feature extraction.

Output:

Preprocessed images are saved in D:/research paper/results/preprocessed/train and D:/research paper/results/preprocessed/test.
2. Feature Extraction
Extracts Local Binary Pattern (LBP) features from preprocessed images and prepares the dataset for training and testing. Each image is represented as a histogram of LBP features, which captures texture-based patterns. Numeric labels are assigned to each disease class, and the features and labels are saved as .npy files.

Output:

Feature files are saved in D:/research paper/results/features:
train_features.npy
train_labels.npy
test_features.npy
test_labels.npy.
3. Model Training with LOOCV
Trains an XGBoost classifier using Leave-One-Out Cross-Validation (LOOCV) to evaluate the model's performance. The script prints overall accuracy, a classification report, and a confusion matrix. A final model is trained on the entire dataset and saved for future use.

Output:

Final trained model is saved as fish_disease_model_xgboost.pkl in D:/research paper/results/predictions.
4. Random Image Prediction
Predicts the disease category for a randomly selected image from a specified test folder. The script preprocesses the image, extracts LBP features, and uses the trained model to predict the disease. The image is displayed with the predicted disease name, and details are printed to the console. Additionally, the path of the randomly selected image is saved to a temporary file (last_predicted_image.txt) for use in the LBP visualization step.

Output:

Displays the randomly selected image with the predicted disease name.
Saves the path of the image to last_predicted_image.txt.
5. LBP Visualization
Visualizes the Local Binary Pattern (LBP) texture patterns for the same image that was randomly selected and predicted in the previous step. This script reads the image path from the temporary file (last_predicted_image.txt) and displays the original image alongside its LBP texture patterns.

Output:

Displays the original image and its LBP texture patterns side by side.
Dependencies
The following Python libraries are required:

numpy
scikit-learn
scikit-image
xgboost
matplotlib
opencv-python
joblib

Directory Structure

The project is organized as follows:

D:/research paper/
├── dataset/                  # Raw dataset organized by disease folders
├── results/
│   ├── preprocessed/         # Preprocessed images (grayscale, resized, normalized)
│   │   ├── train/
│   │   └── test/
│   ├── features/             # Extracted LBP features and labels
│   │   ├── train_features.npy
│   │   ├── train_labels.npy
│   │   ├── test_features.npy
│   │   └── test_labels.npy
│   └── predictions/          # Trained model
│       └── fish_disease_model_xgboost.pkl
└── scripts/                  # Python scripts
    ├── preprocess_images.py
    ├── extract_features.py
    ├── train_model_loocv.py
    ├── predict_random_image.py
    └── visualize_lbp.py



