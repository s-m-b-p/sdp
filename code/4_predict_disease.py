import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import joblib
import matplotlib.pyplot as plt
import random

# Function to preprocess an image
def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess an image by resizing, converting to grayscale, and normalizing.
    """
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    # Resize and normalize
    resized_image = cv2.resize(image, target_size)
    normalized_image = resized_image / 255.0  # Normalize to [0, 1]
    return normalized_image

# Function to extract LBP features from a preprocessed image
def extract_lbp_features(image, radius=3, n_points=8):
    """
    Extract Local Binary Pattern (LBP) features from a preprocessed image.
    """
    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    
    # Histogram of LBP features
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize histogram
    return hist

# Function to predict disease names for a single randomly selected image
def predict_random_disease(folder_path, model_path, label_map):
    """
    Predicts the disease category for a randomly selected image from the folder.
    """
    # Load the trained model
    model = joblib.load(model_path)

    # List all files in the folder
    all_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not all_images:
        raise ValueError("No valid images found in the folder.")

    # Randomly select an image
    img_path = random.choice(all_images)
    print(f"\nRandomly Selected Image: {img_path}")

    try:
        # Load the original image for visualization
        original_image = cv2.imread(img_path)
        if original_image is None:
            raise ValueError(f"Unable to read image: {img_path}")
        
        # Convert BGR to RGB for visualization
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Preprocess the image
        preprocessed_image = preprocess_image(img_path)

        # Extract LBP features and predict
        lbp_features = extract_lbp_features(preprocessed_image)
        predicted_label = model.predict([lbp_features])[0]
        predicted_label = int(predicted_label)  # Convert np.int64 to int

        # Map numeric label back to disease name
        reverse_label_map = {v: k for k, v in label_map.items()}
        predicted_disease = reverse_label_map[predicted_label]

        # Display the image with prediction details
        plt.figure(figsize=(8, 6))
        plt.imshow(original_image_rgb)
        plt.title(f"Predicted Disease: {predicted_disease}", fontsize=14)
        plt.axis('off')  # Turn off axis
        plt.show()

        # Print prediction details in the console
        print(f"Image: {img_path}\nPredicted Disease: {predicted_disease}\n")

        # Save the path of the randomly selected image to a temporary file
        with open("last_predicted_image.txt", "w") as f:
            f.write(img_path)

    except Exception as e:
        print(f"Error processing image: {img_path}\nError: {e}")

# Define paths and label map
model_path = r"D:\research paper\results\predictions\fish_disease_model_xgboost.pkl"
label_map = {
    "Argulus": 0,
    "Broken antennae and rostrum": 1,
    "EUS": 2,
    "Red Spot": 3,
    "Tail And Fin Rot": 4,
    "THE BACTERIAL GILL ROT": 5
}

# Folder containing test images
test_folder = r"C:\Users\sonof\Downloads\random testing images for model"

# Predict disease for a randomly selected image
predict_random_disease(test_folder, model_path, label_map)