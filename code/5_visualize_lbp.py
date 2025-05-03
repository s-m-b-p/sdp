from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import cv2

def visualize_lbp(image_path, radius=3, n_points=8):
    """
    Visualizes Local Binary Pattern (LBP) texture patterns for a given image.
    """
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    
    # Display the original image and its LBP texture patterns
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")  
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(lbp, cmap="gray")
    plt.title("LBP Texture Patterns")
    plt.axis("off")
    
    plt.show()

# Read the path of the last predicted image from the temporary file
try:
    with open("last_predicted_image.txt", "r") as f:
        img_path = f.read().strip()
except FileNotFoundError:
    raise FileNotFoundError("Temporary file 'last_predicted_image.txt' not found. Run the prediction script first.")

# Visualize LBP for the last predicted image
visualize_lbp(img_path)