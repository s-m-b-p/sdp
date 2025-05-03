import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import LeaveOneOut
import joblib
import matplotlib.pyplot as plt
import os

# Load features and labels
features_dir = "D:/research paper/results/features"

X = np.load(os.path.join(features_dir, "train_features.npy"))  # Combined training data
y = np.load(os.path.join(features_dir, "train_labels.npy"))    # Combined training labels

# Initialize LOOCV
loo = LeaveOneOut()

# Lists to store predictions and true labels
y_true, y_pred = [], []

print("Performing Leave-One-Out Cross-Validation...")
for train_index, test_index in loo.split(X):
    # Split data into training and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train XGBoost classifier
    clf = XGBClassifier(
        n_estimators=100,          # Number of boosting rounds
        learning_rate=0.01,       # Smaller learning rate for better generalization
        max_depth=4,              # Reduced depth to prevent overfitting
        subsample=0.7,            # Use 70% of samples per tree
        colsample_bytree=0.7,     # Use 70% of features per tree
        gamma=0.1,                # Minimum loss reduction for a split
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1.0,           # L2 regularization
        random_state=42,
        use_label_encoder=False   # Suppress warnings about label encoding
    )

    # Train the model on the training set
    clf.fit(X_train, y_train)

    # Predict on the test set (single sample in LOOCV)
    y_pred_fold = clf.predict(X_test)

    # Append true and predicted labels
    y_true.append(y_test[0])  # Single test sample
    y_pred.append(y_pred_fold[0])

# Evaluate overall performance
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy (LOOCV): {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[
    "Argulus",
    "Broken antennae and rostrum",
    "EUS",
    "Red Spot",
    "Tail And Fin Rot",
    "THE BACTERIAL GILL ROT"
]))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
    "Argulus",
    "Broken antennae and rostrum",
    "EUS",
    "Red Spot",
    "Tail And Fin Rot",
    "THE BACTERIAL GILL ROT"
])
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Save the trained model (optional: train on full data after LOOCV)
final_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.7,
    colsample_bytree=0.7,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False
)
final_model.fit(X, y)  # Train on the entire dataset

predictions_dir = "D:/research paper/results/predictions"
os.makedirs(predictions_dir, exist_ok=True)

model_path = os.path.join(predictions_dir, "fish_disease_model_xgboost.pkl")
joblib.dump(final_model, model_path)
print(f"\nFinal model saved to {model_path}")