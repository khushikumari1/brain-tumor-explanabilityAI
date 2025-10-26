# evaluate_model.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------ CONFIG ------------------
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
TEST_DIR = r'dataset\Testing'
MODEL_PATH = 'brain_tumor_model.h5'
OUTPUT_DIR = os.path.join(os.getcwd(), "evaluation_results")  # Folder to save metrics

os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create folder if not exists

# ------------------ LOAD MODEL ------------------
model = load_model(MODEL_PATH)
print(f"Model '{MODEL_PATH}' loaded successfully!")

# ------------------ LOAD TEST DATA ------------------
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_data.class_indices.keys())
print("Class names:", class_names)

# ------------------ EVALUATE MODEL ------------------
test_loss, test_acc = model.evaluate(test_data, verbose=1)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ------------------ DETAILED METRICS ------------------
y_true = test_data.classes
y_pred_probs = model.predict(test_data)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:")
print(report)

# ------------------ SAVE CONFUSION MATRIX ------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"Confusion matrix saved to: {cm_path}")

# ------------------ SAVE CLASSIFICATION REPORT ------------------
# Save as text file
report_txt_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
with open(report_txt_path, "w") as f:
    f.write(report)
print(f"Classification report (text) saved to: {report_txt_path}")

# Save as image
plt.figure(figsize=(8,4))
plt.text(0, 1, report, fontsize=12, fontfamily='monospace')
plt.axis('off')
plt.tight_layout()
report_img_path = os.path.join(OUTPUT_DIR, "classification_report.png")
plt.savefig(report_img_path)
plt.close()
print(f"Classification report (image) saved to: {report_img_path}")

# ------------------ OPTIONAL: PLOT PREDICTION DISTRIBUTION ------------------
pred_counts = np.bincount(y_pred)
plt.figure(figsize=(6,4))
plt.bar(class_names, pred_counts, color='skyblue')
plt.title("Number of Predictions per Class")
plt.ylabel("Count")
plt.tight_layout()
dist_path = os.path.join(OUTPUT_DIR, "prediction_distribution.png")
plt.savefig(dist_path)
plt.close()
print(f"Prediction distribution saved to: {dist_path}")
