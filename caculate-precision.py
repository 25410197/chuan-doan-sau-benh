import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 32

class_names = [
    "anthracnose",
    "healthy", 
    "leaf_curl",
    "leaf_spot",
    "nutrient_deficiency"
]

# Load model
print("Loading model...")
model = tf.keras.models.load_model("pepper_disease_model.keras")

# Load validation data
print("Loading validation data...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_resized/val",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get true labels and predictions
print("Making predictions...")
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate metrics
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("\n" + "="*50)
print("METRICS")
print("="*50)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)
print(confusion_matrix(y_true, y_pred))