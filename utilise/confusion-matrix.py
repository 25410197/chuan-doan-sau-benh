import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# Get predictions
print("Making predictions...")
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={'label': 'Count'},
    annot_kws={'size': 12}
)
plt.title('Confusion Matrix - Pepper Disease Classification', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save and show
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Confusion matrix saved as 'confusion_matrix.png'")
plt.show()