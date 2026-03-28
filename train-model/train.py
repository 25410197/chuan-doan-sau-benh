import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import logging
import time

# Suppress PIL DecompressionBombWarning
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import warnings
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

# =====================
# LOGGING CONFIG
# =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('epoch_training_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =====================
# CUSTOM CALLBACK FOR EPOCH LOGGING
# =====================
class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, total_steps_per_epoch=1):
        super().__init__()
        self.total_steps_per_epoch = total_steps_per_epoch
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        loss = logs.get('loss', 0)
        accuracy = logs.get('accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        val_accuracy = logs.get('val_accuracy', 0)
        
        epoch_msg = (f"Epoch {epoch + 1} | "
                     f"loss: {loss:.4f} - accuracy: {accuracy:.4f} - "
                     f"val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
        
        logger.info(epoch_msg)

IMG_SIZE = 224
BATCH_SIZE = 32

# ======================
# LOAD DATASET
# ======================
train_ds = tf.keras.utils.image_dataset_from_directory(
    "../dataset_cropped_new/train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "../dataset_cropped_new/val",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False  # QUAN TRỌNG: không shuffle val để evaluation đúng
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

# ======================
# CLASS WEIGHT (QUAN TRỌNG)
# ======================
y_train = np.concatenate([y for x, y in train_ds], axis=0)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Class weights:", class_weights)

# ======================
# DATA AUGMENTATION
# ======================
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.2),
#     layers.RandomZoom(0.2),
#     layers.RandomContrast(0.2),
# ])

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2),
])

# ======================
# BASE MODEL
# ======================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# base_model = EfficientNetB0(
#     input_shape=(224,224,3),
#     include_top=False,
#     weights="imagenet"
# )

base_model.trainable = False

# ======================
# MODEL
# ======================
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
# Rescaling thay cho preprocess_input: pixel [0,255] -> [-1,1]
x = layers.Rescaling(1./127.5, offset=-1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# PHASE 1 TRAIN
# ======================
early_stop_1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

epoch_logger_1 = EpochLogger()

logger.info("="*60)
logger.info("PHASE 1 TRAINING START")
logger.info("="*60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    callbacks=[early_stop_1, epoch_logger_1]
)

# ======================
# FINE-TUNE PHASE 2
# ======================
base_model.trainable = True

# Freeze 100 layers đầu
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Tạo callback MỚI cho Phase 2 (không dùng lại từ Phase 1)
early_stop_2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

epoch_logger_2 = EpochLogger()

logger.info("="*60)
logger.info("PHASE 2: FINE-TUNING START")
logger.info("="*60)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    callbacks=[early_stop_2, epoch_logger_2]
)


# ===== Evaluation =====
y_true = np.concatenate([y for x, y in val_ds], axis=0)

y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# ===== Save model cuối cùng =====
model.save("../output-model/pepper_disease_model.keras")

# ======================
# PHASE 3: SEPARATE TEST EVALUATION
# ======================
print("\n" + "="*50)
print("PHASE 3: TEST EVALUATION (COMPLETELY SEPARATE)")
print("="*50)

# Load test dataset independently (hoàn toàn tách biệt)
test_ds = tf.keras.utils.image_dataset_from_directory(
    "../dataset_cropped_new/test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False  # không shuffle để evaluation đúng
)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Evaluation trên test set
print("\nEvaluating on TEST set...")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed classification report for test set
y_test_true = np.concatenate([y for x, y in test_ds], axis=0)
y_test_pred_probs = model.predict(test_ds)
y_test_pred = np.argmax(y_test_pred_probs, axis=1)

print("\n" + "-"*50)
print("Test Classification Report:")
print("-"*50)
print(classification_report(y_test_true, y_test_pred, target_names=class_names))

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test_true, y_test_pred))

# Calculate per-class accuracy on test set
from sklearn.metrics import accuracy_score
test_accuracy_per_class = {}
for i, class_name in enumerate(class_names):
    mask = y_test_true == i
    if mask.sum() > 0:
        class_accuracy = accuracy_score(y_test_true[mask], y_test_pred[mask])
        test_accuracy_per_class[class_name] = class_accuracy
        print(f"{class_name}: {class_accuracy:.4f}")

print("\nAverage Test Accuracy: {:.4f}".format(np.mean(list(test_accuracy_per_class.values()))))
