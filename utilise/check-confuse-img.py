import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

IMG_SIZE = 224
BATCH_SIZE = 32

# 1️⃣ Load model đã train
model = load_model("pepper_disease_model_new.keras")

# 2️⃣ Load lại validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset_resized/val",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = val_ds.class_names

# 3️⃣ Hiển thị ảnh dự đoán sai
max_errors = 20
error_count = 0

for images, labels in val_ds:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    for i in range(len(labels)):
        if preds[i] != labels[i]:
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(
                f"True: {class_names[labels[i]]} | Pred: {class_names[preds[i]]}"
            )
            plt.axis("off")
            plt.show()

            error_count += 1
            if error_count >= max_errors:
                break
    if error_count >= max_errors:
        break