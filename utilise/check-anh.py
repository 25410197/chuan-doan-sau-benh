import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ==========================
# 1. Config augmentation
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    shear_range=0.15,
    brightness_range=[0.8,1.2],
    channel_shift_range=30.0,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

# ==========================
# 2. Load ảnh gốc
# ==========================
IMG_HEIGHT = 224
IMG_WIDTH = 224

img_path = "dataset_smartcrop_new/val/leaf_spot/IMG_20260214_162556.jpg"  # 👉 đổi thành đường dẫn ảnh của bạn

img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = img_to_array(img)

# thêm batch dimension (1, H, W, C)
img_array = np.expand_dims(img_array, axis=0)

# ==========================
# 3. Generate ảnh augment
# ==========================
aug_iter = train_datagen.flow(img_array, batch_size=1)

# ==========================
# 4. Visualize
# ==========================
plt.figure(figsize=(10,10))

for i in range(10):
    aug_img = next(aug_iter)[0]

    plt.subplot(3,3,i+1)
    plt.imshow(aug_img)
    plt.title(f"Augmented {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.show()