import os
import shutil
import random
from sklearn.model_selection import train_test_split

input_folder = "dataset_cropped"
output_folder = "dataset_cropped_new"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for class_name in os.listdir(input_folder):

    class_path = os.path.join(input_folder, class_name)
    images = os.listdir(class_path)

    train_imgs, temp_imgs = train_test_split(
        images, test_size=(1 - train_ratio), random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=test_ratio/(test_ratio+val_ratio), random_state=42
    )

    for split, split_imgs in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        split_dir = os.path.join(output_folder, split, class_name)
        os.makedirs(split_dir, exist_ok=True)

        for img in split_imgs:
            shutil.copy(
                os.path.join(class_path, img),
                os.path.join(split_dir, img)
            )