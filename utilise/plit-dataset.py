import os
import shutil
import random
from sklearn.model_selection import train_test_split

input_folder = "dataset_smartcrop"
output_folder = "dataset_smartcrop_new"

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

for class_name in os.listdir(input_folder):

    class_path = os.path.join(input_folder, class_name)
    
    # Skip if not a directory
    if not os.path.isdir(class_path):
        print(f"Skipping {class_name} (not a directory)")
        continue
    
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    
    if not images:
        print(f"No images found in {class_name}")
        continue

    print(f"Processing class: {class_name} ({len(images)} images)")
    
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
            try:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Error copying {img}: {e}")
    
    print(f"  - Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")