from PIL import Image
import os

input_folder = "dataset"
output_folder = "dataset_resized"

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(root, file)
            img = Image.open(path)
            img = img.resize((512, 512))
            
            new_path = path.replace(input_folder, output_folder)
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            img.save(new_path)