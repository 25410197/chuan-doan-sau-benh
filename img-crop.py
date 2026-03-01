# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np

input_root = "dataset/train"
output_root = "dataset_cropped/train"

os.makedirs(output_root, exist_ok=True)

for class_name in os.listdir(input_root):
    input_class_path = os.path.join(input_root, class_name)
    output_class_path = os.path.join(output_root, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(input_class_path):
        img_path = os.path.join(input_class_path, img_name)
        # Fix: Use imread with proper encoding for Unicode filenames
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Green mask
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cropped = img[y:y+h, x:x+w]
        else:
            cropped = img  # fallback

        # Fix: Use imwrite with proper encoding for Unicode filenames
        output_path = os.path.join(output_class_path, img_name)
        cv2.imencode('.jpg', cropped)[1].tofile(output_path)