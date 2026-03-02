import cv2
import os
import numpy as np

input_root = "dataset/val"
output_root = "dataset_cropped/val"

os.makedirs(output_root, exist_ok=True)

for class_name in os.listdir(input_root):
    input_class_path = os.path.join(input_root, class_name)
    output_class_path = os.path.join(output_root, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(input_class_path):
        img_path = os.path.join(input_class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h_img, w_img = img.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Green mask (có thể chỉnh lại nếu cần)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 🔥 Remove noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # 🔥 Add padding 10%
            pad = int(0.1 * max(w, h))

            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(w_img - x, w + 2*pad)
            h = min(h_img - y, h + 2*pad)

            cropped = img[y:y+h, x:x+w]
        else:
            cropped = img  # fallback

        cv2.imwrite(os.path.join(output_class_path, img_name), cropped)