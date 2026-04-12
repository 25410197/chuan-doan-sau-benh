import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

def smart_crop_leaf(image_path, target_size=(224, 224), debug=False):
    """
    Xác định vị trí lá và trái, crop ra target_size
    - Tìm vùng màu xanh/lục (lá) + đỏ/vàng (trái)
    - Lấy vật thể lớn nhất
    - Crop centered
    
    Returns: ndarray ảnh crop, hoặc None nếu không tìm thấy
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mask 1: Màu xanh/lục (LÁ)
    lower_green = np.array([20, 20, 20])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Mask 2: Màu đỏ (TRÁI - phần 1, vì đỏ ở 2 đầu HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Mask 3: Màu vàng/cam (TRÁI)
    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Gộp tất cả masks
    mask = cv2.bitwise_or(mask_green, mask_red1)
    mask = cv2.bitwise_or(mask, mask_red2)
    mask = cv2.bitwise_or(mask, mask_yellow)
    
    # Morphology để clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Tìm contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        if debug:
            print(f"  ⚠️  Không tìm thấy vật thể - dùng toàn ảnh")
        # Fallback: dùng toàn ảnh, crop từ giữa
        return fallback_center_crop(img, target_size)
    
    # Tìm contour lớn nhất (đó là lá hoặc trái chính)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Expand bounding box một chút (margin)
    margin = 20
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)
    
    # Crop từ bounding box
    cropped = img[y:y+h, x:x+w]
    
    if cropped.size == 0:
        return fallback_center_crop(img, target_size)
    
    # Resize tới target_size với aspect ratio
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    cropped_pil = Image.fromarray(cropped_rgb)
    cropped_pil.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Pad tới target_size với màu trắng
    final_img = Image.new('RGB', target_size, (255, 255, 255))
    offset = ((target_size[0] - cropped_pil.size[0]) // 2,
              (target_size[1] - cropped_pil.size[1]) // 2)
    final_img.paste(cropped_pil, offset)
    
    return cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)


def fallback_center_crop(img, target_size=(224, 224)):
    """
    Fallback: crop từ giữa nếu không tìm thấy lá
    """
    h, w = img.shape[:2]
    ratio = max(target_size[0] / w, target_size[1] / h)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    top = (new_h - target_size[1]) // 2
    left = (new_w - target_size[0]) // 2
    cropped = resized[top:top+target_size[1], left:left+target_size[0]]
    
    return cropped


def smart_crop_dataset(src_dir, dest_dir, target_size=(224, 224), quality=90):
    """
    Crop toàn bộ dataset dùng smart detection (lá + trái)
    """
    print(f"Smart Crop Dataset - Target size: {target_size}")
    print(f"Đang xác định lá/trái và crop...\n")
    
    stats = {
        "total": 0, 
        "success": 0, 
        "fallback": 0,  # số lần fallback
        "original_size": 0, 
        "new_size": 0
    }
    
    for root, _, files in os.walk(src_dir):
        for file in tqdm(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, src_dir)
                dest_folder = os.path.join(dest_dir, rel_path)
                os.makedirs(dest_folder, exist_ok=True)
                dest_path = os.path.join(dest_folder, file)
                
                try:
                    original_size = os.path.getsize(src_path)
                    stats["original_size"] += original_size
                    
                    # Try smart crop
                    result = smart_crop_leaf(src_path, target_size, debug=False)
                    
                    if result is None:
                        stats["fallback"] += 1
                        # Fallback lần 2: dùng PIL padding method (màu trắng)
                        img = Image.open(src_path).convert('RGB')
                        img.thumbnail(target_size, Image.Resampling.LANCZOS)
                        new_img = Image.new('RGB', target_size, (255, 255, 255))
                        offset = ((target_size[0] - img.size[0]) // 2,
                                 (target_size[1] - img.size[1]) // 2)
                        new_img.paste(img, offset)
                        new_img.save(dest_path, quality=quality, optimize=True)
                    else:
                        # Save cropped image
                        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                        result_pil.save(dest_path, quality=quality, optimize=True)
                    
                    new_size = os.path.getsize(dest_path)
                    stats["new_size"] += new_size
                    stats["success"] += 1
                    
                except Exception as e:
                    print(f"❌ Lỗi {file}: {e}")
                
                stats["total"] += 1
    
    # Print stats
    reduction = ((stats["original_size"] - stats["new_size"]) / 
                 stats["original_size"] * 100 if stats["original_size"] > 0 else 0)
    
    print(f"\n{'='*70}")
    print(f"SMART CROP DATASET - RESULTS")
    print(f"{'='*70}")
    print(f"✅ Tổng ảnh: {stats['total']} | Thành công: {stats['success']}")
    print(f"⚠️  Fallback (không tìm lá): {stats['fallback']}")
    print(f"📊 Size gốc: {stats['original_size'] / 1024 / 1024:.2f} MB")
    print(f"📊 Size mới: {stats['new_size'] / 1024 / 1024:.2f} MB")
    print(f"🎯 Giảm kích thước: {reduction:.1f}%")
    print(f"{'='*70}\n")
    
    return stats


def test_single_image(image_path, target_size=(224, 224)):
    """
    Test smart crop trên một ảnh để xem kết quả
    """
    print(f"Testing smart crop on: {image_path}")
    
    result = smart_crop_leaf(image_path, target_size, debug=True)
    
    if result is not None:
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        result_pil.save("test_crop_result.jpg", quality=95)
        print("✅ Saved result to: test_crop_result.jpg")
    else:
        print("❌ Crop failed")


if __name__ == "__main__":
    # Test trước trên một ảnh (lá hoặc trái)
    # test_single_image("dataset_new/train/anthracnose/sample.jpg")
    
    # Crop toàn bộ dataset - tự động detect lá và trái
    smart_crop_dataset(
        src_dir="dataset",
        dest_dir="dataset_smartcrop",
        target_size=(224, 224),
        quality=90
    )
