"""
Check for corrupted or oversized images that cause PIL DecompressionBombWarning
"""
import os
from PIL import Image
import warnings

# Suppress the warning temporarily to count issues
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

def check_dataset(dataset_path):
    """Scan dataset for problematic images"""
    issues = []
    total = 0
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                total += 1
                filepath = os.path.join(root, file)
                
                try:
                    # Try to open image
                    img = Image.open(filepath)
                    width, height = img.size
                    pixels = width * height
                    
                    # PIL limit is 89478485 pixels
                    if pixels > 89478485:
                        issues.append({
                            'file': filepath,
                            'type': 'oversized',
                            'pixels': pixels,
                            'size': f'{width}x{height}'
                        })
                    
                    # Verify image can be loaded
                    img.load()
                    
                except Exception as e:
                    issues.append({
                        'file': filepath,
                        'type': 'corrupted',
                        'error': str(e)
                    })
    
    return total, issues

# Check datasets
print("Checking dataset_new/train...")
total_train, issues_train = check_dataset("../dataset_new/train")

print(f"\nDataset check complete!")
print(f"Total images: {total_train}")
print(f"Problematic images: {len(issues_train)}\n")

if issues_train:
    print("=== ISSUES FOUND ===")
    for issue in issues_train:
        print(f"\nFile: {issue['file']}")
        print(f"Type: {issue['type']}")
        if issue['type'] == 'oversized':
            print(f"Size: {issue['size']} ({issue['pixels']} pixels)")
        else:
            print(f"Error: {issue['error']}")
else:
    print("No problematic images found! ✓")
    print("\nThe warning might be coming from intermediate processing.")
    print("Solution: Enable the PIL decompression bomb check:")
    print("  PIL.Image.MAX_IMAGE_PIXELS = None")
