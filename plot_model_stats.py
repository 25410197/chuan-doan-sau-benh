import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Load model - thử load từ .keras format (mới hơn)
try:
    model = tf.keras.models.load_model("pepper_disease_model.keras")
except Exception as e:
    print(f"Không thể load .keras: {e}")
    print("Thử load .h5...")
    model = tf.keras.models.load_model("pepper_disease_model.h5")

# Hiển thị thông tin model
print("=" * 60)
print("MODEL INFORMATION - THÔNG TIN MÔ HÌNH")
print("=" * 60)

# Thông số cơ bản
print(f"\nModel Name: Pepper Disease Classification")
print(f"Total Parameters: {model.count_params():,}")
print(f"Model Layers: {len(model.layers)}")

# In chi tiết từng layer
print("\n" + "-" * 60)
print("LAYER DETAILS:")
print("-" * 60)
model.summary()

# Lấy thông tin về input và output
print("\n" + "-" * 60)
print("INPUT/OUTPUT INFO:")
print("-" * 60)
print(f"Input Shape: {model.input_shape}")
print(f"Output Shape: {model.output_shape}")

# Hiển thị biểu đồ số parameters theo layer
print("\n" + "-" * 60)
print("PARAMETERS PER LAYER:")
print("-" * 60)

layer_names = []
layer_params = []

for layer in model.layers:
    if layer.count_params() > 0:
        layer_names.append(layer.name[:20])  # Cắt tên nếu quá dài
        layer_params.append(layer.count_params())

# Vẽ biểu đồ
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Biểu đồ 1: Parameters per layer
ax1 = axes[0, 0]
ax1.barh(layer_names, layer_params, color='skyblue')
ax1.set_xlabel('Number of Parameters')
ax1.set_title('Parameters per Layer')
ax1.grid(axis='x', alpha=0.3)

# Biểu đồ 2: Tổng quan Model Info (text)
ax2 = axes[0, 1]
ax2.axis('off')
info_text = f"""
MODEL SUMMARY

Name: Pepper Disease Model
Format: H5 Keras Model

Total Parameters: {model.count_params():,}
Total Layers: {len(model.layers)}

Input Shape: {model.input_shape}
Output Shape: {model.output_shape}

Classes: 5
- Anthracnose
- Healthy
- Leaf Curl
- Leaf Spot
- Nutrient Deficiency
"""
ax2.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.5))

# Biểu đồ 3: Trainable vs Non-trainable
ax3 = axes[1, 0]
trainable = sum([tf.size(w).numpy() for layer in model.layers 
                 for w in layer.trainable_weights])
non_trainable = sum([tf.size(w).numpy() for layer in model.layers 
                     for w in layer.non_trainable_weights])
ax3.pie([trainable, non_trainable], labels=['Trainable', 'Non-trainable'],
        autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
ax3.set_title('Trainable vs Non-trainable Parameters')

# Biểu đồ 4: Model Architecture Type
ax4 = axes[1, 1]
ax4.axis('off')
architecture_text = f"""
LAYER TYPES

Total Layers: {len(model.layers)}

Layer Breakdown:
"""

layer_types = {}
for layer in model.layers:
    layer_type = type(layer).__name__
    layer_types[layer_type] = layer_types.get(layer_type, 0) + 1

for layer_type, count in sorted(layer_types.items()):
    architecture_text += f"\n{layer_type}: {count}"

ax4.text(0.1, 0.5, architecture_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('model_statistics.png', dpi=150, bbox_inches='tight')
print("\n✓ Hình ảnh đã lưu: model_statistics.png")
plt.show()

# Plot history (nếu có)
print("\nTo plot training history, uncomment the code below:")
print("# plt.plot(history.history['accuracy'])")
print("# plt.plot(history.history['val_accuracy'])")