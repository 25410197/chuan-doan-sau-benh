import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io

IMG_SIZE = 224

app = FastAPI()

# Load model một lần khi start server
model = tf.keras.models.load_model("pepper_disease_model.keras")

# Class order must match training dataset (alphabetical order from folder names)
class_names_vi = [
    "đốm lá",           # Target spot
    "khảm lá",          # Anthracnose
    "lá khỏe",          # Healthy leaves
    "thán thư",         # Bacterial spot
    "vàng lá"           # Leaf yellowing
]

# Mapping to English for display
class_names_en = {
    "đốm lá": "Target Spot",
    "khảm lá": "Anthracnose",
    "lá khỏe": "Healthy Leaves",
    "thán thư": "Bacterial Spot",
    "vàng lá": "Leaf Yellowing"
}

def prepare_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32)  # giữ nguyên [0, 255]
    image = np.expand_dims(image, axis=0)
    return image  # model đã có Rescaling layer bên trong, KHÔNG preprocess ở đây


@app.get("/")
def health():
    return {"status": "ok", "model": "pepper_disease_model"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Kiểm tra file có phải ảnh không
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "File phải là ảnh (jpg, png, ...)"}
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Không thể đọc file ảnh"}
        )

    processed_image = prepare_image(image)

    prediction = model.predict(processed_image)
    predicted_idx = np.argmax(prediction)
    predicted_class_vi = class_names_vi[predicted_idx]
    predicted_class_en = class_names_en[predicted_class_vi]
    confidence = float(np.max(prediction))

    return JSONResponse({
        "class_vietnamese": predicted_class_vi,
        "class_english": predicted_class_en,
        "confidence": round(confidence, 4)
    })
