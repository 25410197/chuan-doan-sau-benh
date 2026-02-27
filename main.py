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

class_names = [
    "anthracnose",
    "healthy",
    "leaf_curl",
    "leaf_spot",
    "nutrient_deficiency"
]

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
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return JSONResponse({
        "class": predicted_class,
        "confidence": round(confidence, 4)
    })
