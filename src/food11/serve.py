import io
import os

import mlflow
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Tracking URI comes from an env var so a container can override it later
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = "models:/food11@champion"

# Same order as ImageFolder (alphabetical), verified against the validation folders
CLASSES = [
    "Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
    "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit",
]

# Same normalization as train.py
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

mlflow.set_tracking_uri(TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)  # loaded once, at startup

app = FastAPI(title="food11 API")


def preprocess(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[None, ...].astype(np.float32)  # (1, 3, 224, 224)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        x = preprocess(await file.read())
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image")
    logits = np.asarray(model.predict(x))[0]
    probs = np.exp(logits - logits.max())  # softmax
    probs /= probs.sum()
    i = int(probs.argmax())
    return {"category": CLASSES[i], "confidence": float(probs[i])}