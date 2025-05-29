from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import numpy as np
import io, joblib

# Load pipeline
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'rf_pipeline.joblib'
if not MODEL_PATH.exists():
    raise RuntimeError(f"Model not found at {MODEL_PATH}")
pipeline = joblib.load(MODEL_PATH)

# FastAPI setup
app = FastAPI(
    title="Cat vs Dog Classifier",
    version="1.0",
    description="POST an image to /predict to get back the label and probability."
)

class Prediction(BaseModel):
    label: str
    probability: float

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files accepted.")
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file.")
    # Resize to 224×224 to match training pipeline 
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    feats = arr.flatten().reshape(1, -1)

    # Predict
    probs = pipeline.predict_proba(feats)[0]
    idx   = int(np.argmax(probs))
    label = ["Cat", "Dog"][idx]
    return Prediction(label=label, probability=float(probs[idx]))

@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

