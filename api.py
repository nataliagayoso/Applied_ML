from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np
import torch

from cnn_model import load_cnn_model
from predict_utils import preprocess_image

#  Load models 
rf_model_path = Path(__file__).resolve().parent / 'models' / 'rf_pipeline.joblib'
if not rf_model_path.exists():
    raise RuntimeError(f"Random Forest model not found at {rf_model_path}")
rf_pipeline = joblib.load(rf_model_path)

cnn_model, DEVICE = load_cnn_model()

#  FastAPI app setup 
app = FastAPI(
    title="Cat vs Dog Classifier (Random Forest + CNN)",
    version="2.0",
    description="POST an image to /predict_rf or /predict_cnn to get classification results."
)

class Prediction(BaseModel):
    label: str
    probability: float

@app.get("/")
def health_check():
    return {"status": "API running"}

@app.post("/predict_rf", response_model=Prediction)
async def predict_rf(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files accepted.")
    data = await file.read()
    try:
        feats = preprocess_image(data, for_cnn=False)
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    probs = rf_pipeline.predict_proba(feats)[0]
    idx = int(np.argmax(probs))
    label = ["Cat", "Dog"][idx]
    return Prediction(label=label, probability=float(probs[idx]))

@app.post("/predict_cnn", response_model=Prediction)
async def predict_cnn(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files accepted.")
    data = await file.read()
    try:
        tensor = preprocess_image(data, for_cnn=True).to(DEVICE)
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    with torch.no_grad():
        logits = cnn_model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(np.argmax(probs))
    label = ["Cat", "Dog"][idx]
    return Prediction(label=label, probability=float(probs[idx]))

@app.exception_handler(HTTPException)
def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
