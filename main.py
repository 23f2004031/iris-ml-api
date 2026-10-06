from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os

app = FastAPI(title="Iris Classification API")

# Allow CORS for external calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = joblib.load("model.pkl")
    model_loaded = True
except Exception:
    model = None
    model_loaded = False

TARGET_NAMES = ["setosa", "versicolor", "virginica"]

class PredictRequest(BaseModel):
    features: List[float]

@app.get("/")
def serve_ui():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "Iris API is running. Visit /docs for the API documentation."}

@app.get("/health")
def health():
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

@app.post("/predict")
def predict(request: PredictRequest):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    if len(request.features) != 4:
        raise HTTPException(status_code=400, detail="Features array must contain exactly 4 values")

    data = np.array([request.features])
    pred = int(model.predict(data)[0])
    
    return {
        "prediction": pred,
        "class_name": TARGET_NAMES[pred]
    }