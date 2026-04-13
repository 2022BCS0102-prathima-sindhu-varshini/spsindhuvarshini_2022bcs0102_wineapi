from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Input schema
class WineFeatures(BaseModel):
    features: list  # list of feature values

@app.get("/")
def home():
    return {"message": "Wine Quality Prediction API"}

@app.post("/predict")
def predict(data: WineFeatures):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    
    return {
        "predicted_quality": float(prediction[0])
    }