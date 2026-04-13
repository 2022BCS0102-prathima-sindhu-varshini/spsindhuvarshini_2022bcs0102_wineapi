from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class WineFeatures(BaseModel):
    features: list  

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
