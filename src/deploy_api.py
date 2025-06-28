from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Define input schema
class Payload(BaseModel):
    features: list[float]

# Load model
with open("models/readmit_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(payload: Payload):
    """
    Receives a list of features and returns the predicted probability of readmission risk.
    """
    try:
        prob = model.predict_proba([payload.features])[0][1]
        return {"readmission_risk": round(prob, 4)}
    except Exception as e:
        return {"error": str(e)}
