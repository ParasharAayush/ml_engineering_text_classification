from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model.predict import ModelPredictor

# Define a Pydantic model for the request body
class PredictRequest(BaseModel):
    text: str

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "OK"}

predictor = ModelPredictor("model/svm_model.pkl")  # Path to the saved model

@router.post("/predict/")
def predict(request: PredictRequest):
    try:
        result = predictor.predict(request.text)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
