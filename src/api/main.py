"""
FastAPI backend for sentiment analysis inference.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import torch

from src.models.model_loader import ModelLoader, predict
from src.config.config import MODEL_CONFIGS, SENTIMENT_LABELS

# Initialize FastAPI app
app = FastAPI(
    title="Financial News Sentiment Analysis API",
    description="API for sentiment analysis of financial news",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model loader
model_loader = ModelLoader()


# Request/Response models
class PredictionRequest(BaseModel):
    """Request model for prediction."""

    text: str
    model_name: str = "minilm"
    device: str = "cpu"


class SentimentResult(BaseModel):
    """Response model for prediction result."""

    sentiment: str
    probabilities: Dict[str, float]
    model_used: str


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    key: str


@app.get("/", tags=["Health"])
def read_root():
    """Health check endpoint."""
    return {"message": "Financial News Sentiment Analysis API is running"}


@app.get("/models", tags=["Models"])
def get_available_models() -> List[ModelInfo]:
    """Get list of available models."""
    models = []
    for key, config in MODEL_CONFIGS.items():
        models.append(ModelInfo(name=config["name"], key=key))
    return models


@app.post("/predict", tags=["Prediction"], response_model=SentimentResult)
def predict_sentiment(request: PredictionRequest) -> SentimentResult:
    """
    Predict sentiment for input text.

    Args:
        request: PredictionRequest containing text and model_name

    Returns:
        SentimentResult with sentiment prediction and probabilities
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if request.model_name not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_name}' not found. Available models: {list(MODEL_CONFIGS.keys())}",
        )

    if request.device not in {"cpu", "cuda", "mps"}:
        raise HTTPException(
            status_code=400,
            detail="Device must be one of: cpu, cuda, mps",
        )

    try:
        sentiment, probabilities = predict(
            request.text, request.model_name, request.device, model_loader
        )
        return SentimentResult(
            sentiment=sentiment, probabilities=probabilities, model_used=request.model_name
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health", tags=["Health"])
def health_check():
    """Detailed health check."""
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.append("cuda")
    if torch.backends.mps.is_available():
        available_devices.append("mps")

    return {
        "status": "healthy",
        "available_models": list(MODEL_CONFIGS.keys()),
        "sentiment_labels": SENTIMENT_LABELS,
        "available_devices": available_devices,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
