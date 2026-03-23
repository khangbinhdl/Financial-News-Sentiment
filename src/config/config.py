"""
Configuration settings for the Financial News Sentiment Analysis application.
"""
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Paths
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
TOKEN_PADDING_SIZE = 32

# Model configurations
MODEL_CONFIGS = {
    "rnn": {
        "name": "RNN",
        "model_path": SAVED_MODELS_DIR / "rnn_model" / "model_state_dict.pt",
        "tokenizer_path": SAVED_MODELS_DIR / "rnn_model",
        "type": "custom",
        "embedding_dim": 64,
        "hidden_size": 128,
        "n_layers": 2,
        "dropout_prob": 0.2,
        "fc_dim": 32,
    },
    "lstm": {
        "name": "LSTM",
        "model_path": SAVED_MODELS_DIR / "lstm_model" / "model_state_dict.pt",
        "tokenizer_path": SAVED_MODELS_DIR / "lstm_model",
        "type": "custom",
        "embedding_dim": 64,
        "hidden_size": 128,
        "n_layers": 2,
        "dropout_prob": 0.2,
        "fc_dim": 32,
    },
    "gru": {
        "name": "GRU",
        "model_path": SAVED_MODELS_DIR / "gru_model" / "model_state_dict.pt",
        "tokenizer_path": SAVED_MODELS_DIR / "gru_model",
        "type": "custom",
        "embedding_dim": 64,
        "hidden_size": 128,
        "n_layers": 2,
        "dropout_prob": 0.2,
        "fc_dim": 32,
    },
    "distilbert": {
        "name": "DistilBERT",
        "model_path": SAVED_MODELS_DIR / "distilbert_model",
        "tokenizer_path": SAVED_MODELS_DIR / "distilbert_model",
        "type": "hf_bert",
    },
}

# Sentiment labels
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

# API configuration
API_HOST = "127.0.0.1"
API_PORT = 8000
API_RELOAD = True

# Streamlit configuration
STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# Device
DEVICE = "cpu"  # or "cuda" if GPU available
