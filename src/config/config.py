"""
Configuration settings for the Financial News Sentiment Analysis application.
"""
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Paths
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# Model configurations
MODEL_CONFIGS = {
    "rnn": {
        "name": "RNN",
        "model_path": SAVED_MODELS_DIR / "rnn_model",
        "tokenizer_path": SAVED_MODELS_DIR / "rnn_model",
        "type": "custom",
        "embedding_dim": 128,
        "hidden_size": 256,
        "n_layers": 2,
        "dropout_prob": 0.3,
        "fc_dim": 128,
    },
    "lstm": {
        "name": "LSTM",
        "model_path": SAVED_MODELS_DIR / "lstm_model",
        "tokenizer_path": SAVED_MODELS_DIR / "lstm_model",
        "type": "custom",
        "embedding_dim": 128,
        "hidden_size": 256,
        "n_layers": 2,
        "dropout_prob": 0.3,
        "bidirectional": True,
        "fc_dim": 128,
    },
    "gru": {
        "name": "GRU",
        "model_path": SAVED_MODELS_DIR / "gru_model",
        "tokenizer_path": SAVED_MODELS_DIR / "gru_model",
        "type": "custom",
        "embedding_dim": 128,
        "hidden_size": 256,
        "n_layers": 2,
        "dropout_prob": 0.3,
        "bidirectional": True,
        "fc_dim": 128,
    },
    "minilm": {
        "name": "MiniLM",
        "model_path": SAVED_MODELS_DIR / "minilm_model",
        "tokenizer_path": SAVED_MODELS_DIR / "minilm_model",
        "type": "hf_bert",
    },
}

# Sentiment labels
SENTIMENT_LABELS = {
    0: "neutral",
    1: "negative",
    2: "positive",
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
