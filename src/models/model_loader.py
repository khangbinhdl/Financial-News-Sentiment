"""
Model loading and inference utilities.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from typing import Dict, Tuple
import numpy as np

from src.config.config import (
    MODEL_CONFIGS,
    SENTIMENT_LABELS,
    TOKEN_PADDING_SIZE,
    DEVICE,
)


class SentimentClassifierRNN(nn.Module):
    """RNN-based sentiment classifier."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        n_layers,
        n_classes,
        dropout_prob,
        fc_dim,
        pad_token_id=0,
    ):
        super(SentimentClassifierRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, fc_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, n_classes)

    def forward(self, input_ids=None, labels=None):
        x = self.embedding(input_ids)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


class SentimentClassifierLSTM(nn.Module):
    """LSTM-based sentiment classifier."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        n_layers,
        n_classes,
        dropout_prob,
        fc_dim,
        pad_token_id=0,
    ):
        super(SentimentClassifierLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, fc_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, n_classes)

    def forward(self, input_ids=None, labels=None):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


class SentimentClassifierGRU(nn.Module):
    """GRU-based sentiment classifier."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        n_layers,
        n_classes,
        dropout_prob,
        fc_dim,
        pad_token_id=0,
    ):
        super(SentimentClassifierGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.gru = nn.GRU(embedding_dim, hidden_size, n_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_size, fc_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_dim, n_classes)

    def forward(self, input_ids=None, labels=None):
        x = self.embedding(input_ids)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


class ModelLoader:
    """Loads and manages sentiment classification models."""

    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def load_model(self, model_name: str):
        """Load a specific model."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_name} not found in configurations")

        if model_name in self.models:
            return self.models[model_name], self.tokenizers[model_name]

        config = MODEL_CONFIGS[model_name]

        if config["type"] == "custom":
            model = self._load_custom_model(model_name)
            tokenizer = self._load_custom_tokenizer(model_name)
        elif config["type"] == "hf_bert":
            model = self._load_hf_model(model_name)
            tokenizer = self._load_hf_tokenizer(model_name)
        else:
            raise ValueError(f"Unknown model type: {config['type']}")

        self.models[model_name] = model
        self.tokenizers[model_name] = tokenizer

        return model, tokenizer

    def _load_custom_model(self, model_name: str):
        """Load custom PyTorch models (RNN, LSTM, GRU)."""
        config = MODEL_CONFIGS[model_name]

        # Get vocabulary size from tokenizer
        tokenizer_path = Path(config["tokenizer_path"])
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
        vocab_size = tokenizer.vocab_size
        pad_token_id = tokenizer.pad_token_id

        # Create model architecture
        if model_name == "rnn":
            model = SentimentClassifierRNN(
                vocab_size=vocab_size,
                embedding_dim=config["embedding_dim"],
                hidden_size=config["hidden_size"],
                n_layers=config["n_layers"],
                n_classes=len(SENTIMENT_LABELS),
                dropout_prob=config["dropout_prob"],
                fc_dim=config["fc_dim"],
                pad_token_id=pad_token_id,
            )
        elif model_name == "lstm":
            model = SentimentClassifierLSTM(
                vocab_size=vocab_size,
                embedding_dim=config["embedding_dim"],
                hidden_size=config["hidden_size"],
                n_layers=config["n_layers"],
                n_classes=len(SENTIMENT_LABELS),
                dropout_prob=config["dropout_prob"],
                fc_dim=config["fc_dim"],
                pad_token_id=pad_token_id,
            )
        elif model_name == "gru":
            model = SentimentClassifierGRU(
                vocab_size=vocab_size,
                embedding_dim=config["embedding_dim"],
                hidden_size=config["hidden_size"],
                n_layers=config["n_layers"],
                n_classes=len(SENTIMENT_LABELS),
                dropout_prob=config["dropout_prob"],
                fc_dim=config["fc_dim"],
                pad_token_id=pad_token_id,
            )
        else:
            raise ValueError(f"Unknown custom model: {model_name}")

        # Load weights
        model_path = Path(config["model_path"])
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        return model

    def _load_hf_model(self, model_name: str):
        """Load Hugging Face models."""
        config = MODEL_CONFIGS[model_name]
        model_path = Path(config["model_path"])
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), local_files_only=True
        )
        model.to(DEVICE)
        model.eval()
        return model

    def _load_custom_tokenizer(self, model_name: str):
        """Load custom tokenizers."""
        config = MODEL_CONFIGS[model_name]
        tokenizer_path = Path(config["tokenizer_path"])
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
        return tokenizer

    def _load_hf_tokenizer(self, model_name: str):
        """Load Hugging Face tokenizers."""
        config = MODEL_CONFIGS[model_name]
        tokenizer_path = Path(config["tokenizer_path"])
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        return tokenizer


def predict(
    text: str, model_name: str, model_loader: ModelLoader
) -> Tuple[str, Dict[str, float]]:
    """
    Predict sentiment for given text.

    Args:
        text: Input text
        model_name: Name of the model to use
        model_loader: ModelLoader instance

    Returns:
        Tuple of (predicted_sentiment, probabilities_dict)
    """
    model, tokenizer = model_loader.load_model(model_name)
    config = MODEL_CONFIGS[model_name]

    with torch.no_grad():
        if config["type"] == "custom":
            # Custom models (RNN, LSTM, GRU)
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=TOKEN_PADDING_SIZE,
                add_special_tokens=False,
                return_attention_mask=False,
            )
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(DEVICE)
            output = model(input_ids=input_ids)
            logits = output["logits"]
        else:
            # Hugging Face models
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(DEVICE)
            attention_mask = encoded["attention_mask"].to(DEVICE)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits

        # Get probabilities
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        predicted_class = np.argmax(probabilities)
        predicted_sentiment = SENTIMENT_LABELS[int(predicted_class)]

        # Create probability dictionary
        prob_dict = {
            SENTIMENT_LABELS[i]: float(probabilities[i]) for i in range(len(SENTIMENT_LABELS))
        }

        return predicted_sentiment, prob_dict
