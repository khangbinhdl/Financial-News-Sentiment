"""
Model loading and inference utilities.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from src.config.config import DEVICE, MODEL_CONFIGS, SENTIMENT_LABELS


class RNNSequenceClassifierConfig(PretrainedConfig):
    model_type = "rnn-sequence-classifier"

    def __init__(
        self,
        vocab_size=5000,
        embedding_dim=128,
        hidden_dim=256,
        num_labels=3,
        num_layers=2,
        dropout=0.3,
        fc_dim=128,
        pad_token_id=0,
        id2label=None,
        label2id=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_layers = num_layers
        self.dropout = dropout
        self.fc_dim = fc_dim


class RNNForSequenceClassification(PreTrainedModel):
    config_class = RNNSequenceClassifierConfig

    def __init__(self, config):
        super().__init__(config)

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id,
        )

        self.rnn = nn.RNN(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.fc_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_dim, config.num_labels),
        )

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        outputs, _ = self.rnn(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = outputs.mean(dim=1)

        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class LSTMSequenceClassifierConfig(PretrainedConfig):
    model_type = "lstm-sequence-classifier"

    def __init__(
        self,
        vocab_size=5000,
        embedding_dim=128,
        hidden_dim=256,
        num_labels=3,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        fc_dim=128,
        pad_token_id=0,
        id2label=None,
        label2id=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.fc_dim = fc_dim


class LSTMForSequenceClassification(PreTrainedModel):
    config_class = LSTMSequenceClassifierConfig

    def __init__(self, config):
        super().__init__(config)

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id,
        )

        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=config.bidirectional,
        )

        hidden_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, config.fc_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_dim, config.num_labels),
        )

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        outputs, _ = self.lstm(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = outputs.mean(dim=1)

        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class GRUSequenceClassifierConfig(PretrainedConfig):
    model_type = "gru-sequence-classifier"

    def __init__(
        self,
        vocab_size=5000,
        embedding_dim=128,
        hidden_dim=256,
        num_labels=3,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        fc_dim=128,
        pad_token_id=0,
        id2label=None,
        label2id=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            id2label=id2label,
            label2id=label2id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.fc_dim = fc_dim


class GRUForSequenceClassification(PreTrainedModel):
    config_class = GRUSequenceClassifierConfig

    def __init__(self, config):
        super().__init__(config)

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id,
        )

        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=config.bidirectional,
        )

        hidden_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, config.fc_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_dim, config.num_labels),
        )

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.dropout(x)
        outputs, _ = self.gru(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (outputs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = outputs.mean(dim=1)

        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


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
        id2label = {str(k): v for k, v in SENTIMENT_LABELS.items()}
        label2id = {v: int(k) for k, v in SENTIMENT_LABELS.items()}

        if model_name == "rnn":
            model_config = RNNSequenceClassifierConfig(
                vocab_size=vocab_size,
                embedding_dim=config["embedding_dim"],
                hidden_dim=config["hidden_size"],
                num_labels=len(SENTIMENT_LABELS),
                num_layers=config["n_layers"],
                dropout=config["dropout_prob"],
                fc_dim=config["fc_dim"],
                pad_token_id=pad_token_id,
                id2label=id2label,
                label2id=label2id,
            )
            model = RNNForSequenceClassification(model_config)
        elif model_name == "lstm":
            model_config = LSTMSequenceClassifierConfig(
                vocab_size=vocab_size,
                embedding_dim=config["embedding_dim"],
                hidden_dim=config["hidden_size"],
                num_labels=len(SENTIMENT_LABELS),
                num_layers=config["n_layers"],
                dropout=config["dropout_prob"],
                bidirectional=config["bidirectional"],
                fc_dim=config["fc_dim"],
                pad_token_id=pad_token_id,
                id2label=id2label,
                label2id=label2id,
            )
            model = LSTMForSequenceClassification(model_config)
        elif model_name == "gru":
            model_config = GRUSequenceClassifierConfig(
                vocab_size=vocab_size,
                embedding_dim=config["embedding_dim"],
                hidden_dim=config["hidden_size"],
                num_labels=len(SENTIMENT_LABELS),
                num_layers=config["n_layers"],
                dropout=config["dropout_prob"],
                bidirectional=config["bidirectional"],
                fc_dim=config["fc_dim"],
                pad_token_id=pad_token_id,
                id2label=id2label,
                label2id=label2id,
            )
            model = GRUForSequenceClassification(model_config)
        else:
            raise ValueError(f"Unknown custom model: {model_name}")

        # Load weights
        model_path = Path(config["model_path"])
        if model_path.is_file():
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            model_dir = model_path if model_path.is_dir() else model_path.parent
            if model_name == "rnn":
                model = RNNForSequenceClassification.from_pretrained(
                    str(model_dir), local_files_only=True
                )
            elif model_name == "lstm":
                model = LSTMForSequenceClassification.from_pretrained(
                    str(model_dir), local_files_only=True
                )
            elif model_name == "gru":
                model = GRUForSequenceClassification.from_pretrained(
                    str(model_dir), local_files_only=True
                )

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
    text: str, model_name: str, device_name: str, model_loader: ModelLoader
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
    max_length = _resolve_max_seq_len(model, tokenizer)
    if max_length is None:
        raise ValueError("max_seq_len is missing from saved model config")
    device = _resolve_device(device_name)
    model.to(device)

    with torch.no_grad():
        if config["type"] == "custom":
            # Custom models (RNN, LSTM, GRU)
            encoded = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                add_special_tokens=False,
                return_attention_mask=True,
            )
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long).to(device)
            attention_mask = torch.tensor(
                [encoded["attention_mask"]], dtype=torch.long
            ).to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits
        else:
            # Hugging Face models
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits

        # Get probabilities
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        predicted_class = int(np.argmax(probabilities))

        id2label = getattr(model.config, "id2label", None) or SENTIMENT_LABELS
        label_list = [label for _, label in sorted(id2label.items(), key=lambda x: int(x[0]))]

        predicted_sentiment = label_list[predicted_class]

        # Create probability dictionary
        prob_dict = {
            label_list[i]: float(probabilities[i]) for i in range(len(label_list))
        }

        return predicted_sentiment, prob_dict


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is not available on this machine")
        return torch.device("mps")
    raise ValueError(f"Unsupported device: {device_name}")


def _resolve_max_seq_len(model, tokenizer) -> int | None:
    max_seq_len = getattr(model.config, "max_seq_len", None)
    if max_seq_len is None:
        max_seq_len = getattr(model.config, "max_position_embeddings", None)
    if max_seq_len is None:
        max_seq_len = getattr(tokenizer, "model_max_length", None)
    if max_seq_len is None:
        return None
    max_seq_len = int(max_seq_len)
    if max_seq_len > 1_000_000:
        return None
    return max_seq_len
