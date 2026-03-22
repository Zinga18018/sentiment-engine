from dataclasses import dataclass


@dataclass
class SentimentConfig:
    """all settings for the sentiment classifier."""

    model_id: str = "distilbert-base-uncased-finetuned-sst-2-english"
    params: str = "67M"
    max_length: int = 512
    max_batch_size: int = 100
    port: int = 8003
