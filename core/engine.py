import time
import logging
from dataclasses import dataclass

import torch
from transformers import pipeline as hf_pipeline

from .config import SentimentConfig

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    text: str
    label: str
    score: float
    inference_ms: float


class SentimentEngine:
    """DistilBERT-based sentiment classifier.

    supports single text analysis, batch processing, analytics
    aggregation, and side-by-side comparison of two texts.
    """

    def __init__(self, config: SentimentConfig | None = None):
        self.config = config or SentimentConfig()
        self._pipeline = None
        self.device_name = None

    def load(self):
        """load the classification pipeline onto the best available device."""
        device_idx = 0 if torch.cuda.is_available() else -1
        self.device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
        logger.info("loading %s on %s", self.config.model_id, self.device_name)

        self._pipeline = hf_pipeline(
            "sentiment-analysis",
            model=self.config.model_id,
            device=device_idx,
            truncation=True,
            max_length=self.config.max_length,
        )
        logger.info("sentiment engine ready")

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    # ---- single analysis ----

    def analyze(self, text: str) -> SentimentResult:
        """classify a single piece of text."""
        start = time.perf_counter()
        raw = self._pipeline(text)[0]
        ms = (time.perf_counter() - start) * 1000

        return SentimentResult(
            text=text[:200],
            label=raw["label"],
            score=round(raw["score"], 4),
            inference_ms=round(ms, 1),
        )

    # ---- batch analysis ----

    def analyze_batch(self, texts: list[str]) -> tuple[list[SentimentResult], float]:
        """classify multiple texts in one pass."""
        start = time.perf_counter()
        raw_results = self._pipeline(texts)
        total_ms = (time.perf_counter() - start) * 1000

        results = []
        per_text_ms = round(total_ms / len(texts), 1)
        for text, raw in zip(texts, raw_results):
            results.append(SentimentResult(
                text=text[:200],
                label=raw["label"],
                score=round(raw["score"], 4),
                inference_ms=per_text_ms,
            ))
        return results, round(total_ms, 1)

    # ---- analytics ----

    def analytics(self, texts: list[str]) -> dict:
        """run batch analysis and return aggregate statistics."""
        results, total_ms = self.analyze_batch(texts)

        pos = [r for r in results if r.label == "POSITIVE"]
        neg = [r for r in results if r.label == "NEGATIVE"]
        avg_conf = sum(r.score for r in results) / len(results)

        most_pos = max(pos, key=lambda r: r.score) if pos else results[0]
        most_neg = max(neg, key=lambda r: r.score) if neg else results[0]

        return {
            "total_texts": len(results),
            "positive_count": len(pos),
            "negative_count": len(neg),
            "positive_ratio": round(len(pos) / len(results), 4),
            "avg_confidence": round(avg_conf, 4),
            "most_positive": {"text": most_pos.text, "score": most_pos.score},
            "most_negative": {"text": most_neg.text, "score": most_neg.score},
            "inference_ms": total_ms,
        }

    # ---- comparison ----

    def compare(self, text_a: str, text_b: str) -> dict:
        """compare sentiment between two texts side by side."""
        start = time.perf_counter()
        raw = self._pipeline([text_a, text_b])
        ms = (time.perf_counter() - start) * 1000

        def signed_score(r):
            return r["score"] if r["label"] == "POSITIVE" else -r["score"]

        sa, sb = signed_score(raw[0]), signed_score(raw[1])
        return {
            "text_a": {"text": text_a[:200], "label": raw[0]["label"], "score": raw[0]["score"]},
            "text_b": {"text": text_b[:200], "label": raw[1]["label"], "score": raw[1]["score"]},
            "more_positive": "text_a" if sa > sb else "text_b",
            "sentiment_gap": round(abs(sa - sb), 4),
            "inference_ms": round(ms, 1),
        }

    # ---- health ----

    def health(self) -> dict:
        return {
            "status": "healthy" if self.is_loaded else "loading",
            "model": self.config.model_id,
            "device": self.device_name,
            "params": self.config.params,
        }
