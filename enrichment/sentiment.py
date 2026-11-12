"""Lightweight, pinned, lazy multilingual review sentiment classification.

No checkpoint loads at import time. Bounded batches and tokens; one model per worker.
Default CPU preserves GPU VRAM for Qwen embeddings, reranking and generation.
Models must be evaluated on real customer data; softmax scores aren't calibrated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Protocol

MODEL_ID = 'Horizon-Labs/multilingual-sentiment-small'  # Apache-2.0; verify model terms on deployment
SHA = re.compile(r'[a-fA-F0-9]{40}\Z')
LABELS = ('negative', 'neutral', 'positive')
CONTRACT_VERSION = 'sm-sentiment-v1'


@dataclass(frozen=True, slots=True)
class SentimentConfig:
    revision: str
    model_id: str = MODEL_ID
    device: str = 'cpu'  # default: keep CUDA memory for the active Qwen models
    offline: bool = True
    max_tokens: int = 256
    batch_size: int = 8
    max_batch: int = 32
    max_chars: int = 20_000

    def __post_init__(self) -> None:
        if not SHA.fullmatch(self.revision):
            raise ValueError('Pin a 40-character Hugging Face model commit SHA')
        if self.device not in {'cpu', 'cuda', 'auto'}:
            raise ValueError('device must be cpu, cuda or auto')
        if not 8 <= self.max_tokens <= 512 or not 1 <= self.batch_size <= 32:
            raise ValueError('Invalid token or batch cap')
        if not self.batch_size <= self.max_batch <= 128 or not 1 <= self.max_chars <= 100_000:
            raise ValueError('Invalid input caps')
        if not self.model_id.strip():
            raise ValueError('model_id is required')

    @property
    def config_hash(self) -> str:
        settings = (CONTRACT_VERSION, self.model_id, self.revision.lower(), self.max_tokens)
        return hashlib.sha256(json.dumps(settings, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class SentimentPrediction:
    label: str
    scores: dict[str, float]
    model_id: str
    model_revision: str
    config_hash: str
    truncated: bool = False


class SentimentModel(Protocol):
    config: SentimentConfig

    def predict_many(self, texts: list[str]) -> list[SentimentPrediction]: ...


class LocalSentimentClassifier:
    """Batching and lazy loading protected by a single inference lock."""

    def __init__(self, config: SentimentConfig):
        self.config = config
        self._lock = threading.RLock()
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.config.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available')
        device = 'cuda' if (self.config.device == 'cuda' or
                            (self.config.device == 'auto' and torch.cuda.is_available())) else 'cpu'
        path = snapshot_download(repo_id=self.config.model_id,
                                 revision=self.config.revision, local_files_only=self.config.offline)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        # Keep classifier in FP32: robust on CPU and small compared with Qwen.
        model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, trust_remote_code=False)
        model.to(device)
        model.eval()
        expected = {str(label).lower() for label in model.config.id2label.values()}
        if expected != set(LABELS) or model.config.num_labels != 3:
            raise RuntimeError(f'Unexpected model labels: {model.config.id2label!r}')
        self._tokenizer, self._model, self._device = tokenizer, model, device

    def warmup(self) -> None:
        self.predict_many(['The battery is terrible.'])

    def predict_one(self, text: str) -> SentimentPrediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[SentimentPrediction]:
        if not texts:
            return []
        if len(texts) > self.config.max_batch:
            raise ValueError('Split inputs into bounded pages before calling predict_many')
        if any(not isinstance(t, str) or not t.strip() or len(t) > self.config.max_chars for t in texts):
            raise ValueError('Nonempty text required; review exceeds configured max_chars')
        import torch
        with self._lock:
            self._load()
            results: list[SentimentPrediction] = []
            for start in range(0, len(texts), self.config.batch_size):
                batch = texts[start:start + self.config.batch_size]
                lengths = self._tokenizer(batch, padding=False, truncation=False,
                                          return_length=True)['length']
                inputs = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.config.max_tokens, return_tensors='pt')
                inputs = {key: value.to(self._device) for key, value in inputs.items()}
                with torch.inference_mode():
                    probabilities = torch.softmax(self._model(**inputs).logits.float(), dim=-1).cpu().tolist()
                for sample_index, values in enumerate(probabilities):
                    scores = {str(self._model.config.id2label[i]).lower(): float(p)
                              for i, p in enumerate(values)}
                    label = max(LABELS, key=lambda name: scores[name])
                    results.append(SentimentPrediction(
                        label, scores, self.config.model_id, self.config.revision,
                        self.config.config_hash,
                        truncated=lengths[sample_index] > self.config.max_tokens))
            return results

    def close(self) -> None:
        with self._lock:
            self._tokenizer = self._model = self._device = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prefetch', 'test'))
    parser.add_argument('--revision', required=True)
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--device', choices=('cpu','cuda','auto'), default='cpu')
    parser.add_argument('--text', default='The camera is great but the battery is bad.')
    args = parser.parse_args()
    config = SentimentConfig(revision=args.revision, model_id=args.model, device=args.device,
                             offline=args.command != 'prefetch')
    if args.command == 'prefetch':
        from huggingface_hub import snapshot_download
        print(snapshot_download(repo_id=config.model_id, revision=config.revision,
                                local_files_only=False))
    else:
        classifier = LocalSentimentClassifier(config)
        from dataclasses import asdict
        try:
            print(json.dumps(asdict(classifier.predict_one(args.text)), indent=2))
        finally:
            classifier.close()


if __name__ == '__main__':
    main()
