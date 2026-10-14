"""분류기 불러오기와 예측.

학습 담당 스크립트의 저장 형식에 맞춘다.
- BERT: training/evaluate_klue_bert_base.py → save_pretrained (토크나이저·id2label 포함)
- 베이스라인: training/train_tfidf_logistic_regression.py → {"vectorizer", "model"} 딕셔너리를 joblib으로 저장
  (sklearn Pipeline으로 저장한 경우도 처리)
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from app import config


@dataclass
class Prediction:
    label: str
    confidence: float
    latency_s: float


class BertClassifier:
    def __init__(self, model_dir: Path, max_length: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not model_dir.exists():
            raise FileNotFoundError(f"BERT 모델 폴더 없음: {model_dir}")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        # 모델 폴더에 토크나이저가 있으면 사용, 없으면 사전학습 모델의 토크나이저 사용
        tokenizer_source = model_dir if (model_dir / "tokenizer_config.json").exists() else config.BERT_TOKENIZER_FALLBACK
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.max_length = max_length
        self.name = f"BERT ({model_dir.name}, {self.device})"
        self.n_labels = len(self.id2label)

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        with self.torch.inference_mode():
            enc = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[0]
            idx = int(probs.argmax())
            confidence = float(probs[idx])
        return Prediction(self.id2label[idx], round(confidence, 4), round(time.perf_counter() - start, 4))


class BaselineClassifier:
    def __init__(self, model_path: Path):
        import joblib

        if not model_path.exists():
            raise FileNotFoundError(f"베이스라인 모델 파일 없음: {model_path}")
        # 학습 때 직접 만든 토크나이저 함수를 다시 찾을 수 있도록 training 폴더와 프로젝트 폴더를 경로에 추가
        for p in (config.TRAINING_DIR, config.PROJECT_DIR):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        saved = joblib.load(model_path)

        if isinstance(saved, dict):            # {"vectorizer": ..., "model": ...}
            self.vectorizer, self.model = saved["vectorizer"], saved["model"]
        else:                                  # sklearn Pipeline
            self.vectorizer, self.model = None, saved
        self.name = f"TF-IDF + LR ({model_path.name})"
        self.n_labels = len(self.model.classes_)

    def predict(self, text: str) -> Prediction:
        start = time.perf_counter()
        features = self.vectorizer.transform([text]) if self.vectorizer is not None else [text]
        probs = self.model.predict_proba(features)[0]
        idx = int(probs.argmax())
        return Prediction(str(self.model.classes_[idx]), round(float(probs[idx]), 4),
                          round(time.perf_counter() - start, 4))


def load_classifier():
    if config.CLASSIFIER_TYPE == "baseline":
        return BaselineClassifier(config.BASELINE_MODEL_PATH)
    if config.CLASSIFIER_TYPE == "bert":
        return BertClassifier(config.BERT_MODEL_DIR, config.BERT_MAX_LENGTH)
    raise ValueError(f"알 수 없는 CLASSIFIER_TYPE: {config.CLASSIFIER_TYPE}")
