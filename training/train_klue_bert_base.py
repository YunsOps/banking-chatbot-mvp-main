import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, set_seed, DataCollatorWithPadding,
    TrainingArguments, Trainer
)


# 데이터 로드
data_dir = Path(__file__).resolve().parent.parent / "data" / "dataset_v2"
train_df = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
validation_df = pd.read_csv(data_dir / "validation.csv", encoding="utf-8-sig")

# 학습 데이터의 상담 유형을 정렬해 목록으로 만듦
labels = sorted(train_df["qa_topic"].unique())

# 상담 유형과 숫자 ID를 양뱡향으로 연결
label2id = {label: index for index, label in enumerate(labels)}
id2label = {index: label for index, label in enumerate(labels)}

# 문자열 정답을 숫자 정답으로 변환
train_df["labels"] = train_df["qa_topic"].map(label2id)
validation_df["labels"] = validation_df["qa_topic"].map(label2id)

print(f"학습 질문 수: {len(train_df)}")
print(f"검증 질문 수: {len(validation_df)}")
print(f"클래스 수: {len(labels)}")
print(f"라벨 -> ID: {label2id}")
print(f"검증 라벨 변환 누락: {validation_df["labels"].isna().sum()}")



# 사용할 사전학습 모델의 이름을 지정
model_name = "klue/bert-base"
# 해당 모델에 맞는 토크나이저를 설정
tokenizer = AutoTokenizer.from_pretrained(model_name)

# sample_text = "대출 금리를 알려주세요."
# encoded = tokenizer(sample_text)
# tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
# print(f"\n원문: {sample_text}")
# print(f"Token: {tokens}")
# print(f"Token ID: {encoded["input_ids"]}")
# print(f"Attention mask: {encoded["attention_mask"]}")



# 질문과 클래스를 학습용 Dataset으로 구성
train_dataset = Dataset.from_pandas(
    train_df[["question", "labels"]],
    preserve_index=False,
)

validation_dataset = Dataset.from_pandas(
    validation_df[["question", "labels"]],
    preserve_index=False,
)

# 질문을 토큰화하는 처리 함수 정의
def tokenize_batch(batch):
    return tokenizer(batch["question"], truncation=True, max_length=512)

# 토큰화 결과를 추가하고, 학습 입력에서 원문 컬럼을 제거
train_dataset = train_dataset.map(tokenize_batch, batched=True, remove_columns=["question"])
validation_dataset = validation_dataset.map(tokenize_batch, batched=True, remove_columns=["question"])

# print(f"\n학습 Dataset: {train_dataset}")
# print(f"\n검증 Dataset: {validation_dataset}")
# print(f"\n첫 번째 학습 데이터: {train_dataset[0]}")



# 새 가중치의 무작위 초기화를 재현할 수 있도록 난수 기준을 지정
set_seed(42)

# 사전학습 BERT 가중치를 불러오고 분류 모델을 구성
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

print(f"\n모델 종류: {type(model).__name__}")
print(f"분류층: {model.classifier}")
print(f"출력 클래스 수: {model.config.num_labels}")



# 배치 안에서 질문 길이를 맞추고 Pytorch Tensor로 반환
data_collator = DataCollatorWithPadding(
    # tokenizer: PreTrainedTokenizerBase,
    tokenizer=tokenizer,
    # padding: bool | str | PaddingStrategy = True,
    # max_length: int | None = None,
    # pad_to_multiple_of: int | None = None,
    # return_tensors: str = "pt"
    return_tensors="pt",
)

# samples = [train_dataset[i] for i in range(3)]
# print(f"\n원래 토큰 길이: {[
#     len(sample["input_ids"]) for sample in samples
#     ]}")
# batch = data_collator(samples)
# for name, tensor in batch.items():
#     print(f"{name}: {tensor.shape}")
# print(f"배치의 정답: {batch["labels"]}")



# 학습의 체크포인트를 저장할 폴더
output_dir = Path(__file__).resolve().parent.parent / "models" / "bert_run01"

# BERT 학습 설정
training_args = TrainingArguments(
    output_dir=str(output_dir),

    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=0.1,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    bf16=True,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,

    seed=42,
    report_to="none",
)

# print(f"\n학습 장치: {training_args.device}")
# print(f"학습 epoch 수: {training_args.num_train_epochs}")
# print(f"학습 배치 크기: {training_args.per_device_train_batch_size}")
# print(f"학습률: {training_args.learning_rate}")
# print(f"BF16 사용: {training_args.bf16}")



# 평가 결과에서 Accuracy와 Macro F1을 계산
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    label_ids = eval_pred.label_ids
    predicted_ids = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(label_ids, predicted_ids),
        "macro_f1": f1_score(label_ids, predicted_ids, average="macro", zero_division=0)
    }

# 준비한 모델, 데이터, 설정, 처리 도구를 연결
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# 기존 학습 체크포인트가 있으면 덮어쓰지 않고 중단
if any(output_dir.glob("checkpints-*")):
    raise FileExistsError("기존 체크포인트가 있습니다. 새로운 실험 폴더를 지정하세요.")

# 학습 진행
train_result = trainer.train()

print(f"\n학습 완료")
print(f"학습 요약: {train_result.metrics}")
print(f"최고 검증 Macro F1: {trainer.state.best_metric}")
print(f"선택된 체크포인트: {trainer.state.best_model_checkpoint}")