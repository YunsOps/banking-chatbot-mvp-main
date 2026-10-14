from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
    TrainingArguments, Trainer,
)



project_dir = Path(__file__).resolve().parent.parent
checkpoint_dir = project_dir / "models" / "bert_run01" / "checkpoint-7473"
data_dir = project_dir / "data" / "dataset_v2"

# 상담 분류 학습을 마친 모델과 토크나이저를 불러옴
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

test_df = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")

# 학습 당시 저장한 라벨 매핑을 그대로 사용
label2id = model.config.label2id
id2label = model.config.id2label

test_df["labels"] = test_df["qa_topic"].map(label2id)

if test_df["labels"].isna().any():
    raise ValueError("test 데이터에 모델이 학습하지 않은 라벨이 있습니다.")

test_df["labels"] = test_df["labels"].astype("int64")

# print(f"평가 체크포인트: {checkpoint_dir}")
# print(f"Test 질문 수: {len(test_df)}")
# print(f"클래스 수: {model.config.num_labels}")



test_dataset = Dataset.from_pandas(
    test_df[["question", "labels"]],
    preserve_index=False,
)

def tokenize_batch(batch):
    return tokenizer(batch["question"], truncation=True, max_length=512)

test_dataset = test_dataset.map(tokenize_batch, batched=True, remove_columns=["question"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# print(f"Test Dataset: {test_dataset}")

evaluation_args = TrainingArguments(
    output_dir=str(project_dir / "models" / "bert_test_eval"),
    per_device_eval_batch_size=16,
    bf16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=evaluation_args,
    processing_class=tokenizer,
    data_collator=data_collator,
)

prediction_output = trainer.predict(test_dataset)

logits = prediction_output.predictions
true_ids = prediction_output.label_ids
predicted_ids = np.argmax(logits, axis=1)

class_ids = list(range(model.config.num_labels))
class_names = [id2label[class_id] for class_id in class_ids]

accuracy = accuracy_score(true_ids, predicted_ids)
macro_f1 = f1_score(true_ids, predicted_ids, labels=class_ids, average="macro", zero_division=0)

print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Macro F1: {macro_f1:.4f}")
print(f"\nTest 상담 유형별 평가:")
print(classification_report(true_ids, predicted_ids, labels=class_ids, target_names=class_names, digits=4, zero_division=0))

# 업무 외 탐지와 정상 은행 문의 오차단 평가
oos_id = label2id["업무 외"]

actual_oos = true_ids == oos_id
predicted_oos = predicted_ids == oos_id

oos_detection_rate = predicted_oos[actual_oos].mean()

actual_bank = ~actual_oos
false_block_count = np.sum(actual_bank & predicted_oos)
false_block_rate = predicted_oos[actual_bank].mean()

print(f"\nTest 업무 외 탐지율: {oos_detection_rate:.2%}")
print(f"Test 정상 은행 문의 오차단 건수: {false_block_count}")
print(f"Test 정상 은행 문의 오차단율: {false_block_rate:.2%}")

# 최종 모델 저장
final_dir = project_dir / "models" / "klue_bert_base"

# 기존 저장 폴더가 있으면 덮어쓰지 않고 중단
final_dir.mkdir(parents=True, exist_ok=False)

# 학습된 모델 가중치와 설정 저장
model.save_pretrained(final_dir)

# 토크나이저 저장
tokenizer.save_pretrained(final_dir)

print(f"최종 모델과 토크나이저 저장 완료: {final_dir}")
print("저장된 파일:")
for path in sorted(final_dir.iterdir()):
    print(f"- {path.name}")