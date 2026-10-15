from pathlib import Path
import pandas as pd
import joblib
from training.korean_tokenizer import tokenize_korean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# 데이터 경로 설정
data_dir = Path(__file__).resolve().parent.parent / "data" / "dataset_v2"

# 학습 데이터와 검증 데이터 로드
train_df = pd.read_csv(data_dir / "train.csv", encoding="utf-8-sig")
validation_df = pd.read_csv(data_dir / "validation.csv", encoding="utf-8-sig")

# 입력과 정답 로드
X_train = train_df["question"]
X_validation = validation_df["question"]

y_train = train_df["qa_topic"]
y_validation = validation_df["qa_topic"]

# print(f"학습 질문 수: {len(X_train)}")
# print(f"검증 질문 수: {len(X_validation)}")
# print(f"\n학습 라벨렬 수량: ")
# print(y_train.value_counts())
# print(f"\n첫 번째 질문: {X_train.iloc[0]}")
# print(f"첫 번째 정답: {y_train.iloc[0]}")



# Kiwi로 형태소를 분리하고, 형태소 1개와 연속된 형태소 2개를 TF-IDF 특성으로 사용
vectorizer = TfidfVectorizer(
    # *,
    # input: Literal['filename', 'file', 'content'] = "content",
    # encoding: str = "utf-8",
    # decode_error: Literal['strict', 'ignore', 'replace'] = "strict",
    # strip_accents: ((...) -> Unknown) | Literal['ascii', 'unicode'] | None = None,
    # lowercase: bool = True,
    # preprocessor: ((...) -> Unknown) | None = None,
    # tokenizer: ((...) -> Unknown) | None = None,
    tokenizer=tokenize_korean,
    # analyzer: ((...) -> Unknown) | Literal['word', 'char', 'char_wb'] = "word",
    analyzer="word",
    # stop_words: str | Unknown | None = None,
    # token_pattern: str = r"(?u)\b\w\w+\b",
    token_pattern=None,
    # ngram_range: tuple[float, float] = ...,
    ngram_range=(1, 2),
    # max_df: Float = 1,
    # min_df: Float = 1,
    # max_features: Int | None = None,
    # vocabulary: Iterable[Unknown] | Mapping[Unknown, Unknown] | None = None,
    # binary: bool = False,
    # dtype: ... = ...,
    # norm: Literal['l1', 'l2'] | None = "l2",
    # use_idf: bool = True,
    # smooth_idf: bool = True,
    # sublinear_tf: bool = False
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_validation_tfidf = vectorizer.transform(X_validation)
print(f"학습 TF-IDF 크기: {X_train_tfidf.shape}")
print(f"검증 TF-IDF 크기: {X_validation_tfidf.shape}\n")


# 모델 설정
model = LogisticRegression(
    # penalty: Literal['l1', 'l2', 'elasticnet'] | None = "l2",
    # *,
    # dual: bool = False,
    # tol: Float = 0.0001,
    # C: Float = 1,
    # fit_intercept: bool = True,
    # intercept_scaling: Float = 1,
    # class_weight: Mapping[Unknown, Unknown] | str | None = None,
    # random_state: Int | None = None,
    # random_state=42,
    # solver: Literal['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'] = "lbfgs",
    # max_iter: Int = 100,
    max_iter=1000,
    # multi_class: Literal['auto', 'ovr', 'multinomial'] = "auto",
    # verbose: Int = 0,
    # warm_start: bool = False,
    # n_jobs: Int | None = None,
    # l1_ratio: Float | None = None
)
model.fit(X_train_tfidf, y_train)
print(f"학습한 클래스 수: {len(model.classes_)}")
print(f"실제 최적화 반복 횟수: {model.n_iter_}")



# 모델 검증
y_validation_pred = model.predict(X_validation_tfidf)
accuracy = accuracy_score(y_validation, y_validation_pred)
macro_f1 = f1_score(y_validation, y_validation_pred, average="macro", zero_division=0)
# print(f"\n검증 Accuracy: {accuracy:.4f}")
# print(f"검증 Macro F1: {macro_f1:.4f}")
# print(f"상담 유형별 평가:")
# print(classification_report(y_validation, y_validation_pred, digits=4, zero_division=0))



# 모델 최종 평가
test_df = pd.read_csv(data_dir / "test.csv", encoding="utf-8-sig")
X_test = test_df["question"]
y_test = test_df["qa_topic"]

X_test_tfidf = vectorizer.transform(X_test)
y_test_pred = model.predict(X_test_tfidf)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

print(f"\nTest TF-IDF 크기: {X_test_tfidf.shape}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"\nTest 상담 유형별 평가:")
print(classification_report(y_test, y_test_pred, digits=4, zero_division=0))

# 실제 업무 외 질문과 정상 은행 질문의 위치를 구분
oos_mask = y_test == "업무 외"
bank_mask = y_test != "업무 외"

# 모델이 업무 외라고 예측한 위치를 표시
predicted_oos = y_test_pred == "업무 외"

# 실제 업무 외 중 탐지한 비율을 계산
oos_recall = predicted_oos[oos_mask].mean()

# 정상 은행 문의 중 업무 외로 잘못 차단한 건수와 비율을 계산
false_block_count = predicted_oos[bank_mask].sum()
false_block_rate = predicted_oos[bank_mask].mean()

print(f"\nTest 업무 외 탐지율: {oos_recall:.2%}")
print(f"Test 정상 은행 문의 오차단 건수: {false_block_count}")
print(f"Test 정상 은행 문의 오차단율: {false_block_rate:.2%}")



# 모델 저장
model_dir = Path(__file__).resolve().parent.parent / "models"
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "tfidf_logistic_regression.joblib"

saved_model = {
    "vectorizer": vectorizer,
    "model": model,
}

with model_path.open("xb") as file:
    joblib.dump(saved_model, file)
print(f"\n모델 저장 완료: {model_path}")
