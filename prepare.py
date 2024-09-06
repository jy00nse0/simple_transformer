from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

# 토크나이저와 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 데이터셋 로드 (예: IMDB 리뷰 데이터셋)
dataset = load_dataset("imdb", split="train[:1000]")  # 1000개 샘플만 사용

# 데이터 전처리 함수
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# 데이터셋 전처리
encoded_dataset = dataset.map(preprocess_function, batched=True)

print("모델과 데이터 준비 완료")
