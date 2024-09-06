import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 모델과 토크나이저 로드
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 추론 함수
def inference(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return "Positive" if predicted_class == 1 else "Negative"

# 샘플 텍스트로 추론 실행
sample_text = "This movie was fantastic! I really enjoyed every moment of it."
result = inference(sample_text)
print(f"Sample text: {sample_text}")
print(f"Prediction: {result}")
