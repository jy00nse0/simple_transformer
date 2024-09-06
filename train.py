import torch
from transformers import TrainingArguments, Trainer
from prepare import model, encoded_dataset

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # 1 에폭만 훈련
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
)

# 훈련 실행
trainer.train()

# 모델 저장
trainer.save_model("./saved_model")

print("훈련 완료 및 모델 저장됨")
