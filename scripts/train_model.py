from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset
import json

class MedicalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data['data']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        symptoms = item['symptoms']
        diagnosis = item['diagnosis']
        text = "Symptoms: " + ", ".join(symptoms) + " Diagnosis: " + ", ".join(diagnosis)
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {key: torch.squeeze(val) for key, val in encodings.items()}

def main():
    # 데이터 로드
    with open('data/processed_train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open('data/processed_test_unseen.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    model_name = "skt/kogpt2-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 모델에도 새로운 토큰을 알려줌
    model.resize_token_embeddings(len(tokenizer))

    # 데이터셋 준비
    train_dataset = MedicalDataset(train_data, tokenizer)
    test_dataset = MedicalDataset(test_data, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir='results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='logs',
        logging_steps=10,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="epoch"
    )

    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # 모델 훈련
    trainer.train()

if __name__ == "__main__":
    main()