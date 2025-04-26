import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Load Dataset
data_path = "../data/sofmattress_train.csv" 
df = pd.read_csv(data_path)

# Basic Preprocessing
df['sentence'] = df['sentence'].str.lower().str.strip()

# Label encoding
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['label'])
intent_classes = label_encoder.classes_

# Train-Test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42, stratify=df['label_id']
)

# Dataset Class
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Tokenizer and Datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = IntentDataset(train_texts, train_labels, tokenizer, max_len=32)
val_dataset = IntentDataset(val_texts, val_labels, tokenizer, max_len=32)

# Load Pretrained BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intent_classes))

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print("\nEvaluation Results:")
print(eval_results)

# Save Model
os.makedirs("../results/bert_model", exist_ok=True)
model.save_pretrained("../results/bert_model/")
tokenizer.save_pretrained("../results/bert_model/")

print("\nBERT fine-tuned model and tokenizer saved successfully!")
