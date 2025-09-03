import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import os

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = []
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    return classification_report(true_labels, predictions, output_dict=True)

def main():
    df = pd.read_csv('train_clean.csv')
    if 'label' not in df.columns:
        raise ValueError("Column 'label' not found in train_clean.csv. Please check your CSV headers.")
    X_train, X_val, y_train, y_val = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_ds = NewsDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_ds = NewsDataset(X_val.tolist(), y_val.tolist(), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    best_f1 = 0
    for epoch in range(3):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = eval_model(model, val_loader, device)
        f1 = metrics['1']['f1-score']
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs('saved_model', exist_ok=True)
            model.save_pretrained('saved_model/')
            tokenizer.save_pretrained('saved_model/')

if __name__ == "__main__":
    main()
