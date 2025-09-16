import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from collections import Counter

import logging

# Configure logging
logging.basicConfig(
    filename="training_log_swinCESC.txt",  # Log file path
    level=logging.INFO,  # Logging level
    format="%(asctime)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
)

# Dataset Class
class CESCDataSet(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.features[idx]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(self.features[idx]["attention_mask"], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)


# Swin-inspired Transformer Block
class SwinTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, window_size, dropout=0.1):
        super(SwinTransformerBlock, self).__init__()
        self.window_size = window_size

        # Attention and LayerNorm
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feedforward and LayerNorm
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, seq_len, d_model = x.shape

        # Divide into windows
        x = x.view(b, seq_len // self.window_size, self.window_size, d_model)
        x = x.permute(0, 2, 1, 3).reshape(-1, self.window_size, d_model)  # Reshape to batch of windows

        # Apply attention
        x = self.self_attn(x, x, x)[0]
        x = self.norm1(x)

        # Feedforward
        x = self.ffn(x) + x
        x = self.norm2(x)

        # Reshape back
        x = x.view(-1, seq_len // self.window_size, self.window_size, d_model)
        x = x.permute(0, 2, 1, 3).reshape(b, seq_len, d_model)

        return x


# Swin-inspired Transformer Model for CESC
class SwinTransformerForCESC(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_classes, max_len, window_size, depth=2, dropout=0.1):
        super(SwinTransformerForCESC, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(d_model, num_heads, window_size, dropout) for _ in range(depth)
        ])
        self.global_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)

        for block in self.blocks:
            x = block(x)

        # Global attention
        x = x.permute(1, 0, 2)  # Switch to (seq_len, batch, d_model)
        x = self.global_attn(x, x, x)[0]
        x = x.permute(1, 0, 2)

        # Mean pooling and classification
        x = x.mean(dim=1)
        logits = self.fc(x)
        return logits


# Data Loading
def load_data(file_path):
    data = pd.read_csv(file_path, sep="\t", quoting=3, header=None)
    data.columns = ["claim_labels", "topic_sentence", "claim_candidate", "id", "labels"]
    topics = data["topic_sentence"].fillna("").tolist()
    claims = data["claim_candidate"].fillna("").tolist()
    labels = data["labels"].astype(int).tolist()
    return topics, claims, labels


# Apply SMOTE
def apply_smote(tokenizer, topics, claims, labels, max_len):
    encoded_inputs = [
        tokenizer(
            topic + " " + claim,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        for topic, claim in zip(topics, claims)
    ]
    flattened_inputs = [inputs["input_ids"].flatten() for inputs in encoded_inputs]
    X = np.array(flattened_inputs)
    y = np.array(labels)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)


    resampled_inputs = [
        {"input_ids": x.reshape(-1), "attention_mask": (x != 0).astype(int).reshape(-1)}
        for x in X_resampled
    ]
    return resampled_inputs, y_resampled
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, accuracy

# Evaluation Function
def evaluate_model(model, dataloader, device):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids)
            _, batch_preds = torch.max(outputs, dim=1)
            preds.extend(batch_preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    report = classification_report(targets, preds, zero_division=0)
    return report

# Main Function
def main():
    train_file = "/home/wnp23/nlpProject/IAM/claims/train.txt"
    test_file = "/home/wnp23/nlpProject/IAM/claims/test.txt"

    train_topics, train_claims, train_labels = load_data(train_file)
    test_topics, test_claims, test_labels = load_data(test_file)

    # Map labels
    label_mapping = {-1: 0, 0: 1, 1: 2}
    train_labels = [label_mapping[label] for label in train_labels]
    test_labels = [label_mapping[label] for label in test_labels]
    num_classes = len(label_mapping)

     # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")  # Use RoBERTa Large
    max_len = 256  # Standard maximum length for RoBERTa


    resampled_inputs, resampled_labels = apply_smote(tokenizer, train_topics, train_claims, train_labels, max_len)

    train_dataset = CESCDataSet(resampled_inputs, resampled_labels)
    test_encoded_inputs = [
        tokenizer(topic + " " + claim, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        for topic, claim in zip(test_topics, test_claims)
    ]
    test_inputs = [{"input_ids": inputs["input_ids"].squeeze(0), "attention_mask": inputs["attention_mask"].squeeze(0)}
                   for inputs in test_encoded_inputs]
    test_dataset = CESCDataSet(test_inputs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    vocab_size = len(tokenizer)
    d_model = 128
    num_heads = 8
    num_classes = len(set(train_labels))
    window_size = 8

    model = SwinTransformerForCESC(vocab_size, d_model, num_heads, num_classes, max_len, window_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device)
        log_message = f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}"
        print(log_message)  # Print to console
        logging.info(log_message)  # Log to file

    report = evaluate_model(model, test_loader, device)
    print("Evaluation Report:")
    print(report)
    logging.info("Evaluation Report:\n" + report)



if __name__ == "__main__":
    main()
