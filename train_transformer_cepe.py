from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.model_selection import train_test_split

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

def apply_smote(tokenizer, features, labels, max_len):
    tokenized = tokenizer(
        features,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    X = tokenized["input_ids"]
    y = np.array(labels)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    attention_masks = (X_resampled != 0).astype(int)

    resampled_inputs = [
        {"input_ids": X_resampled[i], "attention_mask": attention_masks[i]}
        for i in range(X_resampled.shape[0])
    ]

    return resampled_inputs, y_resampled

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

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, num_classes, max_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = x.squeeze(1).permute(1, 0, 2)  # Switch to (seq_len, batch, d_model)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.mean(dim=0)  # Mean pooling
        return self.fc(x)

# Load Data
def load_data(file_path):
    data = pd.read_csv(file_path)
    features = data[
        ["topic_sentence", "evidence_label", "claim_sentence", "evidence_candidate_sentence", "article_id"]
    ].astype(str).agg(' '.join, axis=1).tolist()
    labels = data["encoded_label"].astype(int).tolist()
    return features, labels

# Apply SMOTE
def apply_smote(tokenizer, features, labels, max_len):
    tokenized = tokenizer(
        features,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    X = tokenized["input_ids"]
    y = np.array(labels)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    attention_masks = (X_resampled != 0).astype(int)

    resampled_inputs = [
        {"input_ids": X_resampled[i], "attention_mask": attention_masks[i]}
        for i in range(X_resampled.shape[0])
    ]

    return resampled_inputs, y_resampled

# Training Function
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
    file_path = "./encoded_cleaned_cepe_data.csv"

    # Load data
    features, labels = load_data(file_path)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    max_len = 128

    # Split data before applying SMOTE
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Apply SMOTE only on training data
    resampled_inputs, resampled_labels = apply_smote(tokenizer, train_features, train_labels, max_len)

    # Tokenize test data without SMOTE
    tokenized_test = tokenizer(
        test_features,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    test_inputs = [
        {"input_ids": tokenized_test["input_ids"][i], "attention_mask": tokenized_test["attention_mask"][i]}
        for i in range(tokenized_test["input_ids"].shape[0])
    ]

    # Create datasets
    train_dataset = CESCDataSet(resampled_inputs, resampled_labels)
    test_dataset = CESCDataSet(test_inputs, test_labels)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Model parameters
    vocab_size = len(tokenizer)
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    num_classes = len(set(labels))
    dropout = 0.1

    # Initialize model
    model = TransformerModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, num_classes, max_len, dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(20):
        train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}")

    # Evaluate model
    report = evaluate_model(model, test_loader, device)
    print("Evaluation Report:")
    print(report)

if __name__ == "__main__":
    main()

