from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import os

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

class RobertaClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits
    
# Load Data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Extract relevant columns
    topics = data["topic_sentence"].tolist()
    claims = data["claim_sentence"].tolist()  # Updated column name
    article_ids = data["article_id"].tolist()
    labels = data["encoded_label"].astype(int).tolist()  # Updated column name

    return topics, claims, article_ids, labels

# Apply SMOTE
def apply_smote(tokenizer, topics, claims, article_ids, labels, max_len):
    # Combine relevant columns into input text
    combined_text = [f"{t} {c} Article ID: {a}" for t, c, a in zip(topics, claims, article_ids)]

    # Tokenize combined text
    tokenized = tokenizer(
        combined_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    X = tokenized["input_ids"]
    y = np.array(labels)

    # Apply SMOTE to balance the dataset
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
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
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
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            _, batch_preds = torch.max(outputs, dim=1)
            preds.extend(batch_preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    report = classification_report(targets, preds, zero_division=0)
    return report

def main():
    # Path
    file_path = "all_claims.txt"

    # Load Data
    topics, claims, labels = load_data(file_path)

    # Map labels
    label_mapping = {-1: 0, 0: 1, 1: 2}
    labels = [label_mapping[label] for label in labels]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    max_len = 128

    # Tokenize topics and claims
    inputs = tokenizer(
        list(zip(topics, claims)),
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="np"
    )["input_ids"]

    # Split data into train/test sets
    train_inputs, test_inputs, train_labels, test_labels = train_test_split(
        inputs, labels, test_size=0.2, random_state=42
    )

    # Apply SMOTE only to the training data
    train_inputs, train_labels = apply_smote(train_inputs, train_labels)

    # Create Dataset and DataLoader
    train_dataset = CESCDataSet(train_inputs, train_labels)
    test_dataset = CESCDataSet(test_inputs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Model Parameters
    vocab_size = len(tokenizer)  # Use tokenizer's vocab size
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 256
    num_classes = len(set(labels))  # Number of classes in labels
    dropout = 0.1

    # Initialize Model
    model = TransformerModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, num_classes, max_len, dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer and Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(20):
        train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}")

    # Evaluation
    report = evaluate_model(model, test_loader, device)
    print("Evaluation Report:")
    print(report)

if __name__ == "__main__":
    main()