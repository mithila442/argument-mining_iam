from imblearn.over_sampling import SMOTE
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
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

    # Extract columns
    topics = data["topic_sentence"].tolist()
    claims = data["claim_sentence"].tolist()
    evidence_candidates = data["evidence_candidate_sentence"].tolist()
    labels = data["encoded_label"].astype(int).tolist()

    return topics, claims, evidence_candidates, labels

# Tokenize and Prepare Dataset Without SMOTE
def prepare_data(tokenizer, topics, claims, evidence_candidates, labels, max_len):
    # Combine relevant columns into input text
    combined_text = [f"{t} {c} {e}" for t, c, e in zip(topics, claims, evidence_candidates)]

    # Tokenize combined text
    tokenized = tokenizer(
        combined_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Create a list of dictionaries for DataLoader compatibility
    inputs = [
        {"input_ids": tokenized["input_ids"][i], "attention_mask": tokenized["attention_mask"][i]}
        for i in range(len(combined_text))
    ]

    return inputs, labels

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
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    file_path = "encoded_cleaned_cepe_data.csv"  # Update path if necessary

    # Load data
    topics, claims, evidence_candidates, labels = load_data(file_path)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    max_len = 256

    # Combine relevant columns into input text
    combined_text = [f"{t} {c} {e}" for t, c, e in zip(topics, claims, evidence_candidates)]

    # Split data into train and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        combined_text, labels, test_size=0.2, random_state=42
    )

    # Apply SMOTE to training data
    train_inputs, train_labels = apply_smote(tokenizer, train_texts, train_labels, max_len)

    # Tokenize test data (no SMOTE applied here)
    test_tokenized = tokenizer(
        test_texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    test_inputs = [
        {"input_ids": test_tokenized["input_ids"][i], "attention_mask": test_tokenized["attention_mask"][i]}
        for i in range(len(test_texts))
    ]

    # Create datasets
    train_dataset = CESCDataSet(train_inputs, train_labels)
    test_dataset = CESCDataSet(test_inputs, test_labels)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=2)

    # Define number of classes
    num_classes = len(set(labels))

    # Initialize the model
    model = RobertaClassifier(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(25):
        train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}")

    # Evaluate the model
    report = evaluate_model(model, test_loader, device)
    print("Evaluation Report:")
    print(report)

if __name__ == "__main__":
    main()
