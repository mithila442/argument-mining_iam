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
# Load Data
def load_data(file_path):
    data = pd.read_csv(file_path)

    # Extract relevant columns
    topics = data["topic_sentence"].tolist()
    claims = data["claim_sentence"].tolist()  # Updated column name
    labels = data["encoded_label"].astype(int).tolist()  # Updated column name
    return topics, claims, labels


# Apply SMOTE
def apply_smote(tokenizer, topics, claims, labels, max_len):
    """
    Apply SMOTE to balance the classes for both train and test splits.
    
    Args:
        tokenizer: HuggingFace tokenizer for tokenizing text.
        topics: List of topic sentences.
        claims: List of claim candidate sentences.
        labels: List of class labels.
        max_len: Maximum sequence length for tokenization.
        
    Returns:
        resampled_inputs: List of tokenized resampled inputs.
        y_resampled: List of resampled class labels.
    """
    # Handle non-string entries in topics and claims
    topic_sentences = [topic if isinstance(topic, str) else "" for topic in topics]
    claim_candidates = [claim if isinstance(claim, str) else "" for claim in claims]

    # Combine topics and claims for tokenization
    combined_text = [f"{t} {c}" for t, c in zip(topic_sentences, claim_candidates)]

    # Tokenize combined text
    tokenized = tokenizer(
        combined_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    # Extract `input_ids` for SMOTE and original labels
    X = tokenized["input_ids"]  # Shape: [num_samples, max_len]
    y = np.array(labels)

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Recreate attention masks for the resampled data
    attention_masks = (X_resampled != 0).astype(int)

    # Create resampled inputs in HuggingFace-compatible format
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


from sklearn.model_selection import train_test_split

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

    # Split data into train/test sets
    train_topics, test_topics, train_claims, test_claims, train_labels, test_labels = train_test_split(
        topics, claims, labels, test_size=0.2, random_state=42
    )

    # Apply SMOTE only to the training data
    resampled_train_inputs, resampled_train_labels = apply_smote(
        tokenizer, train_topics, train_claims, train_labels, max_len
    )

    # Tokenize the test data without SMOTE
    test_combined_text = [f"{t} {c}" for t, c in zip(test_topics, test_claims)]
    tokenized_test = tokenizer(
        test_combined_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    test_inputs = [
        {"input_ids": tokenized_test["input_ids"][i], "attention_mask": tokenized_test["attention_mask"][i]}
        for i in range(len(test_combined_text))
    ]

    # Create Dataset and DataLoader
    train_dataset = CESCDataSet(resampled_train_inputs, resampled_train_labels)
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
    for epoch in range(30):
        train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}")

    # Evaluation
    report = evaluate_model(model, test_loader, device)
    print("Evaluation Report:")
    print(report)


if __name__ == "__main__":
    main()