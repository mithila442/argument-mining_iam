import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import csv
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler
from collections import Counter
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

import logging

# Configure logging
logging.basicConfig(
    filename="training_log_swinCepe.txt",  # Log file path
    level=logging.INFO,  # Logging level
    format="%(asctime)s - %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Date format
)

def load_data_cepe(file_path):
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            quoting=csv.QUOTE_NONE,
            engine='python',
            on_bad_lines='skip'
        )
        df.columns = ['claim_labels', 'topic_sentence', 'evidence_label', 'claim_sentence', 'evidence_candidate_sentence', 'article_id', 'full_label']
        
        topic_sentences = df['topic_sentence'].tolist()
        claim_sentences = df['claim_sentence'].tolist()
        evidence_candidate_sentences = df['evidence_candidate_sentence'].tolist()
        combined_labels = [f"{c}_{e}" for c, e in zip(df['claim_labels'], df['evidence_label'])]
        
        # Check adherence to the four classes
        valid_classes = {"O_O", "C_E", "C_O", "O_E"}
        invalid_classes = set(combined_labels) - valid_classes
        if invalid_classes:
            print(f"Warning: Found invalid classes: {invalid_classes}")

        return topic_sentences, claim_sentences, evidence_candidate_sentences, combined_labels
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], [], [], []
    
class CEPEDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
    
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
    
# Swin-inspired Transformer Model for CEPE
class SwinTransformerForCEPE(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, max_len, window_size, depth=2, dropout=0.1):
        super(SwinTransformerForCEPE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(d_model, num_heads, window_size, dropout) for _ in range(depth)
        ])
        self.fc = nn.Linear(d_model, 4)  # Four classes: O_O, C_E, C_O, E_E

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # Mean pooling
        logits = self.fc(x)
        return logits

def train_model(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch and computes the average loss and accuracy.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to train on.

    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids)

        # Compute loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute predictions
        _, preds = torch.max(outputs, dim=1)  # Get class with highest score
        correct_predictions += (preds == labels).sum().item()  # Count correct predictions
        total_samples += labels.size(0)  # Count total samples

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples  # Calculate accuracy

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
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    report = classification_report(targets, preds, zero_division=0)
    return report

def apply_smote_to_cepe(train_topics, train_claims, train_evidence, train_labels, tokenizer, max_len, label_map):
    """
    Tokenizes the text fields, applies SMOTE, and returns tokenized inputs and balanced labels.
    """
    # Combine text fields for tokenization
    combined_texts = [
        f"Topic: {topic} Claim: {claim} Evidence: {evidence}"
        for topic, claim, evidence in zip(train_topics, train_claims, train_evidence)
    ]

    # Tokenize combined text
    tokenized = tokenizer(
        combined_texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )

    # Extract tokenized input IDs and labels
    input_ids = tokenized["input_ids"]
    attention_masks = tokenized["attention_mask"]
    labels = np.array([label_map[label] for label in train_labels])

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(input_ids, labels)

    # Adjust attention masks for resampled data
    attention_masks_resampled = (X_resampled != 0).astype(int)

    # Return resampled tokenized inputs and labels
    return X_resampled, attention_masks_resampled, y_resampled

def main():
    # File paths
    train_file = "/home/wnp23/nlpProject/IAM/CEPE/train.txt"
    test_file = "/home/wnp23/nlpProject/IAM/CEPE/test.txt"

    # Define label mapping
    label_map = {"O_O": 0, "C_E": 1, "C_O": 2, "O_E": 3}

    # Load training and test data
    train_topics, train_claims, train_evidence, train_labels = load_data_cepe(train_file)
    test_topics, test_claims, test_evidence, test_labels = load_data_cepe(test_file)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_len = 128

    # Tokenize and balance training data using SMOTE
    train_input_ids, train_attention_masks, train_labels = apply_smote_to_cepe(
        train_topics, train_claims, train_evidence, train_labels, tokenizer, max_len, label_map
    )

    # Tokenize test data (without SMOTE)
    combined_test_texts = [
        f"Topic: {topic} Claim: {claim} Evidence: {evidence}"
        for topic, claim, evidence in zip(test_topics, test_claims, test_evidence)
    ]
    tokenized_test = tokenizer(
        combined_test_texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    test_input_ids = tokenized_test["input_ids"]
    test_attention_masks = tokenized_test["attention_mask"]
    test_labels = [label_map[label] for label in test_labels]

    # Create datasets
    train_dataset = CEPEDataset(train_input_ids, train_attention_masks, train_labels)
    test_dataset = CEPEDataset(test_input_ids, test_attention_masks, test_labels)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)

    #Model Parameters
    vocab_size = len(tokenizer)
    d_model = 128
    num_heads = 8
    window_size = 4

    # Initialize model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = SwinTransformerForCEPE(vocab_size, d_model, num_heads, max_len, window_size).to(device)

    # Optimizer, scheduler, and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * 10
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(50):
        train_loss, train_accuracy = train_model(model, train_loader, optimizer, criterion, device)
        log_message = f"Epoch {epoch + 1}: Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}"
        print(log_message)  # Print to console
        logging.info(log_message)  # Log to file
        scheduler.step()

    # Evaluate
    report = evaluate_model(model, test_loader, device)
    print("Evaluation Report:")
    print(report)
    logging.info("Evaluation Report:\n" + report)


if __name__ == "__main__":
    main()
