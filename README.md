# Automating Argument Mining: NLP Project

Extracting and analyzing claims, stances, and evidence from texts using advanced transformer models.

---

## Project Overview

This project automates two major argument mining tasks in NLP:
- **Claim Extraction with Stance Classification (CESC)**
- **Claim-Evidence Pair Extraction (CEPE)**

By leveraging advanced transformer models such as RoBERTa and Swin Transformers, it improves upon traditional baselines and achieves stronger balanced performance on complex argumentative data.

---

## Key Tasks

- **CESC**: Identify argumentative claims and classify stance (support/contest/no relation) towards debate topics.
- **CEPE**: Map claims to supporting evidence, uncovering deeper argumentative relationships.

---

## Dataset

- **IAM Dataset:** 70,000 sentences from 1,000+ publications, 123 topics.
- Download: [IAM Github](https://github.com/LiyingCheng95/IAM)
- Main columns:
    - CESC: `claim_label`, `topic_sentence`, `claim_candidate_sentence`, `stance_label`
    - CEPE: `claim_label`, `topic_sentence`, `evidence_label`, `claim_sentence`, `evidence_candidate_sentence`

---

## Methodology

- **Preprocessing**: Data cleaning, missing value handling, label encoding.
- **Balancing**: Synthetic Minority Over-Sampling Technique (SMOTE) to address class imbalance.
- **Model Architectures**:
    - Transformer baseline
    - RoBERTa fine-tuning
    - Swin Transformer adaptation for text
- **Training**: AdamW optimizer, regularizations
- **Evaluation**: Macro-averaged F1, Precision, Recall, and Accuracy

---

# Directory Structure

├── baseline.ipynb              # Baseline model experiments  
├── cepe_project.ipynb          # Full pipeline for CEPE  
├── preprocess_cepe.py          # Preprocessing scripts for CEPE data  
├── project_cesc2.ipynb         # Full pipeline for CESC  
├── swinCESC.py                 # Swin Transformer CESC model code  
├── swinCEPE.py                 # Swin Transformer CEPE model code  
├── train_roberta_cesc.py       # Train RoBERTa model for CESC  
├── train_roberta_cepe.py       # Train RoBERTa model for CEPE  
├── train_transformer_cesc.py   # Train baseline Transformer for CESC  
└── train_transformer_cepe.py   # Train baseline Transformer for CEPE  


---

## Key Tasks

**Claim Extraction with Stance Classification (CESC):**
- Identifies claims and classifies stances:
  - Support
  - Contest
  - No Relation
- Uses joint label encoding and advanced transformers.

**Claim-Evidence Pair Extraction (CEPE):**
- Pairs claims with relevant evidence.
- Employs transformer models and effective label encoding strategies.

---

## Getting Started

1. **Clone the repository:**
    ```
    git clone https://github.com/mithila442/argument-mining_iam.git
    cd argument-mining_iam
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    # Additional: 
    # pip install torch scikit-learn imbalanced-learn transformers notebook
    ```

3. **Prepare the dataset:**
    - Download IAM or other argument mining datasets.
    - Place raw files in a `data/` folder.

4. **Preprocess data (example):**
    ```
    python preprocess_cepe.py
    ```
    Or run initial cells in notebooks.

---

## Scripts & Notebooks

### Notebooks (.ipynb)
- **baseline.ipynb:** Quick baseline metrics for comparison.
- **cepe_project.ipynb:** End-to-end modeling and evaluation for the CEPE task.
- **project_cesc2.ipynb:** End-to-end modeling and evaluation for CESC.

### Scripts (.py)
- **preprocess_cepe.py:** Data cleaning and encoding for CEPE.
- **swinCESC.py / swinCEPE.py:** Swin Transformer architectures.
- **train_roberta_cesc.py / train_roberta_cepe.py:** RoBERTa model training routines.
- **train_transformer_cesc.py / train_transformer_cepe.py:** Baseline Transformer model training.

---

## Results & Evaluation

Model performances (macro-averaged metrics preferred, illustrating fair class treatment):

| Model                | Precision | Recall | F1-score | Accuracy | Task  |
|----------------------|-----------|--------|----------|----------|-------|
| Random Baseline      | 0.34      | 0.34   | 0.34     | 0.86     | CESC  |
| Most Frequent Class  | 0.31      | 0.33   | 0.32     | 0.93     | CESC  |
| RoBERTa              | 0.61      | 0.63   | 0.62     | 0.92     | CESC  |
| Swin Transformer     | 0.56      | 0.53   | 0.52     | 0.88     | CESC  |
| Transformer Baseline | 0.34      | 0.35   | 0.34     | 0.92     | CESC  |

Similar improvements observed for CEPE.

---

## Team

- **Md Shahriar Kabir** — Data preprocessing, RoBERTa for CEPE
- **Swarna Chakraborty** — Swin Transformer design, baseline models
- **Mayesha Maliha Rahman** — Transformer baseline, RoBERTa, optimization

---

## Acknowledgement

This project is performed under Department of Computer Science, Texas State University

---

