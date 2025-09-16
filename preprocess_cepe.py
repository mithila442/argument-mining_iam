import pandas as pd
import os

# Load the dataset
file_path = "cepe_dataset.txt"  # Replace with the actual file path
try:
    data = pd.read_csv(
        file_path,
        delimiter="\t",  # Assuming tab-separated values
        header=None,  # No predefined headers
        names=[
            'claim_label', 'topic_sentence', 'evidence_label',
            'claim_sentence', 'evidence_candidate_sentence', 'article_id', 'full_label'
        ],
        skipinitialspace=True
    )
except Exception as e:
    raise RuntimeError(f"Error loading file: {e}")

# Define helper functions
def clean_claim_label(label):
    """Validate and map claim labels. 'C' -> 1 (claim), 'O' -> 0 (non-claim)."""
    return 1 if label == 'C' else 0 if label == 'O' else None

def clean_evidence_label(label):
    """Validate and map evidence labels. 'E' -> 1 (evidence), 'O' -> 0 (non-evidence)."""
    return 1 if label == 'E' else 0 if label == 'O' else None

def validate_full_label(label):
    """
    Validate the format of full_label (e.g., 'C-index', 'E-B-index', 'E-I-index', 'O').
    """
    if label == 'O':
        return label
    if isinstance(label, str) and all(
        part.startswith(('C-', 'E-B-', 'E-I-')) or part == 'O'
        for part in label.split('|')
    ):
        return label
    return None

def validate_article_id(article_id):
    """
    Validate article_id format as 'number_number' (e.g., '1_3').
    """
    return isinstance(article_id, str) and bool(pd.Series([article_id]).str.match(r'^\d+_\d+$').iloc[0])

# Clean and validate data
try:
    data['claim_label'] = data['claim_label'].apply(clean_claim_label)
    data['evidence_label'] = data['evidence_label'].apply(clean_evidence_label)
    data['full_label'] = data['full_label'].apply(validate_full_label)

    valid_rows = (
        data['claim_label'].notna() &
        data['evidence_label'].notna() &
        data['full_label'].notna() &
        data['topic_sentence'].apply(lambda x: isinstance(x, str) and x.strip() != "") &
        data['claim_sentence'].apply(lambda x: isinstance(x, str) and x.strip() != "") &
        data['evidence_candidate_sentence'].apply(lambda x: isinstance(x, str) and x.strip() != "") &
        data['article_id'].apply(validate_article_id)
    )

    # Separate valid and invalid rows
    cleaned_data = data[valid_rows]
    discarded_data = data[~valid_rows]

    # Save processed data
    cleaned_data.to_csv("cleaned_cepe_data.csv", index=False)
    discarded_data.to_csv("discarded_cepe_data.csv", index=False)

except Exception as e:
    raise RuntimeError(f"Error during preprocessing: {e}")

# Load the cleaned dataset
file_path = 'cleaned_cepe_data.csv'  # Replace with the correct file path
data = pd.read_csv(file_path)

# Add a new column 'combined_label' which concatenates 'claim_label' and 'evidence_label'
try:
    data['combined_label'] = data['claim_label'].astype(str) + "_" + data['evidence_label'].astype(str)

    # Save the updated dataset
    output_file = 'updated_cleaned_cepe_data.csv'  # Update the file name/path as needed
    data.to_csv(output_file, index=False)

except Exception as e:
    raise RuntimeError(f"Error while adding the combined_label column: {e}")

# Load the updated dataset
file_path = 'updated_cleaned_cepe_data.csv'  # Replace with the correct file path
data = pd.read_csv(file_path)

# Define the label encoding mapping
label_encoding = {
    '0_0': 0,
    '0_1': 1,
    '1_0': 2,
    '1_1': 3
}

# Add a new column 'encoded_label' with encoded values
try:
    data['encoded_label'] = data['combined_label'].map(label_encoding)

    # Save the updated dataset with encoded labels
    output_file = 'encoded_cleaned_cepe_data.csv'  # Update the file name/path as needed
    data.to_csv(output_file, index=False)

except Exception as e:
    raise RuntimeError(f"Error while encoding the combined_label column: {e}")