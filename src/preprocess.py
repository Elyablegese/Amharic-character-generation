import os
import re
import sys
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
MODEL_DICT = {
    "rasyosef/bert-amharic-tokenizer": "rasyosef/bert-amharic-tokenizer",
    "rasyosef/bert-medium-amharic": "rasyosef/bert-medium-amharic",
    "NathyB/Hate-Speech-Detection-in-Amharic-Language-mBERT": "NathyB/Hate-Speech-Detection-in-Amharic-Language-mBERT",
    "iocuydi/llama-2-amharic-3784m": "iocuydi/llama-2-amharic-3784m",
    "xlm-roberta-base": "xlm-roberta-base",
    "xlm-roberta-large": "xlm-roberta-large",
    "bert-base-multilingual-cased": "bert-base-multilingual-cased",
    "facebook/mbart-large-cc25": "facebook/mbart-large-cc25",
}

# --- Text Preprocessing Functions ---
def remove_noise(text):
    return re.sub(r'[^ሀ-፿\s]+', '', text)

def standardize_text(text):
    return text.strip()

def tokenize_text(tokenizer, text):
    return tokenizer.tokenize(text)

def preprocess_amharic_text(tokenizer, text):
    text = remove_noise(text)
    text = standardize_text(text)
    tokens = tokenize_text(tokenizer, text)
    return " ".join(tokens)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_dataframe(df, tokenizer, is_amharic=False):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            if is_amharic:
                df[col] = df[col].apply(lambda x: preprocess_amharic_text(tokenizer, x))
            else:
                df[col] = df[col].apply(preprocess_text)
    return df

def split_text(lines):
    if len(lines) < 5:
        return lines, [], []
    train, temp = train_test_split(lines, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)
    return train, val, test

# --- Model Loading ---
def load_model(model_name):
    print(f"Loading model '{model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[model_name])
        model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[model_name])
        if torch.cuda.is_available():
            model.to('cuda')
            print("Using GPU acceleration.")
        else:
            print("CUDA not available. Using CPU.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# --- File Processing ---
def process_file(file_path, tokenizer, is_amharic, output_dir):
    _, ext = os.path.splitext(file_path)
    try:
        if ext in ['.csv', '.xls', '.xlsx']:
            print(f"Processing {file_path}...")
            if ext == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            else:
                df = pd.read_excel(file_path)
            df = clean_dataframe(df, tokenizer, is_amharic)
            text_for_splitting = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).str.cat(sep='\n')
            lines = text_for_splitting.split('\n')
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if is_amharic:
                cleaned_text = preprocess_amharic_text(tokenizer, text)
            else:
                cleaned_text = preprocess_text(text)
            lines = cleaned_text.split('\n')
        else:
            print(f"Unsupported file type: {file_path}")
            return None, None, None
        return split_text(lines)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

# --- Main Function ---
def main(args):
    # Load model
    tokenizer, model = load_model(args.model_name)

    merged_train_lines = []
    merged_val_lines = []
    merged_test_lines = []

    is_amharic = "amharic" in args.model_name.lower()

    for file_path in args.input_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        train, val, test = process_file(file_path, tokenizer, is_amharic, args.output_dir)
        if train:
            merged_train_lines.extend(train)
        if val:
            merged_val_lines.extend(val)
        if test:
            merged_test_lines.extend(test)

    safe_model_name = re.sub(r'[^a-zA-Z0-9_]+', '-', args.model_name)

    # Save outputs
    def save_data(data, filename):
        path = os.path.join(args.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(data))
        print(f"Saved: {path}")

    if merged_train_lines:
        save_data(merged_train_lines, f"{safe_model_name}_train.txt")
    if merged_val_lines:
        save_data(merged_val_lines, f"{safe_model_name}_val.txt")
    if merged_test_lines:
        save_data(merged_test_lines, f"{safe_model_name}_test.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files using HuggingFace models.")
    parser.add_argument("--model_name", choices=list(MODEL_DICT.keys()), required=True,
                        help="Pretrained model name")
    parser.add_argument("--input_files", nargs='+', required=True,
                        help="List of input file paths (CSV, TXT, Excel)")
    parser.add_argument("--output_dir", default="./output", help="Directory to save processed files")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
