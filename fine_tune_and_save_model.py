# 1. Install necessary libraries
print("Installing required libraries...")
!pip install -q sentence-transformers pypdf datasets accelerate wandb
os.environ["WANDB_DISABLED"] = "true"
print("Installation complete.\n")

import os
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
import torch
import random
from google.colab import drive
import shutil
from google.colab import files
import sys

# --- Fine-Tuning and Saving Section ---

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

fine_tuned_model_path = 'fine-tuned-medical-model'

print("Fine-tuned model not found. Initializing a new model and preparing for fine-tuning...")
model = SentenceTransformer('all-mpnet-base-v2', device=device)
print(f"Sentence Transformer model loaded: {model.get_sentence_embedding_dimension()} dimensions.\n")

data_dir = '/home/Data'
if not os.path.exists(data_dir):
    print(f"Error: Directory '{data_dir}' not found. Please create it and upload your files.")
else:
    input_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                if f.endswith(('.pdf', '.txt', '.csv'))]
    all_terminology_sentences = []
    print("Processing terminology files...")
    for file_path in input_files:
        print(f"  - Processing {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()
        file_data = []
        if file_extension == '.pdf':
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    file_data.append(page.extract_text())
            except Exception as e:
                print(f"    - Failed to read PDF: {e}")
        elif file_extension == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data.extend(f.readlines())
            except Exception as e:
                print(f"    - Failed to read TXT: {e}")
        elif file_extension == '.csv':
            try:
                df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin1')
                found_text_columns = False
                common_text_columns = ['Description', 'text', 'content', 'body', 'term', 'definition']
                
                for col in common_text_columns:
                    if col in df.columns:
                        file_data.extend(df[col].dropna().tolist())
                        found_text_columns = True
                        print(f"    - Found and processed column '{col}' in {os.path.basename(file_path)}.")
                        break
                if not found_text_columns:
                    print(f"    - No common text columns found. Processing all columns in {os.path.basename(file_path)}.")
                    for col in df.columns:
                        try:
                            file_data.extend(df[col].dropna().tolist())
                        except Exception as inner_e:
                            print(f"      - Warning: Failed to process column '{col}' due to: {inner_e}")
            except Exception as e:
                print(f"    - Failed to read CSV: {e}")
        problem_found = False
        temp_list = []
        for item in file_data:
            if not isinstance(item, str):
                print(f"    - !!! Found non-string data in {os.path.basename(file_path)}.")
                print(f"    - !!! The problematic data has type: {type(item)}. Skipping this file.")
                problem_found = True
                break
            temp_list.append(item)
        if not problem_found:
            all_terminology_sentences.extend(temp_list)
            print(f"    - Successfully processed data from {os.path.basename(file_path)}.")
        else:
            print(f"    - Skipping file {os.path.basename(file_path)} due to non-string data.")
    clean_terminology_sentences = [str(s).strip() for s in all_terminology_sentences if s is not None and isinstance(s, str)]
    print(f"\nSuccessfully loaded {len(clean_terminology_sentences)} sentences from all files.")
if clean_terminology_sentences:
    print("\nStarting the fine-tuning process...")
    train_examples = []
    num_fine_tune_examples = min(10000, len(clean_terminology_sentences))
    if num_fine_tune_examples > 1:
        for i in range(num_fine_tune_examples):
            positive_sentence = clean_terminology_sentences[(i + 1) % len(clean_terminology_sentences)]
            train_examples.append(InputExample(texts=[clean_terminology_sentences[i], positive_sentence]))
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1)
        print(f"Fine-tuning complete. Saving model to '{fine_tuned_model_path}'...")
        model.save(fine_tuned_model_path)
    else:
        print("Not enough training examples to perform fine-tuning. Skipping.")
else:
    print("No valid terminology sentences found. Skipping fine-tuning.")

