
import os
import shutil
from google.colab import files, drive
import sys
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
import torch
import random
import glob

# --- Section 1: Setup and Model Management ---

# Path for the fine-tuned model directory
fine_tuned_model_path = 'fine-tuned-medical-model'
# Flag to indicate if the model needs to be fine-tuned
should_fine_tune = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Check for existing fine-tuned model
if os.path.exists(fine_tuned_model_path):
    print(f"Loading fine-tuned model from '{fine_tuned_model_path}'...")
    try:
        model = SentenceTransformer(fine_tuned_model_path, device=device)
        print("Fine-tuned model loaded successfully.")
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}. Reverting to base model.")
        should_fine_tune = True
elif os.path.exists(f'{fine_tuned_model_path}.zip'):
    print(f"Found zipped model file '{fine_tuned_model_path}.zip'. Unzipping...")
    try:
        shutil.unpack_archive(f'{fine_tuned_model_path}.zip', fine_tuned_model_path)
        model = SentenceTransformer(fine_tuned_model_path, device=device)
        print("Model unzipped and loaded successfully.")
    except Exception as e:
        print(f"Error unzipping or loading model: {e}. Reverting to base model.")
        should_fine_tune = True
else:
    print("Fine-tuned model not found. Initializing a new model and preparing for fine-tuning...")
    should_fine_tune = True

if should_fine_tune:
    # --- Section 2: Fine-Tuning Process ---
    print("Installing required libraries...")
    !pip install -q sentence-transformers pypdf datasets accelerate wandb
    os.environ["WANDB_DISABLED"] = "true"
    print("Installation complete.\n")

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
        
    if len(clean_terminology_sentences) >= 2:
        print("\nStarting the fine-tuning process...")
        train_examples = []
        num_fine_tune_examples = min(10000, len(clean_terminology_sentences))
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
        sys.exit()

# --- Section 3: Semantic Matching Process ---

print("\n--- Starting Semantic Matching ---")

cpt_file = os.path.join(data_dir, 'CPT Codes.csv')
icd10_file = os.path.join(data_dir, 'ICD-10.csv')
hcpcs_file = os.path.join(data_dir, 'HCPCS Codes.csv')
cpt_output_file = os.path.join(data_dir, 'CPT_Code_with_ICD-10_Matches.csv')
hcpcs_output_file = os.path.join(data_dir, 'HCPCS_Code_with_ICD-10_Matches.csv')

def process_and_match(input_file, output_file, checkpoint_file_prefix, name):
    try:
        print(f"\nLoading {name} file from: {input_file}")
        df_to_process = pd.read_csv(input_file, on_bad_lines='skip', encoding='latin1')
        print(f"Loading ICD-10 file from: {icd10_file}")
        icd10_df = pd.read_csv(icd10_file, on_bad_lines='skip', encoding='latin1')
    except FileNotFoundError as e:
        print(f"\nError: A required file was not found. Please check your file paths.")
        print(f"  - Missing file: {e.filename}")
        return
    
    print(f"\n{name} and ICD-10 data loaded successfully. Starting semantic matching.")
    
    icd10_descriptions = [str(d) for d in icd10_df['Description'].tolist() if pd.notna(d)]
    icd10_embeddings = model.encode(icd10_descriptions, convert_to_tensor=True, device=device)
    
    chunk_size = 2000
    processed_chunks = []
    
    # Check for existing processed chunks
    chunk_files = sorted(glob.glob(f'{checkpoint_file_prefix}*.csv'))
    start_row = 0
    if chunk_files:
        print(f"Found {len(chunk_files)} existing chunks. Combining and resuming...")
        processed_chunks = [pd.read_csv(f, on_bad_lines='skip', encoding='latin1') for f in chunk_files]
        final_df = pd.concat(processed_chunks, ignore_index=True)
        start_row = len(final_df)
        print(f"Resuming from row {start_row}.")
    else:
        final_df = pd.DataFrame(columns=list(df_to_process.columns) + ['Matching_ICD-10_Codes'])

    total_rows = len(df_to_process)
    
    for i in range(start_row, total_rows, chunk_size):
        chunk = df_to_process.iloc[i:i + chunk_size].copy()
        chunk['Matching_ICD-10_Codes'] = [[] for _ in range(len(chunk))]
        
        chunk_descriptions = [str(d) for d in chunk['Description'].tolist() if pd.notna(d)]
        chunk_embeddings = model.encode(chunk_descriptions, convert_to_tensor=True, device=device)
        
        print(f"Performing semantic similarity search for chunk starting at row {i}...")
        for j, cpt_desc in enumerate(chunk_descriptions):
            cosine_scores = util.cos_sim(chunk_embeddings[j], icd10_embeddings)[0]
            similarity_threshold = 0.6
            matches_indices = [idx for idx, score in enumerate(cosine_scores) if score > similarity_threshold]
            
            for match_idx in matches_indices:
                icd10_code = icd10_df.iloc[match_idx]['Code']
                chunk.iloc[j, chunk.columns.get_loc('Matching_ICD-10_Codes')].append(icd10_code)
        
        final_df = pd.concat([final_df, chunk], ignore_index=True)
        
        chunk_num = i // chunk_size + 1
        chunk_filename = f"{checkpoint_file_prefix}_Chunk_{chunk_num}.csv"
        chunk.to_csv(chunk_filename, index=False)
        print(f"  - Processed and saved chunk {chunk_num} to '{chunk_filename}'.")
        
    final_df.to_csv(output_file, index=False)
    print(f"\nSemantic matching for {name} complete. The updated file is saved as: {output_file}")
    print("You can download this file from your Colab file explorer on the left.")
    
    for f in chunk_files:
        os.remove(f)
    print("Removed temporary chunk files.")

process_and_match(cpt_file, cpt_output_file, 'CPT_Processed_Chunk', 'CPT Codes')
process_and_match(hcpcs_file, hcpcs_output_file, 'HCPCS_Processed_Chunk', 'HCPCS Codes')

