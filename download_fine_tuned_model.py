import os
import shutil
from google.colab import files
import sys

# --- Download Section ---
fine_tuned_model_path = 'fine-tuned-medical-model'
if os.path.exists(fine_tuned_model_path):
    # Try to download to local machine
    try:
        print(f"\nZipping and downloading model to your local machine...")
        shutil.make_archive('fine-tuned-medical-model', 'zip', fine_tuned_model_path)
        files.download('fine-tuned-medical-model.zip')
        print(f"\nModel successfully downloaded to your local machine. You can find it in your downloads folder.")
    except Exception as e:
        print(f"Warning: Could not download model to local machine. Error: {e}")
else:
    print(f"Error: Fine-tuned model directory '{fine_tuned_model_path}' not found.")
    print("Please run the fine-tuning script first to create the model.")

