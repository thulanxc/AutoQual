# -*- coding: utf-8 -*-

"""
This script trains a baseline model using sentence embeddings to predict the score.
It serves as a benchmark to compare against the AutoQual feature engineering pipeline.
"""

import os
import config # Import config early to set device

# --- Force Single-GPU & Set Target Device ---
# Re-using the same logic as the fine-tuning script to ensure consistent device handling.
target_device = config.FINETUNE_DEVICE
if 'cuda' in target_device:
    gpu_id = '0'
    if ':' in target_device:
        gpu_id = target_device.split(':')[-1]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    target_device = 'cuda'
    print(f"Running on physical GPU specified by '{config.FINETUNE_DEVICE}'. Set CUDA_VISIBLE_DEVICES='{gpu_id}'.")
else:
    print(f"Running on CPU.")

# Now import libraries that use CUDA
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import numpy as np

def prepare_data():
    """
    Loads the original data and creates the same train/test split as the main pipeline
    to ensure a fair comparison.
    """
    if not os.path.exists(config.DATA_FILE):
        print(f"Error: Original data file not found at '{config.DATA_FILE}'")
        return None, None, None, None

    print(f"Loading original data from '{config.DATA_FILE}'...")
    full_df = pd.read_csv(config.DATA_FILE).dropna(subset=['text', 'score'])
    
    train_df, test_df = train_test_split(
        full_df, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )

    X_train_text = train_df['text'].tolist()
    y_train_raw = train_df['score'].values
    X_test_text = test_df['text'].tolist()
    y_test_raw = test_df['score'].values
    
    # Normalize scores to make MAE comparable
    score_scaler = MinMaxScaler()
    y_train = score_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test = score_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    print(f"Data prepared: {len(X_train_text)} train samples, {len(X_test_text)} test samples. Scores are normalized.")
    return X_train_text, y_train, X_test_text, y_test

def main():
    """
    Main function to run the embedding baseline model pipeline.
    """
    print("--- Running Embedding Baseline Model ---")
    
    # 1. Prepare Data
    X_train_text, y_train, X_test_text, y_test = prepare_data()
    if X_train_text is None:
        return

    # 2. Generate Embeddings
    print(f"\nLoading sentence-transformer model (BAAI/bge-small-en-v1.5) onto device: '{target_device}'...")
    print("This may take a moment on the first run as the model is downloaded.")
    try:
        model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=target_device)
    except Exception as e:
        print(f"\nError loading SentenceTransformer model. Please ensure you have an internet connection.")
        print("You may also need to install PyTorch. See instructions at https://pytorch.org/")
        print(f"Also check if the device '{config.FINETUNE_DEVICE}' is available and drivers are correctly installed.")
        print(f"Original error: {e}")
        return
    
    print("\nGenerating embeddings for training data...")
    X_train_embeddings = model.encode(X_train_text, normalize_embeddings=True, show_progress_bar=True)
    
    print("\nGenerating embeddings for test data...")
    X_test_embeddings = model.encode(X_test_text, normalize_embeddings=True, show_progress_bar=True)

    # 3. Train Linear Regression Model
    print("\nTraining Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_embeddings, y_train)

    # 4. Evaluate Model
    print("\nEvaluating model on the test set...")
    predictions = lr_model.predict(X_test_embeddings)

    rho, _ = spearmanr(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("\n" + "="*50)
    print("      Baseline Model Performance")
    print("="*50)
    print(f"  - Spearman's Rho: {rho:.4f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print("="*50)

if __name__ == "__main__":
    main() 