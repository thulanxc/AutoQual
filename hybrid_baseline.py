# -*- coding: utf-8 -*-

"""
This script implements a hybrid baseline that combines effective features selected by AutoQual
with features from an embedding model. To ensure absolute comparability of results, this script
precisely replicates the data preparation pipeline of each individual baseline script
(embedding_baseline.py, feature_selector.py) for the respective models.
"""

import os
import config

# --- Device Setup (consistent across all baselines) ---
target_device = config.FINETUNE_DEVICE
if 'cuda' in target_device:
    gpu_id = '0'
    if ':' in target_device:
        gpu_id = target_device.split(':')[-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    target_device = 'cuda'
    print(f"Running on GPU: '{config.FINETUNE_DEVICE}'.")
else:
    print(f"Running on CPU.")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# --- 1. Replicate data preparation from embedding_baseline.py ---
def prepare_embedding_baseline_data():
    """Exactly duplicates the logic from embedding_baseline.py."""
    print("\n--- Preparing Data (Embedding Baseline Method) ---")
    if not os.path.exists(config.DATA_FILE):
        raise FileNotFoundError(f"Original data file not found: '{config.DATA_FILE}'")
    
    full_df = pd.read_csv(config.DATA_FILE).dropna(subset=['text', 'score'])
    train_df, test_df = train_test_split(
        full_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    X_train_text = train_df['text'].tolist()
    X_test_text = test_df['text'].tolist()
    y_train_raw = train_df['score'].values
    y_test_raw = test_df['score'].values
    
    score_scaler = MinMaxScaler()
    y_train = score_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test = score_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    print(f"Data prepared: {len(X_train_text)} train samples, {len(X_test_text)} test samples.")
    return X_train_text, y_train, X_test_text, y_test

# --- 2. Replicate data preparation from feature_selector.py ---
def prepare_autoqual_baseline_data():
    """Exactly duplicates the logic from feature_selector.py."""
    print("\n--- Preparing Data (AutoQual Baseline Method) ---")
    if not os.path.exists(config.ANNOTATED_DATA_FILE):
        raise FileNotFoundError(f"Annotated data file not found: '{config.ANNOTATED_DATA_FILE}'")
    if not os.path.exists(config.BEST_FEATURES_FILE):
        raise FileNotFoundError(f"Best features file not found: '{config.BEST_FEATURES_FILE}'")

    with open(config.BEST_FEATURES_FILE, 'r', encoding='utf-8') as f:
        best_features = [line.strip() for line in f if line.strip()]

    full_df = pd.read_csv(config.ANNOTATED_DATA_FILE)
    
    # Clean: remove NaNs on all required columns
    initial_rows = len(full_df)
    clean_df = full_df.dropna(subset=['text', 'score'] + best_features).copy()
    cleaned_rows = len(clean_df)
    if initial_rows > cleaned_rows:
        print(f"Data cleaning: Removed {initial_rows - cleaned_rows} rows with null values.")

    train_df, test_df = train_test_split(
        clean_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"Data prepared: {len(train_df)} train samples, {len(test_df)} test samples.")

    # Extract features and scores
    X_train_autoqual_raw = train_df[best_features]
    X_test_autoqual_raw = test_df[best_features]
    y_train_raw = train_df['score'].values
    y_test_raw = test_df['score'].values
    
    # Normalization (consistent with feature_selector.py)
    feature_scaler = MinMaxScaler()
    X_train_autoqual = feature_scaler.fit_transform(X_train_autoqual_raw)
    X_test_autoqual = feature_scaler.transform(X_test_autoqual_raw)

    score_scaler = MinMaxScaler()
    y_train = score_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test = score_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    # Also return text for the hybrid model
    X_train_text = train_df['text'].tolist()
    X_test_text = test_df['text'].tolist()

    return X_train_autoqual, y_train, X_test_autoqual, y_test, X_train_text, X_test_text

def generate_embeddings(X_train_text, X_test_text):
    """Generates embeddings for given lists of text."""
    print(f"Loading sentence-transformer model to device: '{target_device}'...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=target_device)
    
    print(f"Generating embeddings for {len(X_train_text)} train samples...")
    X_train_embeddings = model.encode(X_train_text, normalize_embeddings=True, show_progress_bar=True)
    
    print(f"Generating embeddings for {len(X_test_text)} test samples...")
    X_test_embeddings = model.encode(X_test_text, normalize_embeddings=True, show_progress_bar=True)
    return X_train_embeddings, X_test_embeddings

def evaluate_model(X_train, X_test, y_train, y_test, model_name):
    """Trains and evaluates a Linear Regression model."""
    print(f"Training {model_name} Linear Regression model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    predictions = lr_model.predict(X_test)
    rho, _ = spearmanr(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return rho, mae

def main():
    print("--- Running Hybrid Baseline Model (AutoQual + Embedding) ---")
    
    try:
        # --- Part 1: Exactly replicate the Embedding Baseline ---
        print("\n" + "="*60)
        print("Evaluating standalone Embedding features (aligned with embedding_baseline.py)")
        print("="*60)
        X_train_text_eb, y_train_eb, X_test_text_eb, y_test_eb = prepare_embedding_baseline_data()
        X_train_emb, X_test_emb = generate_embeddings(X_train_text_eb, X_test_text_eb)
        rho_embedding, mae_embedding = evaluate_model(X_train_emb, X_test_emb, y_train_eb, y_test_eb, "Embedding Baseline")
        print(f"Embedding Feature Performance: Rho={rho_embedding:.4f}, MAE={mae_embedding:.4f}")

        # --- Part 2: Exactly replicate the AutoQual Baseline ---
        print("\n" + "="*60)
        print("Evaluating standalone AutoQual features (aligned with main.py -> feature_selector.py)")
        print("="*60)
        (X_train_aq, y_train_aq, X_test_aq, y_test_aq, 
         X_train_text_hybrid, X_test_text_hybrid) = prepare_autoqual_baseline_data()
        rho_autoqual, mae_autoqual = evaluate_model(X_train_aq, X_test_aq, y_train_aq, y_test_aq, "AutoQual Baseline")
        print(f"AutoQual Feature Performance: Rho={rho_autoqual:.4f}, MAE={mae_autoqual:.4f}")
        
        # --- Part 3: Hybrid Model (on the aligned data subset from AutoQual) ---
        print("\n" + "="*60)
        print("Evaluating combined features (AutoQual + Embedding)")
        print("="*60)
        # Generate corresponding embeddings for the hybrid model
        X_train_emb_hybrid, X_test_emb_hybrid = generate_embeddings(X_train_text_hybrid, X_test_text_hybrid)
        
        # Concatenate features
        X_train_combined = np.concatenate([X_train_aq, X_train_emb_hybrid], axis=1)
        X_test_combined = np.concatenate([X_test_aq, X_test_emb_hybrid], axis=1)
        print(f"Combined feature shape: Train {X_train_combined.shape}, Test {X_test_combined.shape}")
        
        # Evaluate using the same y as the AutoQual model
        rho_combined, mae_combined = evaluate_model(X_train_combined, X_test_combined, y_train_aq, y_test_aq, "Hybrid Model")
        print(f"Hybrid Feature Performance: Rho={rho_combined:.4f}, MAE={mae_combined:.4f}")

        # --- Part 4: Summary ---
        print("\n" + "="*60)
        print("         Model Performance Summary")
        print("="*60)
        print(f"{'Model':<25} {'Spearman Rho':<15} {'MAE':<10}")
        print("-"*60)
        print(f"{'AutoQual Baseline':<25} {rho_autoqual:<15.4f} {mae_autoqual:<10.4f}")
        print(f"{'Embedding Baseline':<25} {rho_embedding:<15.4f} {mae_embedding:<10.4f}")
        print(f"{'Hybrid Model':<25} {rho_combined:<15.4f} {mae_combined:<10.4f}")
        print("="*60)

        improvement_vs_autoqual = ((rho_combined - rho_autoqual) / rho_autoqual) * 100 if rho_autoqual != 0 else float('inf')
        improvement_vs_embedding = ((rho_combined - rho_embedding) / rho_embedding) * 100 if rho_embedding != 0 else float('inf')
        
        print(f"\nHybrid Model vs AutoQual: Spearman's Rho improved by {improvement_vs_autoqual:.2f}%")
        print(f"Hybrid Model vs Embedding: Spearman's Rho improved by {improvement_vs_embedding:.2f}%")
        
    except FileNotFoundError as e:
        print(f"\nError: A required file was not found. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")

if __name__ == "__main__":
    main() 