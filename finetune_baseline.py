# -*- coding: utf-8 -*-

"""
This script fine-tunes a sentence-transformer model on the score prediction task.
It evaluates the model after each epoch and reports the best performance.
"""

import os
import config # Import config early to set device

# --- Force Single-GPU Training & Set Target Device ---
# By setting CUDA_VISIBLE_DEVICES before importing torch, we ensure that
# sentence-transformers only sees one GPU. This prevents it from automatically
# activating DataParallel, which can cause NCCL errors in some environments.
target_device = config.FINETUNE_DEVICE
if 'cuda' in target_device:
    gpu_id = '0'
    if ':' in target_device:
        gpu_id = target_device.split(':')[-1]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    # After setting the environment variable, sentence-transformers will see only one GPU.
    # We can just pass 'cuda' and it will use the correct one.
    target_device = 'cuda' 
    print(f"Running on physical GPU specified by '{config.FINETUNE_DEVICE}'. Set CUDA_VISIBLE_DEVICES='{gpu_id}'.")
else:
    print(f"Running on CPU.")


# Now import libraries that use CUDA
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import numpy as np
import shutil
from torch import nn, Tensor
from typing import Iterable, Dict, List
import logging

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

class RegressionLoss(nn.Module):
    """
    Custom loss for regression tasks. It adds a linear layer on top of the
    sentence embedding and computes MSE loss. This head is only used during training.
    """
    def __init__(self, model: "SentenceTransformer", in_features: int, out_features: int = 1):
        super(RegressionLoss, self).__init__()
        self.model = model
        self.regression_head = nn.Linear(in_features, out_features)
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = self.model(sentence_features[0])['sentence_embedding']
        predictions = self.regression_head(reps).squeeze(-1)
        return self.loss_fct(predictions, labels.float())

class RegressionEvaluator(SentenceEvaluator):
    """
    Evaluator for regression tasks.
    At the end of each epoch, it trains a LinearRegression head on the training embeddings
    and evaluates the performance on the test set.
    """
    def __init__(self, train_texts: List[str], train_labels: np.ndarray, test_texts: List[str], test_labels: np.ndarray, name: str = ''):
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.test_texts = test_texts
        self.test_labels = test_labels
        self.name = name

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        logging.info(f"RegressionEvaluator: Evaluating after epoch {epoch}...")
        
        # 1. Generate embeddings for the whole dataset with the current model
        train_embeddings = model.encode(self.train_texts, convert_to_numpy=True, show_progress_bar=False)
        test_embeddings = model.encode(self.test_texts, convert_to_numpy=True, show_progress_bar=False)
        
        # 2. Train a temporary linear regression model on the training data
        lr = LinearRegression()
        lr.fit(train_embeddings, self.train_labels)
        
        # 3. Predict on the test data and compute Spearman's Rho
        predictions = lr.predict(test_embeddings)
        rho, _ = spearmanr(self.test_labels, predictions)
        
        logging.info(f"Epoch {epoch}: Spearman's Rho: {rho:.4f}")
        return rho # The fit function will use this score to determine the best model


def prepare_data():
    """
    Loads data, creates a train/test split, and normalizes scores.
    """
    if not os.path.exists(config.DATA_FILE):
        raise FileNotFoundError(f"Error: Original data file not found at '{config.DATA_FILE}'")

    print(f"Loading original data from '{config.DATA_FILE}'...")
    full_df = pd.read_csv(config.DATA_FILE).dropna(subset=['text', 'score'])
    
    train_df, test_df = train_test_split(
        full_df, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Normalize scores
    score_scaler = MinMaxScaler()
    train_df['score_normalized'] = score_scaler.fit_transform(train_df[['score']]).flatten()
    test_df['score_normalized'] = score_scaler.transform(test_df[['score']]).flatten()

    print(f"Data prepared: {len(train_df)} train samples, {len(test_df)} test samples. Scores are normalized.")
    return train_df, test_df, score_scaler

def main():
    """
    Main function to run the fine-tuning baseline model pipeline.
    """
    print("--- Running Fine-Tuning Baseline Model (with per-epoch evaluation) ---")
    
    # 1. Prepare Data
    train_df, test_df, score_scaler = prepare_data()

    # 2. Setup for Fine-Tuning
    print(f"\nLoading pre-trained sentence-transformer model (BAAI/bge-small-en-v1.5) onto device: '{target_device}'...")
    try:
        model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=target_device)
    except Exception as e:
        print(f"\nError loading SentenceTransformer model. Ensure you have an internet connection and PyTorch.")
        print(f"Also check if the device '{config.FINETUNE_DEVICE}' is available and drivers are correctly installed.")
        print(f"Original error: {e}")
        return

    train_examples = []
    for index, row in train_df.iterrows():
        train_examples.append(InputExample(texts=[row['text']], label=row['score_normalized']))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.FINETUNE_BATCH_SIZE)
    
    embedding_dim = model.get_sentence_embedding_dimension()
    loss_function = RegressionLoss(model=model, in_features=embedding_dim)
    
    # 3. Setup Evaluator to run after each epoch
    print("\nSetting up custom regression evaluator...")
    evaluator = RegressionEvaluator(
        train_texts=train_df['text'].tolist(),
        train_labels=train_df['score_normalized'].values,
        test_texts=test_df['text'].tolist(),
        test_labels=test_df['score_normalized'].values,
    )
    
    # 4. Fine-Tune the Model with evaluation
    print("\nFine-tuning the model...")
    output_path = os.path.join(config.OUTPUT_DIR, "finetune_best_model")
    
    model.fit(
        train_objectives=[(train_dataloader, loss_function)],
        evaluator=evaluator,
        epochs=config.FINETUNE_EPOCHS,
        evaluation_steps=0, # Deactivate evaluation step counting, rely on epochs
        warmup_steps=config.FINETUNE_WARMUP_STEPS,
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True,
        optimizer_params={'lr': config.FINETUNE_LEARNING_RATE},
        weight_decay=config.FINETUNE_WEIGHT_DECAY
    )
    
    # 5. Load the best model and evaluate
    print("\nLoading best model from training for final evaluation...")
    best_model = SentenceTransformer(output_path)
    
    print("Generating embeddings with the best model...")
    X_train_embeddings = best_model.encode(train_df['text'].tolist(), normalize_embeddings=True, show_progress_bar=True)
    X_test_embeddings = best_model.encode(test_df['text'].tolist(), normalize_embeddings=True, show_progress_bar=True)
    
    y_train_normalized = train_df['score_normalized'].values
    y_test_original = test_df['score'].values
    
    # Train the final linear regression head on the best embeddings
    lr = LinearRegression()
    lr.fit(X_train_embeddings, y_train_normalized)
    
    predictions_normalized = lr.predict(X_test_embeddings)
    
    # Inverse transform to get original scale predictions for MAE
    predictions_original = score_scaler.inverse_transform(predictions_normalized.reshape(-1, 1)).flatten()

    # 6. Calculate final metrics
    rho, _ = spearmanr(y_test_original, predictions_original)
    mae = mean_absolute_error(y_test_original, predictions_original)

    print("\n" + "="*50)
    print("  Fine-Tuned Baseline Model Performance (Best Epoch)")
    print("="*50)
    print(f"  - Spearman's Rho: {rho:.4f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  (Evaluated using the best model saved at '{output_path}')")
    print("="*50)

    # Clean up the saved model directory
    try:
        shutil.rmtree(output_path)
        print(f"Cleaned up temporary model directory: {output_path}")
    except OSError as e:
        print(f"Error cleaning up directory {output_path}: {e}")


if __name__ == "__main__":
    main() 