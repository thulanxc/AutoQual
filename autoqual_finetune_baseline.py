# -*- coding: utf-8 -*-

"""
This script implements an advanced hybrid baseline model.
It combines the features selected by AutoQual with a trainable Sentence Transformer model
and fine-tunes the entire network end-to-end to predict review quality scores.
"""

import os
import config
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# --- Device and Model Settings ---
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

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# --- 1. Data Preparation ---

def prepare_data_for_finetuning():
    """
    Loads, cleans, and splits the data, preparing everything needed for end-to-end fine-tuning.
    This function's logic is aligned with the data preparation in feature_selector.py.
    """
    print("\n--- Step 1: Preparing Data ---")
    # Check for files
    if not os.path.exists(config.ANNOTATED_DATA_FILE) or not os.path.exists(config.BEST_FEATURES_FILE):
        raise FileNotFoundError("Error: 'final_annotated_data.csv' or 'best_features.txt' not found. Please run the main pipeline first.")

    # Load feature list and annotated data
    with open(config.BEST_FEATURES_FILE, 'r', encoding='utf-8') as f:
        best_features = [line.strip() for line in f if line.strip()]
    
    full_df = pd.read_csv(config.ANNOTATED_DATA_FILE)
    
    # Data cleaning
    clean_df = full_df.dropna(subset=['text', 'score'] + best_features).copy()
    print(f"After cleaning, {len(clean_df)} / {len(full_df)} rows are retained.")

    # Train/test split
    train_df, test_df = train_test_split(clean_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    print(f"Data split: {len(train_df)} train samples, {len(test_df)} test samples.")

    # Feature normalization
    feature_scaler = MinMaxScaler()
    train_df[best_features] = feature_scaler.fit_transform(train_df[best_features])
    test_df[best_features] = feature_scaler.transform(test_df[best_features])

    # Score normalization
    score_scaler = MinMaxScaler()
    train_df['score'] = score_scaler.fit_transform(train_df[['score']]).flatten()
    test_df['score'] = score_scaler.transform(test_df[['score']]).flatten()
    
    return train_df, test_df, best_features

# --- 2. PyTorch Dataset ---

class CombinedFeatureDataset(Dataset):
    def __init__(self, dataframe, features, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.texts = dataframe['text'].tolist()
        self.autoqual_features = dataframe[features].values.astype(np.float32)
        self.scores = dataframe['score'].values.astype(np.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'autoqual_feats': torch.tensor(self.autoqual_features[idx], dtype=torch.float),
            'targets': torch.tensor(self.scores[idx], dtype=torch.float)
        }

# --- 3. Custom Model ---

class AutoQualFinetuneModel(nn.Module):
    def __init__(self, transformer_model_name, num_autoqual_features):
        super(AutoQualFinetuneModel, self).__init__()
        # Body: Transformer model
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        transformer_output_dim = self.transformer.config.hidden_size
        
        # Head: A simple MLP
        self.head = nn.Sequential(
            nn.Linear(transformer_output_dim + num_autoqual_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def _mean_pooling(self, model_output, attention_mask):
        """Performs mean pooling to get sentence-level embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, ids, mask, autoqual_feats):
        # Get token-level embeddings from the transformer
        transformer_output = self.transformer(ids, attention_mask=mask)
        
        # Pool token-level embeddings to get a sentence-level embedding
        sentence_embedding = self._mean_pooling(transformer_output, mask)
        
        # Concatenate sentence embedding and AutoQual features
        combined_features = torch.cat([sentence_embedding, autoqual_feats], dim=1)
        
        # Get final prediction through the head
        output = self.head(combined_features)
        return output

# --- 4. Training and Evaluation ---

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Training"):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        autoqual_feats = data['autoqual_feats'].to(device)
        targets = data['targets'].to(device)

        outputs = model(ids, mask, autoqual_feats)
        loss = loss_fn(outputs.squeeze(), targets)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            autoqual_feats = data['autoqual_feats'].to(device)
            targets = data['targets'].to(device)

            outputs = model(ids, mask, autoqual_feats)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            
    rho, _ = spearmanr(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    return rho, mae

def main():
    """Main execution function"""
    try:
        # 1. Prepare data
        train_df, test_df, best_features = prepare_data_for_finetuning()
        
        # 2. Create DataLoaders
        print("\n--- Step 2: Creating PyTorch DataLoaders ---")
        tokenizer = AutoTokenizer.from_pretrained(config.FINETUNE_MODEL_NAME)
        
        train_dataset = CombinedFeatureDataset(train_df, best_features, tokenizer)
        test_dataset = CombinedFeatureDataset(test_df, best_features, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=False, num_workers=0)
        print("✅ DataLoaders created successfully.")
        
        # 3. Initialize model, loss function, and optimizer
        print("\n--- Step 3: Initializing Model and Optimizer ---")
        model = AutoQualFinetuneModel(config.FINETUNE_MODEL_NAME, num_autoqual_features=len(best_features))
        model.to(target_device)
        
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.FINETUNE_LEARNING_RATE)
        print("✅ Model initialized successfully.")

        # 4. Training loop
        print("\n--- Step 4: Starting End-to-End Fine-tuning ---")
        best_rho = -1
        for epoch in range(config.FINETUNE_EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{config.FINETUNE_EPOCHS} ---")
            avg_loss = train_epoch(model, train_loader, loss_fn, optimizer, target_device)
            print(f"Training complete. Average Loss: {avg_loss:.4f}")
            
            rho, mae = eval_model(model, test_loader, target_device)
            print(f"Evaluation results: Spearman's Rho = {rho:.4f}, MAE = {mae:.4f}")

            if rho > best_rho:
                best_rho = rho
                print("✨ New best Spearman's Rho!")

        print("\n" + "="*50)
        print("      AutoQual Trainable Hybrid Model Performance")
        print("="*50)
        print(f"  - Best Spearman's Rho: {best_rho:.4f}")
        print("="*50)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")


if __name__ == "__main__":
    main() 