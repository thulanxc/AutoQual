# -*- coding: utf-8 -*-

"""
This script implements a Zero-Shot LLM baseline.
It uses a general-purpose LLM to directly score text quality on a scale of 1-10,
without any examples or fine-tuning. The results are then compared against the
ground truth scores.
"""

import os
import pandas as pd
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

import config
from llm_provider import LLMProvider

# --- Prompt for Zero-Shot Scoring ---
ZERO_SHOT_PROMPT = """
You are an expert review analyst. Your task is to evaluate the quality of the following product review on a scale from 1 to 10, where 1 represents a very low-quality, unhelpful review, and 10 represents a very high-quality, helpful review.

Carefully read the review below. Your output MUST be a single integer from 1 to 10 and nothing else. Do not provide any explanation or reasoning.

Review text:
---
{text}
---

Your quality score (1-10):
"""

def prepare_data():
    """
    Loads data and creates a train/test split.
    The 'train' split is not used in zero-shot, but we create it for consistency.
    """
    if not os.path.exists(config.DATA_FILE):
        print(f"Error: Original data file not found at '{config.DATA_FILE}'")
        return None, None

    print(f"Loading original data from '{config.DATA_FILE}'...")
    full_df = pd.read_csv(config.DATA_FILE).dropna(subset=['text', 'score'])
    
    _, test_df = train_test_split(
        full_df, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Sample a subset for evaluation if the test set is larger than the sample size
    if len(test_df) > config.EVAL_SAMPLE_SIZE:
        print(f"Sampling {config.EVAL_SAMPLE_SIZE} records from the test set for evaluation.")
        test_df = test_df.sample(n=config.EVAL_SAMPLE_SIZE, random_state=config.RANDOM_STATE)
    
    print(f"Data prepared: Using {len(test_df)} test samples for zero-shot evaluation.")
    return test_df

def get_llm_score(text, provider):
    """
    Gets a score for a single piece of text from the LLM.
    Includes robust parsing to extract the number.
    """
    prompt = ZERO_SHOT_PROMPT.format(text=text)
    try:
        # Use the provider's robust method which includes retries
        content = provider.call_general_model(prompt, temperature=0)
        
        # Use regex to find the first sequence of digits
        match = re.search(r'\d+', content)
        if match:
            return int(match.group(0))
        else:
            print(f"Warning: Could not parse score from LLM response: '{content}'")
            return None
    except Exception as e:
        # This will now only catch truly unexpected errors, as call_general_model handles retries.
        print(f"An unexpected error occurred while processing a text snippet: {e}")
        return None

def main():
    """
    Main function to run the Zero-Shot LLM baseline pipeline.
    """
    print("--- Running Zero-Shot LLM Baseline Model ---")

    # 1. Prepare Data
    test_df = prepare_data()
    if test_df is None:
        return

    # 2. Setup LLM Provider
    # We use the GP (General Purpose) model for this task
    try:
        provider = LLMProvider()
    except Exception as e:
        print(f"Error initializing LLMProvider: {e}")
        return

    # 3. Get LLM predictions for the test set
    texts_to_score = test_df['text'].tolist()
    results = []
    
    print(f"\nGetting Zero-Shot scores from LLM (max_workers={config.ANNOTATION_MAX_WORKERS})...")
    with ThreadPoolExecutor(max_workers=config.ANNOTATION_MAX_WORKERS) as executor:
        future_to_text = {executor.submit(get_llm_score, text, provider): text for text in texts_to_score}
        
        for future in tqdm(as_completed(future_to_text), total=len(texts_to_score), desc="Scoring texts"):
            results.append(future.result())

    # 4. Process and align results
    test_df['llm_score'] = results
    
    # Filter out cases where scoring failed
    eval_df = test_df.dropna(subset=['score', 'llm_score'])
    
    if len(eval_df) == 0:
        print("\nError: No valid scores were returned by the LLM. Cannot evaluate.")
        return

    y_true = eval_df['score'].values
    y_pred = eval_df['llm_score'].values

    # 5. Normalize both true scores and predicted scores to [0, 1] for fair comparison
    scaler = MinMaxScaler()
    y_true_normalized = scaler.fit_transform(y_true.reshape(-1, 1)).flatten()
    
    # The LLM scores are on a 1-10 scale, so we fit a new scaler for them
    pred_scaler = MinMaxScaler()
    y_pred_normalized = pred_scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()

    # 6. Evaluate model
    print("\nEvaluating model on the test set...")
    rho, _ = spearmanr(y_true_normalized, y_pred_normalized)
    mae = mean_absolute_error(y_true_normalized, y_pred_normalized)

    print("\n" + "="*50)
    print("  Zero-Shot LLM Baseline Performance")
    print("="*50)
    print(f"  - Spearman's Rho: {rho:.4f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  ({len(eval_df)}/{len(test_df)} samples successfully evaluated)")
    print("="*50)

if __name__ == "__main__":
    main() 