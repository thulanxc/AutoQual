# -*- coding: utf-8 -*-

"""
This script implements a Few-Shot LLM baseline.
For each text to be evaluated, it dynamically samples a number of examples
(e.g., 20) from the training set to provide in-context learning for the LLM.
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

# --- Prompt for Few-Shot Scoring ---
FEW_SHOT_PROMPT_TEMPLATE = """You are an expert review analyst. Your task is to evaluate the quality of a product review on a scale from 1 to 10, where 1 is low quality and 10 is high quality.

Here are {num_examples} examples of reviews and their corresponding quality scores:

--- EXAMPLES START ---
{examples}
--- EXAMPLES END ---

Now, using the same criteria demonstrated in the examples, evaluate the quality of the following review.
Your response MUST be a single integer from 1 to 10. Only output a single integer and nothing else. Do not add any extra text, explanation, or punctuation.

Review to evaluate:
---
{text}
---

Your quality score (1-10):
"""

def prepare_data():
    """
    Loads data, creates a train/test split, and samples the test set.
    """
    if not os.path.exists(config.DATA_FILE):
        print(f"Error: Original data file not found at '{config.DATA_FILE}'")
        return None, None

    print(f"Loading original data from '{config.DATA_FILE}'...")
    full_df = pd.read_csv(config.DATA_FILE).dropna(subset=['text', 'score'])
    
    train_df, test_df = train_test_split(
        full_df, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Sample a subset for evaluation
    if len(test_df) > config.EVAL_SAMPLE_SIZE:
        print(f"Sampling {config.EVAL_SAMPLE_SIZE} records from the test set for evaluation.")
        test_df = test_df.sample(n=config.EVAL_SAMPLE_SIZE, random_state=config.RANDOM_STATE)

    print(f"Data prepared: {len(train_df)} train samples, {len(test_df)} test samples for evaluation.")
    return train_df, test_df

def format_examples(df_examples):
    """Formats sampled examples for inclusion in the prompt."""
    example_strs = []
    # Use the 'score_1_to_10' column created in main()
    for _, row in df_examples.iterrows():
        example_strs.append(f"Review: \"{row['text']}\"\\nScore: {row['score_1_to_10']}")
    return "\\n\\n".join(example_strs)

def get_llm_score(text_to_score, train_df, provider):
    """
    Gets a score using a few-shot prompt with dynamically sampled examples.
    Includes robust non-regex parsing and defaults to 5 on failure.
    """
    examples_df = train_df.sample(n=config.FEW_SHOT_EXAMPLES)
    example_str = format_examples(examples_df)
    
    prompt = FEW_SHOT_PROMPT_TEMPLATE.format(
        num_examples=config.FEW_SHOT_EXAMPLES,
        examples=example_str,
        text=text_to_score
    )
    
    try:
        content = provider.call_general_model(prompt, temperature=0).strip()
        
        # Robust parsing without regex, defaulting to 5
        num_str = ""
        in_number = False
        for char in content:
            if char.isdigit():
                num_str += char
                in_number = True
            elif in_number:
                # Finished reading the first number in the string
                break
        
        if num_str:
            score = int(num_str)
            if 1 <= score <= 10:
                return score
            else:
                print(f"Warning: Parsed score '{score}' from '{content}' is out of range. Defaulting to 5.")
                return 5
        else:
            print(f"Warning: No digits found in LLM response: '{content}'. Defaulting to 5.")
            return 5
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Defaulting to 5.")
        return 5

def main():
    """
    Main function to run the Few-Shot LLM baseline pipeline.
    """
    print("--- Running Few-Shot LLM Baseline Model ---")

    # 1. Prepare Data
    train_df, test_df = prepare_data()
    if train_df is None or test_df is None:
        return

    # Create a 1-10 score for the training examples to be used in prompts
    score_scaler_1_10 = MinMaxScaler(feature_range=(1, 10))
    train_df['score_1_to_10'] = score_scaler_1_10.fit_transform(train_df[['score']]).flatten()
    train_df['score_1_to_10'] = train_df['score_1_to_10'].round().astype(int)

    # 2. Setup LLM Provider
    try:
        provider = LLMProvider()
    except Exception as e:
        print(f"Error initializing LLMProvider: {e}")
        return

    # 3. Get LLM predictions for the test set
    texts_to_score = test_df['text'].tolist()
    results = []
    
    print(f"\nGetting Few-Shot scores from LLM (max_workers={config.ANNOTATION_MAX_WORKERS})...")
    with ThreadPoolExecutor(max_workers=config.ANNOTATION_MAX_WORKERS) as executor:
        future_to_score = {
            executor.submit(get_llm_score, text, train_df, provider): text 
            for text in texts_to_score
        }
        
        for future in tqdm(as_completed(future_to_score), total=len(texts_to_score), desc="Scoring texts"):
            results.append(future.result())

    # 4. Process and align results
    test_df['llm_score'] = results
    eval_df = test_df.dropna(subset=['score', 'llm_score'])
    
    if len(eval_df) == 0:
        print("\nError: No valid scores were returned by the LLM. Cannot evaluate.")
        return

    y_true = eval_df['score'].values
    y_pred = eval_df['llm_score'].values

    # Normalize both true scores and predicted scores to [0, 1] for fair comparison
    scaler = MinMaxScaler()
    y_true_normalized = scaler.fit_transform(y_true.reshape(-1, 1)).flatten()
    
    pred_scaler = MinMaxScaler()
    y_pred_normalized = pred_scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()

    # Evaluate model
    rho, _ = spearmanr(y_true_normalized, y_pred_normalized)
    mae = mean_absolute_error(y_true_normalized, y_pred_normalized)

    print("\n" + "="*50)
    print("  Few-Shot LLM Baseline Performance")
    print("="*50)
    print(f"  - Spearman's Rho: {rho:.4f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  ({len(eval_df)}/{len(test_df)} samples successfully evaluated)")
    print("="*50)

if __name__ == "__main__":
    main() 