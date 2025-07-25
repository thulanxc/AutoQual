# -*- coding: utf-8 -*-

"""
This script implements a Bag-of-Words (BoW) baseline using TF-IDF.
It vectorizes the text data and trains a linear regression model to predict
the quality score. This serves as a simple, traditional baseline.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import config

def prepare_data():
    """
    Loads data, creates a train/test split, and normalizes scores.
    """
    if not os.path.exists(config.DATA_FILE):
        print(f"Error: Original data file not found at '{config.DATA_FILE}'")
        return None

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
    return train_df, test_df

def main():
    """
    Main function to run the Bag-of-Words baseline model pipeline.
    """
    print("--- Running Bag-of-Words (TF-IDF) Baseline Model ---")

    # 1. Prepare Data
    train_df, test_df = prepare_data()
    if train_df is None:
        return

    # 2. Vectorize Text using TF-IDF
    print(f"\nVectorizing text using TF-IDF (max_features={config.BOW_MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(
        max_features=config.BOW_MAX_FEATURES,
        stop_words='english'
    )
    
    # Fit on training data and transform both train and test data
    X_train_bow = vectorizer.fit_transform(train_df['text'])
    X_test_bow = vectorizer.transform(test_df['text'])
    
    y_train = train_df['score_normalized'].values
    y_test = test_df['score_normalized'].values

    # 3. Train Linear Regression Model
    print("\nTraining Linear Regression model on TF-IDF features...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_bow, y_train)

    # 4. Evaluate Model
    print("\nEvaluating model on the test set...")
    predictions = lr_model.predict(X_test_bow)

    rho, _ = spearmanr(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("\n" + "="*50)
    print("  Bag-of-Words (TF-IDF) Baseline Performance")
    print("="*50)
    print(f"  - Spearman's Rho: {rho:.4f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print("="*50)


if __name__ == "__main__":
    main() 