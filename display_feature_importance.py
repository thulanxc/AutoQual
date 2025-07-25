import os
import re
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import numpy as np
from collections import Counter

def tokenize(text: str) -> set:
    """
    A simple tokenizer that converts text to a set of lowercased words.
    e.g., "Review length, the length of the review string" -> {'review', 'length', 'the', 'of', 'string'}
    e.g., "review_length_the_length_of_the_review_string" -> {'review', 'length', 'the', 'of', 'string'}
    """
    text = text.lower()
    # Split by common delimiters: comma, space, underscore
    words = re.split(r'[,\s_]+', text)
    return set(filter(None, words))

def find_best_match(feature_id: str, full_names: list) -> str:
    """
    Finds the best matching full name for a given feature ID using word overlap.
    This is much more robust than simple string matching.
    """
    id_tokens = tokenize(feature_id)
    best_match = feature_id  # Default to the ID itself
    max_overlap = 0

    for name in full_names:
        name_tokens = tokenize(name)
        overlap = len(id_tokens.intersection(name_tokens))
        
        # If we find a better match (more overlapping words), update our best guess.
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = name
            
    return best_match

def create_automatic_name_mapper(scene_output_dir: str):
    """
    Creates a mapping function that dynamically finds the best full name for any given feature ID.
    """
    integrated_features_path = os.path.join(scene_output_dir, "06_integrated_features.txt")
    
    try:
        with open(integrated_features_path, 'r', encoding='utf-8') as f:
            full_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Warning: Could not find '{integrated_features_path}'. Cannot map to full feature names.")
        return None

    # Return a function (a "closure") that has access to the list of full names
    return lambda feature_id: find_best_match(feature_id, full_names)

def display_feature_importance(scene_name: str):
    """
    Calculates and displays the importance of features for a specific scene.
    """
    print("\n" + "="*80)
    print(f"--- Feature Importance Analysis for Scene: '{scene_name}' ---")
    
    scene_output_dir = os.path.join("output", scene_name)
    best_features_path = os.path.join(scene_output_dir, "best_features.txt")
    train_data_path = os.path.join(scene_output_dir, "train_data.csv")

    try:
        with open(best_features_path, 'r', encoding='utf-8') as f:
            best_feature_ids = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: 'best_features.txt' not found for this scene. Aborting.")
        return

    # This mapper function will find the full name for us
    name_mapper = create_automatic_name_mapper(scene_output_dir)
    if name_mapper is None:
        print("ERROR: Could not create feature name map. Aborting for this scene.")
        return

    try:
        required_cols = best_feature_ids + ['score']
        df = pd.read_csv(train_data_path, usecols=lambda c: c in required_cols)
    except Exception as e:
        print(f"ERROR: Failed to load or parse training data '{train_data_path}'. Reason: {e}. Aborting.")
        return

    df_clean = df.dropna(subset=required_cols)
    if len(df_clean) < 2:
        print("ERROR: Not enough valid data after cleaning. Aborting.")
        return

    X = df_clean[best_feature_ids]
    y = df_clean['score']

    mi_scores = mutual_info_regression(X, y, random_state=42)
    total_mi = np.sum(mi_scores)
    normalized_scores = (mi_scores / total_mi) * 100 if total_mi > 0 else np.zeros_like(mi_scores)

    results = []
    for i, feature_id in enumerate(best_feature_ids):
        # Here we use the new robust mapper to get the FULL NAME
        full_name = name_mapper(feature_id)
        results.append((full_name, normalized_scores[i]))

    results.sort(key=lambda item: item[1], reverse=True)

    print("\nFeature Importance (Normalized to 100%):")
    print("-----------------------------------------")
    for name, score in results:
        # We print the full name directly.
        print(f"-> {name}")
        print(f"   (Importance: {score:.2f}%)")
    print("="*80)

def main():
    """
    Main function to run the analysis for a list of scenes.
    """
    scenes_to_process = [
        "Amazon_cellphones_new",
        "Amazon_clothing_new",
        "Amazon_grocery",
        "Amazon_office"  # Corrected typo from user request
    ]
    
    print("Starting analysis for all configured scenes...")
    for scene in scenes_to_process:
        display_feature_importance(scene)

if __name__ == "__main__":
    main() 