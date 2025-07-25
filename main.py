# -*- coding: utf-8 -*-

import os
import pandas as pd
import config
from llm_provider import LLMProvider
from feature_generator import FeatureGenerator
from tool_generator import ToolGenerator
from annotator import Annotator
from feature_selector import FeatureSelector
from memory_manager import MemoryManager

def main():
    """
    Main function, entry point of the project.
    """
    # Check if input files exist
    if not os.path.exists(config.SCENE_DESCRIPTION_FILE):
        print(f"Error: Scene description file not found at: {config.SCENE_DESCRIPTION_FILE}")
        return
    if not os.path.exists(config.DATA_FILE):
        print(f"Error: Data file not found at: {config.DATA_FILE}")
        return

    # Load inputs
    try:
        with open(config.SCENE_DESCRIPTION_FILE, 'r', encoding='utf-8') as f:
            scene_description = f.read()
        
        df = pd.read_csv(config.DATA_FILE)
        
        # Validate CSV format
        if 'text' not in df.columns or 'score' not in df.columns:
            print("Error: The CSV file must contain 'text' and 'score' columns.")
            return
            
    except Exception as e:
        print(f"Error loading input files: {e}")
        return

    print("="*50)
    print(" AutoQual: Automated Feature Discovery Agent")
    print("="*50)
    print(f"Current Mode: {config.EXECUTION_MODE}")
    print(f"Scene Description File: {config.SCENE_DESCRIPTION_FILE}")
    print(f"Data File: {config.DATA_FILE}")
    print("="*50)

    try:
        # Initialize core modules
        llm_provider = LLMProvider()
        memory_manager = MemoryManager(llm_provider)
        feature_generator = FeatureGenerator(llm_provider, memory_manager)

        # Part 1: Generate initial feature pool
        final_features_str = feature_generator.generate_initial_features(scene_description, df)

        if not final_features_str or final_features_str.startswith("Error"):
            print("\nCritical error in feature generation. Aborting.")
            return

        # Part 2: Generate annotation tools
        tool_generator = ToolGenerator(llm_provider, df)
        tool_generator.generate_all_tools(final_features_str, feature_generator.overwrite_files)
        
        # Part 3: Annotate features
        print("\n--- STAGE 6/6: ANNOTATING FEATURES WITH ALL TOOLS ---")
        annotator = Annotator(llm_provider, df)
        annotated_df = annotator.annotate_features()

        if annotated_df.empty:
            print("\nAnnotation resulted in an empty dataframe. Aborting feature selection.")
            return

        # Part 4: Reflective Feature Selection
        print("\n--- STAGE 7/7: SELECTING BEST FEATURES WITH REFLECTION ---")
        selector = FeatureSelector(config.ANNOTATED_DATA_FILE, llm_provider, scene_description)
        best_feature_set = selector.select_features()

        # Part 5: Cross-Task Memory Summarization
        if best_feature_set:
            print("\n--- STAGE 8/8: SUMMARIZING FINDINGS FOR CROSS-TASK MEMORY ---")
            task_name = os.path.basename(config.DATA_DIR)
            memory_manager.summarize_and_save_experience(
                task_name=task_name,
                scene_description=scene_description,
                best_features=best_feature_set
            )

        print("\n" + "="*50)
        print("      AutoQual process completed successfully!")
        print("="*50)
        print("Final candidate features are in 'output/06_integrated_features.txt'")
        print("Generated annotation tools are in the 'tools/' directory.")
        print(f"Final annotated data is in '{config.ANNOTATED_DATA_FILE}'")
        print(f"Best feature set is in '{config.BEST_FEATURES_FILE}'")
        print(f"Cross-task memory updated at '{config.CROSS_TASK_MEMORY_FILE}'")
        print("="*50)

    except ValueError as ve:
        print(f"\nConfiguration Error: {ve}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during execution: {e}")


if __name__ == "__main__":
    main() 