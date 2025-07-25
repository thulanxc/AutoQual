# -*- coding: utf-8 -*-

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import prompts
from llm_provider import LLMProvider
from memory_manager import MemoryManager

def _wait_for_user_confirmation(stage_name, files_generated):
    """
    Waits for the user to press Enter to continue, but only in manual mode.
    """
    if config.EXECUTION_MODE == 'manual':
        print(f"\n--- PAUSED: Review Stage '{stage_name}' ---")
        print("Please review the following generated file(s):")
        for f in files_generated:
            print(f"  - {f}")
        input("Press Enter to continue to the next stage...")

class FeatureGenerator:
    """
    Handles the generation of the initial feature pool using multiple strategies,
    including leveraging cross-task memory.
    """
    def __init__(self, llm_provider: LLMProvider, memory_manager: MemoryManager):
        self.llm = llm_provider
        self.memory_manager = memory_manager
        self.overwrite_files = 'overwrite' in config.EXECUTION_MODE

    def generate_initial_features(self, scene_description: str, df: pd.DataFrame) -> str:
        """
        Orchestrates the initial feature generation process. It first tries to use
        cross-task memory and falls back to standard methods if no relevant memory is found.
        """
        print("\n--- STAGE 1/7: INITIAL FEATURE HYPOTHESIS GENERATION ---")

        # Attempt to generate features from cross-task memory first
        retrieved_memories = self.memory_manager.retrieve_relevant_memories(scene_description)
        if retrieved_memories:
            print("  Generating initial features informed by cross-task memory...")
            prompt = prompts.CROSS_TASK_INFORMED_HYPOTHESIS_PROMPT.format(
                scene_description=scene_description,
                retrieved_memories=retrieved_memories,
                feature_count=config.FEATURE_COUNT_PER_ROLE * config.ROLE_COUNT # Generate a comparable number of features
            )
            informed_features = self.llm.get_completion(prompt, "01_generated_roles.txt", self.overwrite_files) # Re-using a file name
            # Since this is a direct, informed generation, we can skip other steps
            # and proceed directly to integration.
            _wait_for_user_confirmation("Informed Feature Generation", ["01_generated_roles.txt"])
            return self._integrate_features(informed_features, "informed generation")

        # Fallback to standard generation methods if no memory was used
        print("  No relevant memories found or used. Proceeding with standard hypothesis generation.")
        role_features_str = self._generate_features_from_roles(scene_description)
        _wait_for_user_confirmation("Role-based Feature Generation", [
            os.path.join(config.OUTPUT_DIR, "01_generated_roles.txt"),
            os.path.join(config.OUTPUT_DIR, "02_role_1_features.txt") # etc.
        ])
        
        data_features_str = self._generate_features_from_data(scene_description, df)
        _wait_for_user_confirmation("Data-based Feature Generation", [
            os.path.join(config.OUTPUT_DIR, "03_data_positive_features.txt"),
            os.path.join(config.OUTPUT_DIR, "04_data_negative_features.txt"),
            os.path.join(config.OUTPUT_DIR, "05_data_contrastive_features.txt")
        ])

        print("\n--- STAGE 3/7: INTEGRATING ALL GENERATED FEATURES ---")
        combined_features = f"{role_features_str}\n{data_features_str}"
        final_features = self._integrate_features(combined_features, "role and data-based generation")
        _wait_for_user_confirmation("Feature Integration", [os.path.join(config.OUTPUT_DIR, "06_integrated_features.txt")])
        
        print("\n--- ✅ Initial Feature Generation Complete ---")
        return final_features

    def _generate_features_from_roles(self, scene_description: str) -> str:
        """
        Generates features by creating expert personas and asking each for ideas.
        """
        print("\n--- STAGE 1.1: Generating Features from Multiple Personas ---")
        
        # 1. Generate roles
        prompt = prompts.GENERATE_ROLES_PROMPT.format(
            scene_description=scene_description,
            role_count=config.ROLE_COUNT
        )
        roles_str = self.llm.get_completion(prompt, "01_generated_roles.txt", self.overwrite_files)
        roles = [r.strip() for r in roles_str.strip().split('\n') if r.strip()]

        # 2. Generate features for each role in parallel
        all_role_features = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._get_features_for_role, scene_description, role, i): role for i, role in enumerate(roles)}
            for future in as_completed(futures):
                try:
                    role_features = future.result()
                    if role_features:
                        all_role_features.append(role_features)
                except Exception as e:
                    print(f"Failed to get features for role '{futures[future]}': {e}")
        
        return "\n".join(all_role_features)

    def _get_features_for_role(self, scene_description: str, role_description: str, index: int) -> str:
        """Helper function to get features for a single role."""
        prompt = prompts.GENERATE_FEATURES_FROM_ROLE_PROMPT.format(
            scene_description=scene_description,
            role_description=role_description,
            feature_count_per_role=config.FEATURE_COUNT_PER_ROLE
        )
        filename = f"02_role_{index+1}_features.txt"
        return self.llm.get_completion(prompt, filename, self.overwrite_files)

    def _generate_features_from_data(self, scene_description: str, df: pd.DataFrame) -> str:
        """
        Generates features by performing contrastive analysis on high and low-quality samples.
        """
        print("\n--- STAGE 1.2: Generating Features from Contrastive Data Analysis ---")
        # Prepare samples
        df_sorted = df.sort_values(by='score', ascending=False)
        high_quality_samples = "\n".join([f"- {text}" for text in df_sorted.head(config.SAMPLE_COUNT)['text']])
        low_quality_samples = "\n".join([f"- {text}" for text in df_sorted.tail(config.SAMPLE_COUNT)['text']])
        
        tasks = {
            "positive": (prompts.CONTRASTIVE_ANALYSIS_PROMPT_POSITIVE, high_quality_samples, config.FEATURE_COUNT_POSITIVE, "03_data_positive_features.txt"),
            "negative": (prompts.CONTRASTIVE_ANALYSIS_PROMPT_NEGATIVE, low_quality_samples, config.FEATURE_COUNT_NEGATIVE, "04_data_negative_features.txt"),
            "contrastive": (prompts.CONTRASTIVE_ANALYSIS_PROMPT_CONTRASTIVE, f"High-Score Examples:\n{high_quality_samples}\n\nLow-Score Examples:\n{low_quality_samples}", config.FEATURE_COUNT_CONTRASTIVE, "05_data_contrastive_features.txt")
        }

        all_data_features = []
        with ThreadPoolExecutor() as executor:
            future_to_task = {executor.submit(self._get_features_from_analysis, scene_description, prompt_template, samples, count, filename): task for task, (prompt_template, samples, count, filename) in tasks.items()}
            for future in as_completed(future_to_task):
                try:
                    data_features = future.result()
                    if data_features:
                        all_data_features.append(data_features)
                except Exception as e:
                    print(f"Data analysis task '{future_to_task[future]}' failed: {e}")
        
        return "\n".join(all_data_features)

    def _get_features_from_analysis(self, scene_description: str, prompt_template: str, samples: str, count: int, filename: str) -> str:
        """Helper for data-based feature generation tasks."""
        prompt = prompt_template.format(
            scene_description=scene_description,
            samples=samples,
            feature_count=count
        )
        return self.llm.get_completion(prompt, filename, self.overwrite_files)

    def _integrate_features(self, feature_list: str, source_description: str) -> str:
        """
        Consolidates and de-duplicates a list of features from any source.
        """
        print(f"  Integrating features from {source_description}...")
        prompt = prompts.INTEGRATE_FEATURES_PROMPT.format(feature_list=feature_list)
        final_features = self.llm.get_completion(prompt, "06_integrated_features.txt", self.overwrite_files)
        return final_features 