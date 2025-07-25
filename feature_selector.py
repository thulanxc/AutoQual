# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import config
from llm_provider import LLMProvider
import prompts
from tool_generator import ToolGenerator
from annotator import Annotator
from sklearn.linear_model import LinearRegression

class FeatureSelector:
    """
    Selects the best feature set using a reflective beam search algorithm,
    driven by mutual information and guided by an LLM-based reflection step.
    """
    def __init__(self, annotated_data_path: str, llm_provider: LLMProvider, scene_description: str):
        self.annotated_data_path = annotated_data_path
        self.llm = llm_provider
        self.scene_description = scene_description
        self.df = pd.read_csv(self.annotated_data_path)
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names = self._prepare_data()
        
    def _prepare_data(self) -> tuple:
        """
        Loads, splits, and normalizes the data.
        """
        # Identify feature columns (everything except 'text' and 'score')
        feature_names = [col for col in self.df.columns if col not in ['text', 'score']]
        
        cleaned_df = self.df.dropna(subset=feature_names + ['score'])
        
        train_df, test_df = train_test_split(
            cleaned_df,
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE
        )
        
        X_train_raw = train_df[feature_names]
        y_train = train_df['score']
        X_test_raw = test_df[feature_names]
        y_test = test_df['score']

        # Normalize features and scores
        feature_scaler = MinMaxScaler()
        X_train = pd.DataFrame(feature_scaler.fit_transform(X_train_raw), columns=feature_names, index=X_train_raw.index)
        X_test = pd.DataFrame(feature_scaler.transform(X_test_raw), columns=feature_names, index=X_test_raw.index)

        score_scaler = MinMaxScaler()
        y_train = pd.Series(score_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten(), name='score', index=y_train.index)
        y_test = pd.Series(score_scaler.transform(y_test.values.reshape(-1, 1)).flatten(), name='score', index=y_test.index)

        print(f"Data prepared: {len(X_train)} train samples, {len(X_test)} test samples. Features and scores are normalized.")
        return X_train, X_test, y_train, y_test, feature_names

    def select_features(self) -> list:
        """
        Executes the reflective beam search to find the best feature set.
        """
        print("\n--- Starting Reflective Feature Selection using Beam Search ---")
        
        # --- Stage 0: Initialize Beam ---
        print("Evaluating single features with Mutual Information to initialize the beam...")
        initial_candidates = []
        for feature in tqdm(self.feature_names, desc="Initial MI Evaluation"):
            score = self._evaluate_mi([feature])
            initial_candidates.append(([feature], score))
        
        initial_candidates.sort(key=lambda x: x[1], reverse=True)
        beams = initial_candidates[:config.BEAM_WIDTH]
        print(f"Initial beam initialized with top {config.BEAM_WIDTH} features.")
        for features, score in beams:
            print(f"  - Feature: {features[0]}, Score: {score:.4f}")

        # --- Iterative Search with Reflection ---
        for i in tqdm(range(2, config.MAX_FEATURES + 1), desc="Reflective Beam Search"):
            # --- INTRA-TASK REFLECTION ---
            print(f"\nIteration {i}: Performing intra-task reflection...")
            best_beam_features, best_beam_score = beams[0]
            self._intra_task_reflection(best_beam_features, best_beam_score)

            # --- BEAM EXPANSION ---
            candidates = []
            for feature_set, _ in beams:
                remaining_features = [f for f in self.feature_names if f not in feature_set]
                for new_feature in remaining_features:
                    current_features = feature_set + [new_feature]
                    score = self._evaluate_mi(current_features)
                    candidates.append((current_features, score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:config.BEAM_WIDTH]
            
            if not beams:
                print("No more candidates found. Halting search.")
                break
            best_beam_features, best_beam_score = beams[0]
            print(f"Iteration {i}/{config.MAX_FEATURES} -> Best MI Score: {best_beam_score:.4f} with {len(best_beam_features)} features.")

        # --- Final Selection ---
        if not beams:
            print("Error: Beam search ended with no selected features.")
            return []
            
        best_feature_set, final_score = beams[0]
        
        print("\n--- ✅ Reflective Feature Selection Complete ---")
        print(f"Best Feature Set Found ({len(best_feature_set)} features):")
        for feature in best_feature_set:
            print(f"  - {feature}")
        print(f"\nFinal Score (Joint Mutual Information): {final_score:.4f}")
        
        with open(config.BEST_FEATURES_FILE, 'w', encoding='utf-8') as f:
            for feature in best_feature_set:
                f.write(f"{feature}\n")
        print(f"\nBest feature set saved to '{config.BEST_FEATURES_FILE}'")

        return best_feature_set

    def _intra_task_reflection(self, current_features: list, current_score: float):
        """
        Performs the reflection step: analyzes current state and generates new feature hypotheses.
        """
        candidate_features = [f for f in self.feature_names if f not in current_features]
        
        feature_performance = ""
        for feature in current_features:
            mi = self._evaluate_mi([feature])
            feature_performance += f"- {feature}: {mi:.4f}\n"

        prompt = prompts.INTRA_TASK_REFLECTION_PROMPT.format(
            scene_description=self.scene_description,
            iteration_step=len(current_features),
            k_features=config.MAX_FEATURES,
            current_features=", ".join(current_features),
            current_score=current_score,
            feature_performance=feature_performance,
            candidate_features=", ".join(candidate_features[:10]), # Show a subset
            num_new_features_to_generate=3 # Ask for 3 new ideas
        )

        print("  Asking LLM to reflect and generate new feature hypotheses...")
        new_feature_hypotheses_str = self.llm.get_completion(prompt, "temp_reflection.txt", overwrite=True)
        
        if not new_feature_hypotheses_str or new_feature_hypotheses_str.strip() == "":
            print("  LLM did not generate new hypotheses. Continuing search.")
            return

        print("  New hypotheses received. Implementing and annotating new features...")
        new_features = [f.strip() for f in new_feature_hypotheses_str.strip().split('\n') if f.strip()]

        # Implement tools for new features
        # We need a temporary dataframe to pass to the tool generator for sampling
        temp_df_for_tools = pd.read_csv(config.DATA_FILE)
        tool_generator = ToolGenerator(self.llm, temp_df_for_tools)
        tool_generator.generate_all_tools("\n".join(new_features), overwrite=True)
        
        # Annotate new features
        annotator = Annotator(self.llm, self.df)
        # Modify annotator to only process the new features
        newly_annotated_df = annotator.annotate_features(feature_list=new_features)

        if newly_annotated_df.empty:
            print("  Annotation of new features failed. Continuing search.")
            return

        # Update the main dataframe and feature sets for the ongoing search
        print("  Integrating new features into the current search...")
        for new_feature in new_features:
            if new_feature in newly_annotated_df.columns:
                self.df[new_feature] = newly_annotated_df[new_feature]
        
        # Re-run data preparation to get updated and normalized X, y, and feature_names
        self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names = self._prepare_data()
        print(f"  Feature pool expanded. Total features now: {len(self.feature_names)}")


    def _evaluate_mi(self, features: list) -> float:
        """
        Evaluates a feature set using mutual information.
        For multiple features, it computes the joint mutual information with the target.
        """
        if not features:
            return 0.0
        # scikit-learn's mutual_info_regression can handle multiple features in X
        # It estimates the MI between each feature in X and y and returns an array.
        # To get joint MI, we need a single value. A common approach is to use a model
        # or sum the individual MIs, but for beam search, we need a consistent metric.
        # The most direct interpretation of I(Y; F_S) is the MI of the joint distribution.
        # We'll use a practical estimation: the MI of a single feature that is the best
        # proxy for the set, which we can get from a simple regressor's prediction.
        
        # A simple and fast proxy for Joint mi
        lr = pd.DataFrame(self.X_train[features])
        if lr.shape[1] > 1:
            # For multiple features, create a single composite feature via regression
            model = LinearRegression()
            model.fit(self.X_train[features], self.y_train)
            composite_feature_train = model.predict(self.X_train[features]).reshape(-1, 1)
            return mutual_info_regression(composite_feature_train, self.y_train, random_state=config.RANDOM_STATE)[0]
        else:
            # For a single feature
            return mutual_info_regression(self.X_train[features], self.y_train, random_state=config.RANDOM_STATE)[0]

    def _evaluate_linear_regression(self, features: list):
        # This function is kept for potential future use but is not the primary method anymore.
        pass

    def _evaluate_xgboost(self, features: list):
        # This function is kept for potential future use but is not the primary method anymore.
        pass 