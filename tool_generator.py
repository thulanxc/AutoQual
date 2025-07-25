# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import prompts
from llm_provider import LLMProvider
from feature_generator import _wait_for_user_confirmation

def _slugify(text: str) -> str:
    """
    Converts a string into a clean, file-safe name.
    Example: "Feature: Readability Score (Flesch-Kincaid)" -> "feature_readability_score_flesch_kincaid"
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s-]', '', text)  # Remove non-alphanumeric characters
    text = re.sub(r'[\s-]+', '_', text.strip()) # Replace spaces and hyphens with underscores
    return text[:50] # Truncate to 50 chars

class ToolGenerator:
    """
    Responsible for generating and refining annotation tools (code or prompts) for a given list of features.
    Implements the propose-validate-refine cycle.
    """
    def __init__(self, llm_provider: LLMProvider, df: pd.DataFrame):
        self.llm = llm_provider
        self.df = df
        # Ensure tool directories exist
        if not os.path.exists(config.CODE_TOOLS_DIR):
            os.makedirs(config.CODE_TOOLS_DIR)
        if not os.path.exists(config.PROMPT_TOOLS_DIR):
            os.makedirs(config.PROMPT_TOOLS_DIR)

    def generate_all_tools(self, features_list_str: str, overwrite: bool):
        """
        Main method to orchestrate the entire tool generation process.
        """
        print("\n--- STAGE 4/5: TOOL ASSIGNMENT & GENERATION (WITH REFINE CYCLE) ---")
        features = [f.strip() for f in features_list_str.strip().split('\n') if f.strip()]
        
        # In auto mode, we handle assignment and generation together in the parallel loop
        if config.EXECUTION_MODE == 'auto':
            print("Auto mode: Determining tool types and generating with refine cycle in parallel...")
            generated_tools = self._generate_tools_parallelly_with_refine(features, overwrite)
            _wait_for_user_confirmation("Tool Generation", generated_tools)
        else: # Manual mode remains the same
            assignments = self._get_tool_assignments_manual(features, overwrite)
            assignment_filename = os.path.join(config.OUTPUT_DIR, "07_tool_type_assignments.csv")
            _wait_for_user_confirmation("Tool Type Assignment", [assignment_filename])

            assignments_df = pd.read_csv(assignment_filename)
            print("\nGenerating tools based on manual assignments...")
            generated_tools = self._generate_tools_parallelly_from_assignments(assignments_df, overwrite)
            _wait_for_user_confirmation("Tool Generation", generated_tools)
        
        print("\n--- ✅ Tool Generation Complete ---")

    def _get_tool_assignments_manual(self, features: list, overwrite: bool):
        """
        Handles the manual process of creating a CSV for the user to fill.
        """
        assignment_file = os.path.join(config.OUTPUT_DIR, "07_tool_type_assignments.csv")
        print(f"Manual mode: Please define tool types in '{assignment_file}'.")
        if not os.path.exists(assignment_file) or overwrite:
            pd.DataFrame({
                'feature': features, 
                'tool_type': ['' for _ in features]
            }).to_csv(assignment_file, index=False)
        return

    def _generate_tools_parallelly_from_assignments(self, assignments_df: pd.DataFrame, overwrite: bool):
        """Generates tools from a pre-filled assignments CSV (manual mode)."""
        generated_files = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._propose_validate_refine_cycle, row['feature'], row['tool_type'], overwrite) for _, row in assignments_df.iterrows()]
            for future in as_completed(futures):
                try:
                    result_path = future.result()
                    if result_path:
                        generated_files.append(result_path)
                except Exception as e:
                    print(f"A tool generation task failed: {e}")
        return generated_files

    def _generate_tools_parallelly_with_refine(self, features: list, overwrite: bool):
        """Determines tool type and runs the refine cycle for each feature in parallel."""
        generated_files = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._decide_and_generate_tool, feature, overwrite) for feature in features]
            for future in as_completed(futures):
                try:
                    result_path = future.result()
                    if result_path:
                        generated_files.append(result_path)
                except Exception as e:
                    print(f"A tool generation task failed for a feature: {e}")
        return generated_files

    def _decide_and_generate_tool(self, feature: str, overwrite: bool):
        """Decides the tool type and then initiates the generation cycle."""
        tool_type = self._decide_one_tool_type(feature)
        return self._propose_validate_refine_cycle(feature, tool_type, overwrite)

    def _decide_one_tool_type(self, feature_description: str) -> str:
        prompt = prompts.DECIDE_TOOL_TYPE_PROMPT.format(feature_description=feature_description)
        filename = f"temp_decision_{_slugify(feature_description)}.txt"
        try:
            tool_type = self.llm.get_completion(prompt, filename, overwrite=True).strip().upper()
            if tool_type not in ["CODE", "PROMPT"]:
                return "PROMPT" # Default
            return tool_type
        except Exception:
            return "PROMPT" # Default on error

    def _propose_validate_refine_cycle(self, feature_desc: str, tool_type: str, overwrite: bool, max_refinements: int = 2):
        """
        Manages the propose-validate-refine loop for a single feature.
        """
        print(f"\nStarting generation cycle for feature: '{feature_desc[:50]}...' (Type: {tool_type})")
        
        feature_slug = _slugify(feature_desc)
        tool_definition = ""
        feedback = ""

        for i in range(max_refinements + 1):
            # --- PROPOSE/REFINE ---
            if i == 0:
                print(f"  Attempt {i+1}: Proposing initial tool...")
                tool_definition = self._propose_tool(feature_slug, feature_desc, tool_type)
            else:
                print(f"  Attempt {i+1}: Refining tool based on feedback...")
                tool_definition = self._refine_tool(tool_definition, feature_desc, tool_type, feedback)

            if not tool_definition:
                print(f"  Failed to generate or refine tool for '{feature_desc}'. Aborting this feature.")
                return None

            # --- VALIDATE ---
            print(f"  Validating tool...")
            validation_result, feedback = self._validate_tool(tool_definition, feature_desc, tool_type)
            
            if validation_result == "ALIGNED":
                print(f"  Tool for '{feature_desc[:50]}...' validated successfully.")
                break
            else:
                print(f"  Validation failed. Feedback: {feedback}")
                if i == max_refinements:
                    print(f"  Max refinements reached. Using last version of the tool.")
                    break
        
        # --- SAVE FINAL TOOL ---
        return self._save_final_tool(feature_slug, tool_definition, tool_type)

    def _propose_tool(self, feature_slug: str, feature_desc: str, tool_type: str) -> str:
        """Generates the initial version of a tool."""
        if tool_type == "CODE":
            function_name = f"annotate_{feature_slug}"
            prompt = prompts.GENERATE_CODE_TOOL_PROMPT.format(function_name=function_name, feature_name=feature_slug, feature_description=feature_desc)
        else: # PROMPT
            prompt = prompts.GENERATE_PROMPT_TOOL_PROMPT.format(feature_description=feature_desc)
        
        filename = f"temp_{tool_type.lower()}_{feature_slug}.txt"
        return self.llm.get_completion(prompt, filename, overwrite=True)

    def _validate_tool(self, tool_definition: str, feature_desc: str, tool_type: str) -> (str, str):
        """Validates a tool against a sample and returns the verdict and feedback."""
        sample_text = self.df['text'].dropna().sample(1).iloc[0]
        
        # This is a simplification. A real implementation would need to execute the code/prompt.
        # For now, we'll simulate this by asking the LLM to review the tool and a sample.
        # A more robust version would use a sandboxed exec() for code or an actual LLM call for the prompt tool.
        
        # We'll just pass the definition and a sample to the validation prompt.
        # The prompt is designed to let the LLM infer the output.
        prompt = prompts.VALIDATE_TOOL_PROMPT.format(
            feature_description=feature_desc,
            tool_type=tool_type,
            tool_definition=tool_definition,
            sample_text=sample_text,
            tool_output="[LLM, please infer the likely output of the tool on the sample text and use it for your assessment]"
        )
        filename = f"temp_validation_{_slugify(feature_desc)}.txt"
        response = self.llm.get_completion(prompt, filename, overwrite=True).strip()

        if response.startswith("ALIGNED"):
            return "ALIGNED", ""
        elif response.startswith("NEEDS_REFINEMENT"):
            feedback = response.replace("NEEDS_REFINEMENT", "").strip()
            return "NEEDS_REFINEMENT", feedback
        else:
            # If the LLM gives an unexpected response, assume it needs refinement
            return "NEEDS_REFINEMENT", response

    def _refine_tool(self, original_tool: str, feature_desc: str, tool_type: str, feedback: str) -> str:
        """Generates an improved version of a tool based on feedback."""
        sample_text = self.df['text'].dropna().sample(1).iloc[0] # Use a consistent sample for refinement context
        prompt = prompts.REFINE_TOOL_PROMPT.format(
            tool_type=tool_type,
            feature_description=feature_desc,
            original_tool=original_tool,
            sample_text=sample_text,
            failure_explanation=feedback
        )
        filename = f"temp_refine_{tool_type.lower()}_{_slugify(feature_desc)}.txt"
        return self.llm.get_completion(prompt, filename, overwrite=True)

    def _save_final_tool(self, feature_slug: str, tool_definition: str, tool_type: str):
        """Saves the final, validated tool definition to the correct file."""
        if tool_type == "CODE":
            filepath = os.path.join(config.CODE_TOOLS_DIR, f"{feature_slug}.py")
            # Clean up potential markdown
            code_match = re.search(r'```python\n(.*?)```', tool_definition, re.DOTALL)
            content_to_save = code_match.group(1).strip() if code_match else tool_definition
        else: # PROMPT
            filepath = os.path.join(config.PROMPT_TOOLS_DIR, f"{feature_slug}.txt")
            content_to_save = tool_definition

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content_to_save)
        print(f"  Saved final tool: {filepath}")
        return filepath 