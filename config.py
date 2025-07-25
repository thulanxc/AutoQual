# -*- coding: utf-8 -*-

"""
Project configuration file.
"""

import os
import glob

# --- High-Performance Model (for generation, reflection) ---
HP_API_KEY = "sk-d5dd39c2f59c486d99f7aca2791d28a3"
HP_BASE_URL = "https://api.deepseek.com"
HP_MODEL_NAME = "deepseek-reasoner"

# --- General Model (for annotation, simple tasks) ---
GP_API_KEY = "sk-cf850a19c4374fbeb72ef098034f94f3"
GP_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
GP_MODEL_NAME = "qwen-plus-latest"

# --- Annotation Settings ---
# Number of parallel workers for prompt-based annotation
ANNOTATION_MAX_WORKERS = 50 
# Save a checkpoint file after every N rows during prompt annotation
ANNOTATION_CHECKPOINT_INTERVAL = 100

# Execution Mode: 'auto' (fully automated) or 'manual' (user intervention)
# In 'manual' mode, the process will pause after each major LLM call
# to await user confirmation and allow for modifications.
EXECUTION_MODE = 'manual'

# Initial Feature Pool Generation Parameters
# 1. Role-based Feature Generation
ROLE_COUNT = 3  # Number of roles to generate

# 2. Data-based Feature Generation
SAMPLE_COUNT = 20 # Number of high/low score samples to analyze

# 3. Configurable generation counts for prompts
#   - Number of features generated per role
FEATURE_COUNT_PER_ROLE = 3
#   - Number of features generated from high-score samples
FEATURE_COUNT_POSITIVE = 3
#   - Number of features generated from low-score samples
FEATURE_COUNT_NEGATIVE = 3
#   - Number of features generated from contrastive analysis
FEATURE_COUNT_CONTRASTIVE = 10

# --- Scene Configuration ---
# Set the scene name. All data and output paths will be derived from this.
# To switch scenes, just modify SCENE_NAME here.
# SCENE_NAME = "Amazon_cellphones" # Example of another scene
SCENE_NAME = "Amazon_office" 

# --- Base Directories ---
DATA_DIR_BASE = "data"
OUTPUT_DIR_BASE = "output"

# --- Derived Paths (Do not edit below) ---
SCENE_DATA_DIR = os.path.join(DATA_DIR_BASE, SCENE_NAME)
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, SCENE_NAME)

# --- Input Files ---
SCENE_DESCRIPTION_FILE = os.path.join(SCENE_DATA_DIR, "scene_description.txt")

# Automatically find the data CSV file in the scene directory
csv_files = glob.glob(os.path.join(SCENE_DATA_DIR, "*.csv"))
if not csv_files:
    # If no CSV file is found, scripts using DATA_FILE will raise a clear error
    DATA_FILE = f"Error: No CSV file found in the scene directory '{SCENE_DATA_DIR}'"
elif len(csv_files) > 1:
    print(f"Warning: Found multiple CSV files in {SCENE_DATA_DIR}. Using the first one: {csv_files[0]}")
    DATA_FILE = csv_files[0]
else:
    DATA_FILE = csv_files[0]

# Tool Paths
TOOLS_DIR = os.path.join(OUTPUT_DIR, "tools")
CODE_TOOLS_DIR = os.path.join(TOOLS_DIR, "code")
PROMPT_TOOLS_DIR = os.path.join(TOOLS_DIR, "prompts")

# --- Feature Selection Settings ---
EVALUATION_METHOD = "linear_regression" # Options: "mutual_information", "linear_regression", "xgboost"
BEAM_WIDTH = 10
MAX_FEATURES = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Fine-tuning Baseline Settings ---
FINETUNE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
FINETUNE_EPOCHS = 4
FINETUNE_LEARNING_RATE = 2e-5
FINETUNE_WARMUP_STEPS = 100
FINETUNE_WEIGHT_DECAY = 0.01
FINETUNE_BATCH_SIZE = 16
FINETUNE_DEVICE = "cuda:0"  # "cuda:0", "cpu", etc.

# Bag-of-Words Baseline Settings
BOW_MAX_FEATURES = 500

# LLM Baselines
EVAL_SAMPLE_SIZE = 500 # Number of samples to use for LLM baseline evaluations
FEW_SHOT_EXAMPLES = 10 # Number of examples for the few-shot baseline

# Output Files
TRAIN_DATA_FILE = os.path.join(OUTPUT_DIR, "train_data.csv")
TEST_DATA_FILE = os.path.join(OUTPUT_DIR, "test_data.csv")
CODE_ANNOTATED_FILE = os.path.join(OUTPUT_DIR, "intermediate_code_annotated.csv")
ANNOTATED_DATA_FILE = os.path.join(OUTPUT_DIR, "final_annotated_data.csv")
BEST_FEATURES_FILE = os.path.join(OUTPUT_DIR, "best_features.txt")

# Paths
DATA_DIR = "data" 