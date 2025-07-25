# -*- coding: utf-8 -*-

"""
项目配置文件
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

# 执行模式: 'auto' (全自动) 或 'manual' (手动干预)
# 'manual' 模式下，每次调用大模型后会暂停，等待用户确认和修改
EXECUTION_MODE = 'manual'

# 初始特征池生成参数
# 1. 基于角色的特征生成
ROLE_COUNT = 3  # 要生成的角色数量

# 2. 基于数据的特征生成
SAMPLE_COUNT = 20 # 用于分析的高分/低分样本数量

# 3. Prompt中可配置的生成数量
#   - 每个角色生成的特征数
FEATURE_COUNT_PER_ROLE = 3
#   - 从高分样本中生成的特征数
FEATURE_COUNT_POSITIVE = 3
#   - 从低分样本中生成的特征数
FEATURE_COUNT_NEGATIVE = 3
#   - 从对比分析中生成的特征数
FEATURE_COUNT_CONTRASTIVE = 10

# --- Scene Configuration ---
# 设置场景名称，所有数据和输出路径都将从此派生。
# 切换场景时，只需修改此处的 SCENE_NAME 即可。
# SCENE_NAME = "Amazon_cellphones" # 另一个场景示例
SCENE_NAME = "Amazon_office" 

# --- Base Directories ---
DATA_DIR_BASE = "data"
OUTPUT_DIR_BASE = "output"

# --- Derived Paths (Do not edit below) ---
SCENE_DATA_DIR = os.path.join(DATA_DIR_BASE, SCENE_NAME)
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, SCENE_NAME)

# --- Input Files ---
SCENE_DESCRIPTION_FILE = os.path.join(SCENE_DATA_DIR, "scene_description.txt")

# 自动在场景目录中查找数据CSV文件
csv_files = glob.glob(os.path.join(SCENE_DATA_DIR, "*.csv"))
if not csv_files:
    # 如果找不到CSV文件，后续使用DATA_FILE的脚本会报错，提示清晰
    DATA_FILE = f"错误：在场景目录 '{SCENE_DATA_DIR}' 中未找到CSV文件"
elif len(csv_files) > 1:
    print(f"警告：在 {SCENE_DATA_DIR} 中找到多个CSV文件。将使用第一个：{csv_files[0]}")
    DATA_FILE = csv_files[0]
else:
    DATA_FILE = csv_files[0]

# 工具路径
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

# 输出文件
TRAIN_DATA_FILE = os.path.join(OUTPUT_DIR, "train_data.csv")
TEST_DATA_FILE = os.path.join(OUTPUT_DIR, "test_data.csv")
CODE_ANNOTATED_FILE = os.path.join(OUTPUT_DIR, "intermediate_code_annotated.csv")
ANNOTATED_DATA_FILE = os.path.join(OUTPUT_DIR, "final_annotated_data.csv")
BEST_FEATURES_FILE = os.path.join(OUTPUT_DIR, "best_features.txt")

# Paths
DATA_DIR = "data" 