# -*- coding: utf-8 -*-

"""
这个脚本实现了一个混合baseline，结合AutoQual选择的有效特征和embedding模型特征。
为了确保结果的绝对可比性，此脚本为每个独立模型（Embedding、AutoQual）
精确复刻了其对应基线脚本（embedding_baseline.py, feature_selector.py）的数据准备流程。
"""

import os
import config

# --- 设备设置 (与所有基线一致) ---
target_device = config.FINETUNE_DEVICE
if 'cuda' in target_device:
    gpu_id = '0'
    if ':' in target_device:
        gpu_id = target_device.split(':')[-1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    target_device = 'cuda'
    print(f"在GPU上运行: '{config.FINETUNE_DEVICE}'.")
else:
    print(f"在CPU上运行.")

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# --- 1. 复刻 embedding_baseline.py 的数据准备 ---
def prepare_embedding_baseline_data():
    """完全复制 embedding_baseline.py 的逻辑。"""
    print("\n--- 准备数据 (Embedding Baseline方式) ---")
    if not os.path.exists(config.DATA_FILE):
        raise FileNotFoundError(f"未找到原始数据文件: '{config.DATA_FILE}'")
    
    full_df = pd.read_csv(config.DATA_FILE).dropna(subset=['text', 'score'])
    train_df, test_df = train_test_split(
        full_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    X_train_text = train_df['text'].tolist()
    X_test_text = test_df['text'].tolist()
    y_train_raw = train_df['score'].values
    y_test_raw = test_df['score'].values
    
    score_scaler = MinMaxScaler()
    y_train = score_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test = score_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    print(f"数据准备完成: {len(X_train_text)} 训练样本, {len(X_test_text)} 测试样本。")
    return X_train_text, y_train, X_test_text, y_test

# --- 2. 复刻 feature_selector.py 的数据准备 ---
def prepare_autoqual_baseline_data():
    """完全复制 feature_selector.py 的逻辑。"""
    print("\n--- 准备数据 (AutoQual Baseline方式) ---")
    if not os.path.exists(config.ANNOTATED_DATA_FILE):
        raise FileNotFoundError(f"未找到注释数据文件: '{config.ANNOTATED_DATA_FILE}'")
    if not os.path.exists(config.BEST_FEATURES_FILE):
        raise FileNotFoundError(f"未找到最佳特征文件: '{config.BEST_FEATURES_FILE}'")

    with open(config.BEST_FEATURES_FILE, 'r', encoding='utf-8') as f:
        best_features = [line.strip() for line in f if line.strip()]

    full_df = pd.read_csv(config.ANNOTATED_DATA_FILE)
    
    # 清洗: 在所有需要的列上移除NaN
    initial_rows = len(full_df)
    clean_df = full_df.dropna(subset=['text', 'score'] + best_features).copy()
    cleaned_rows = len(clean_df)
    if initial_rows > cleaned_rows:
        print(f"数据清洗: 移除了 {initial_rows - cleaned_rows} 个包含空值的行。")

    train_df, test_df = train_test_split(
        clean_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"数据准备完成: {len(train_df)} 训练样本, {len(test_df)} 测试样本。")

    # 提取特征和分数
    X_train_autoqual_raw = train_df[best_features]
    X_test_autoqual_raw = test_df[best_features]
    y_train_raw = train_df['score'].values
    y_test_raw = test_df['score'].values
    
    # 归一化 (与feature_selector.py保持一致)
    feature_scaler = MinMaxScaler()
    X_train_autoqual = feature_scaler.fit_transform(X_train_autoqual_raw)
    X_test_autoqual = feature_scaler.transform(X_test_autoqual_raw)

    score_scaler = MinMaxScaler()
    y_train = score_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test = score_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
    
    # 同时返回文本，为混合模型做准备
    X_train_text = train_df['text'].tolist()
    X_test_text = test_df['text'].tolist()

    return X_train_autoqual, y_train, X_test_autoqual, y_test, X_train_text, X_test_text

def generate_embeddings(X_train_text, X_test_text):
    """为给定的文本列表生成Embeddings。"""
    print(f"加载 sentence-transformer 模型到设备: '{target_device}'...")
    model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=target_device)
    
    print(f"为 {len(X_train_text)} 个训练样本生成embedding...")
    X_train_embeddings = model.encode(X_train_text, normalize_embeddings=True, show_progress_bar=True)
    
    print(f"为 {len(X_test_text)} 个测试样本生成embedding...")
    X_test_embeddings = model.encode(X_test_text, normalize_embeddings=True, show_progress_bar=True)
    return X_train_embeddings, X_test_embeddings

def evaluate_model(X_train, X_test, y_train, y_test, model_name):
    """训练和评估Linear Regression模型。"""
    print(f"训练 {model_name} Linear Regression模型...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    predictions = lr_model.predict(X_test)
    rho, _ = spearmanr(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return rho, mae

def main():
    print("--- 运行混合Baseline模型 (AutoQual + Embedding) ---")
    
    try:
        # --- Part 1: 精确复现 Embedding Baseline ---
        print("\n" + "="*60)
        print("评估单独的Embedding特征 (与embedding_baseline.py对齐)")
        print("="*60)
        X_train_text_eb, y_train_eb, X_test_text_eb, y_test_eb = prepare_embedding_baseline_data()
        X_train_emb, X_test_emb = generate_embeddings(X_train_text_eb, X_test_text_eb)
        rho_embedding, mae_embedding = evaluate_model(X_train_emb, X_test_emb, y_train_eb, y_test_eb, "Embedding Baseline")
        print(f"Embedding特征性能: Rho={rho_embedding:.4f}, MAE={mae_embedding:.4f}")

        # --- Part 2: 精确复现 AutoQual Baseline ---
        print("\n" + "="*60)
        print("评估单独的AutoQual特征 (与main.py -> feature_selector.py对齐)")
        print("="*60)
        (X_train_aq, y_train_aq, X_test_aq, y_test_aq, 
         X_train_text_hybrid, X_test_text_hybrid) = prepare_autoqual_baseline_data()
        rho_autoqual, mae_autoqual = evaluate_model(X_train_aq, X_test_aq, y_train_aq, y_test_aq, "AutoQual Baseline")
        print(f"AutoQual特征性能: Rho={rho_autoqual:.4f}, MAE={mae_autoqual:.4f}")
        
        # --- Part 3: 混合模型 (在AutoQual的对齐数据子集上) ---
        print("\n" + "="*60)
        print("评估拼接后的特征 (AutoQual + Embedding)")
        print("="*60)
        # 为混合模型生成对应的embeddings
        X_train_emb_hybrid, X_test_emb_hybrid = generate_embeddings(X_train_text_hybrid, X_test_text_hybrid)
        
        # 拼接特征
        X_train_combined = np.concatenate([X_train_aq, X_train_emb_hybrid], axis=1)
        X_test_combined = np.concatenate([X_test_aq, X_test_emb_hybrid], axis=1)
        print(f"拼接后特征形状: 训练集 {X_train_combined.shape}, 测试集 {X_test_combined.shape}")
        
        # 使用与AutoQual模型相同的y进行评估
        rho_combined, mae_combined = evaluate_model(X_train_combined, X_test_combined, y_train_aq, y_test_aq, "混合模型")
        print(f"混合特征性能: Rho={rho_combined:.4f}, MAE={mae_combined:.4f}")

        # --- Part 4: 总结 ---
        print("\n" + "="*60)
        print("         模型性能总结")
        print("="*60)
        print(f"{'模型':<25} {'Spearman Rho':<15} {'MAE':<10}")
        print("-"*60)
        print(f"{'AutoQual Baseline':<25} {rho_autoqual:<15.4f} {mae_autoqual:<10.4f}")
        print(f"{'Embedding Baseline':<25} {rho_embedding:<15.4f} {mae_embedding:<10.4f}")
        print(f"{'混合模型':<25} {rho_combined:<15.4f} {mae_combined:<10.4f}")
        print("="*60)

        improvement_vs_autoqual = ((rho_combined - rho_autoqual) / rho_autoqual) * 100 if rho_autoqual != 0 else float('inf')
        improvement_vs_embedding = ((rho_combined - rho_embedding) / rho_embedding) * 100 if rho_embedding != 0 else float('inf')
        
        print(f"\n混合模型 vs AutoQual: Spearman's Rho 提升 {improvement_vs_autoqual:.2f}%")
        print(f"混合模型 vs Embedding: Spearman's Rho 提升 {improvement_vs_embedding:.2f}%")
        
    except FileNotFoundError as e:
        print(f"\n错误: 缺少必需文件，无法继续。 {e}")
    except Exception as e:
        print(f"\n程序执行时发生意外错误: {e}")

if __name__ == "__main__":
    main() 