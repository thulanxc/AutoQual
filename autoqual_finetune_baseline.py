# -*- coding: utf-8 -*-

"""
本脚本实现了一个先进的混合基线模型。
它将AutoQual选择的特征与一个可训练的Sentence Transformer模型相结合，
并对整个网络进行端到端的微调，以预测评论质量分数。
"""

import os
import config
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# --- 设备与模型设置 ---
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

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# --- 1. 数据准备 ---

def prepare_data_for_finetuning():
    """
    加载、清洗并切分数据，为端到端微调准备所有需要的数据。
    该函数逻辑与 feature_selector.py 中的数据准备对齐。
    """
    print("\n--- 步骤1: 准备数据 ---")
    # 检查文件
    if not os.path.exists(config.ANNOTATED_DATA_FILE) or not os.path.exists(config.BEST_FEATURES_FILE):
        raise FileNotFoundError("错误: 未找到 'final_annotated_data.csv' 或 'best_features.txt'。请先运行主流程。")

    # 加载特征列表和已标注数据
    with open(config.BEST_FEATURES_FILE, 'r', encoding='utf-8') as f:
        best_features = [line.strip() for line in f if line.strip()]
    
    full_df = pd.read_csv(config.ANNOTATED_DATA_FILE)
    
    # 数据清洗
    clean_df = full_df.dropna(subset=['text', 'score'] + best_features).copy()
    print(f"数据清洗后，保留 {len(clean_df)} / {len(full_df)} 行。")

    # 划分训练/测试集
    train_df, test_df = train_test_split(clean_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    print(f"数据切分: {len(train_df)} 训练样本, {len(test_df)} 测试样本。")

    # 特征归一化
    feature_scaler = MinMaxScaler()
    train_df[best_features] = feature_scaler.fit_transform(train_df[best_features])
    test_df[best_features] = feature_scaler.transform(test_df[best_features])

    # 分数归一化
    score_scaler = MinMaxScaler()
    train_df['score'] = score_scaler.fit_transform(train_df[['score']]).flatten()
    test_df['score'] = score_scaler.transform(test_df[['score']]).flatten()
    
    return train_df, test_df, best_features

# --- 2. PyTorch Dataset ---

class CombinedFeatureDataset(Dataset):
    def __init__(self, dataframe, features, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.texts = dataframe['text'].tolist()
        self.autoqual_features = dataframe[features].values.astype(np.float32)
        self.scores = dataframe['score'].values.astype(np.float32)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'ids': inputs['input_ids'].flatten(),
            'mask': inputs['attention_mask'].flatten(),
            'autoqual_feats': torch.tensor(self.autoqual_features[idx], dtype=torch.float),
            'targets': torch.tensor(self.scores[idx], dtype=torch.float)
        }

# --- 3. 自定义模型 ---

class AutoQualFinetuneModel(nn.Module):
    def __init__(self, transformer_model_name, num_autoqual_features):
        super(AutoQualFinetuneModel, self).__init__()
        # 身体：Transformer模型
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        transformer_output_dim = self.transformer.config.hidden_size
        
        # 头部：一个简单的MLP
        self.head = nn.Sequential(
            nn.Linear(transformer_output_dim + num_autoqual_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )

    def _mean_pooling(self, model_output, attention_mask):
        """执行平均池化，以获得句子级别的Embedding。"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, ids, mask, autoqual_feats):
        # 通过Transformer获取词级别的Embeddings
        transformer_output = self.transformer(ids, attention_mask=mask)
        
        # 将词级别Embeddings池化为句子级别的Embedding
        sentence_embedding = self._mean_pooling(transformer_output, mask)
        
        # 拼接句子Embedding和AutoQual特征
        combined_features = torch.cat([sentence_embedding, autoqual_feats], dim=1)
        
        # 通过头部得到最终预测
        output = self.head(combined_features)
        return output

# --- 4. 训练与评估 ---

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Training"):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        autoqual_feats = data['autoqual_feats'].to(device)
        targets = data['targets'].to(device)

        outputs = model(ids, mask, autoqual_feats)
        loss = loss_fn(outputs.squeeze(), targets)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Evaluating"):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            autoqual_feats = data['autoqual_feats'].to(device)
            targets = data['targets'].to(device)

            outputs = model(ids, mask, autoqual_feats)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            
    rho, _ = spearmanr(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    return rho, mae

def main():
    """主执行函数"""
    try:
        # 1. 准备数据
        train_df, test_df, best_features = prepare_data_for_finetuning()
        
        # 2. 创建DataLoader
        print("\n--- 步骤2: 创建PyTorch DataLoaders ---")
        tokenizer = AutoTokenizer.from_pretrained(config.FINETUNE_MODEL_NAME)
        
        train_dataset = CombinedFeatureDataset(train_df, best_features, tokenizer)
        test_dataset = CombinedFeatureDataset(test_df, best_features, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config.FINETUNE_BATCH_SIZE, shuffle=False, num_workers=0)
        print("✅ DataLoaders创建完成。")
        
        # 3. 初始化模型、损失函数和优化器
        print("\n--- 步骤3: 初始化模型和优化器 ---")
        model = AutoQualFinetuneModel(config.FINETUNE_MODEL_NAME, num_autoqual_features=len(best_features))
        model.to(target_device)
        
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.FINETUNE_LEARNING_RATE)
        print("✅ 模型初始化完成。")

        # 4. 训练循环
        print("\n--- 步骤4: 开始端到端微调 ---")
        best_rho = -1
        for epoch in range(config.FINETUNE_EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{config.FINETUNE_EPOCHS} ---")
            avg_loss = train_epoch(model, train_loader, loss_fn, optimizer, target_device)
            print(f"训练完成. 平均损失: {avg_loss:.4f}")
            
            rho, mae = eval_model(model, test_loader, target_device)
            print(f"评估结果: Spearman's Rho = {rho:.4f}, MAE = {mae:.4f}")

            if rho > best_rho:
                best_rho = rho
                print("✨ 新的最佳Spearman's Rho！")

        print("\n" + "="*50)
        print("      AutoQual可训练混合模型性能")
        print("="*50)
        print(f"  - 最佳 Spearman's Rho: {best_rho:.4f}")
        print("="*50)

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
    except Exception as e:
        print(f"\n程序执行时发生意外错误: {e}")


if __name__ == "__main__":
    main() 