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
    主函数，项目的入口。
    """
    # 检查输入文件是否存在
    if not os.path.exists(config.SCENE_DESCRIPTION_FILE):
        print(f"错误：场景描述文件未找到，路径: {config.SCENE_DESCRIPTION_FILE}")
        return
    if not os.path.exists(config.DATA_FILE):
        print(f"错误：数据文件未找到，路径: {config.DATA_FILE}")
        return

    # 加载输入
    try:
        with open(config.SCENE_DESCRIPTION_FILE, 'r', encoding='utf-8') as f:
            scene_description = f.read()
        
        df = pd.read_csv(config.DATA_FILE)
        
        # 验证CSV文件格式
        if 'text' not in df.columns or 'score' not in df.columns:
            print("错误：CSV文件必须包含 'text' 和 'score' 两列。")
            return
            
    except Exception as e:
        print(f"加载输入文件时出错: {e}")
        return

    print("="*50)
    print(" AutoQual: 自动特征发现代理")
    print("="*50)
    print(f"当前模式: {config.EXECUTION_MODE}")
    print(f"场景描述文件: {config.SCENE_DESCRIPTION_FILE}")
    print(f"数据文件: {config.DATA_FILE}")
    print("="*50)

    try:
        # 初始化核心模块
        llm_provider = LLMProvider()
        memory_manager = MemoryManager(llm_provider)
        feature_generator = FeatureGenerator(llm_provider, memory_manager)

        # 执行第一部分：生成初始特征池
        final_features_str = feature_generator.generate_initial_features(scene_description, df)

        if not final_features_str or final_features_str.startswith("Error"):
            print("\nCritical error in feature generation. Aborting.")
            return

        # 执行第二部分：生成注释工具
        tool_generator = ToolGenerator(llm_provider, df)
        tool_generator.generate_all_tools(final_features_str, feature_generator.overwrite_files)
        
        # 执行第三部分：特征注释
        print("\n--- STAGE 6/6: ANNOTATING FEATURES WITH ALL TOOLS ---")
        annotator = Annotator(llm_provider, df)
        annotated_df = annotator.annotate_features()

        if annotated_df.empty:
            print("\nAnnotation resulted in an empty dataframe. Aborting feature selection.")
            return

        # --- Part 4: Reflective Feature Selection (Stage 7) ---
        print("\n--- STAGE 7/7: SELECTING BEST FEATURES WITH REFLECTION ---")
        selector = FeatureSelector(config.ANNOTATED_DATA_FILE, llm_provider, scene_description)
        best_feature_set = selector.select_features()

        # --- Part 5: Cross-Task Memory Summarization ---
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
        print(f"\n配置错误: {ve}")
    except Exception as e:
        print(f"\n程序执行时发生意外错误: {e}")


if __name__ == "__main__":
    main() 