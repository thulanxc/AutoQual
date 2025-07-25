# -*- coding: utf-8 -*-

import os
import time
from typing import Optional
from openai import OpenAI
import config

class LLMProvider:
    """
    A class that encapsulates interactions with multiple Large Language Models.
    - A High-Performance (HP) model for complex generation tasks.
    - A General-Purpose (GP) model for routine annotation tasks.
    """
    def __init__(self):
        # High-Performance Client (e.g., DeepSeek)
        if not config.HP_API_KEY:
            raise ValueError("High-Performance model API Key (HP_API_KEY) is not set in config.py.")
        self.hp_client = OpenAI(
            api_key=config.HP_API_KEY,
            base_url=config.HP_BASE_URL
        )
        self.hp_model = config.HP_MODEL_NAME

        # General-Purpose Client (e.g., Qwen)
        if not config.GP_API_KEY:
            raise ValueError("General-Purpose model API Key (GP_API_KEY) is not set in config.py.")
        self.gp_client = OpenAI(
            api_key=config.GP_API_KEY,
            base_url=config.GP_BASE_URL,
            timeout=30.0
        )
        self.gp_model = config.GP_MODEL_NAME

        # For error handling in general model calls
        self.error_counts = {"gp_model": 0}
        self.last_error_time = {"gp_model": 0}

    def get_completion(self, prompt: str, output_filename: str, overwrite: bool = False) -> str:
        """
        Gets a completion from the High-Performance model.
        This is used for complex generation tasks (roles, features, tools).
        It ensures a result exists, either by generating it or reading from a cache file.
        """
        if not os.path.exists(config.OUTPUT_DIR):
            os.makedirs(config.OUTPUT_DIR)
        
        filepath = os.path.join(config.OUTPUT_DIR, output_filename)

        if not overwrite and os.path.exists(filepath):
            print(f"File '{output_filename}' already exists. Reading from cache.")
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()

        print(f"Calling HP model to generate '{output_filename}'...")
        try:
            response = self.hp_client.chat.completions.create(
                model=self.hp_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            content = response.choices[0].message.content
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"...Successfully generated and saved to '{output_filename}'.")
            return content
        except Exception as e:
            error_message = f"Error calling HP model for '{output_filename}': {e}"
            print(error_message)
            # Still save the error to the file so the process can potentially continue
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(error_message)
            return error_message

    def call_general_model(self, prompt: str, temperature: float = 0, thread_id: Optional[int] = None) -> str:
        """
        Calls the General-Purpose model (e.g., qwen-turbo) with robust error handling.
        Used for high-volume tasks like annotation.
        """
        thread_info = f"[Thread {thread_id}] " if thread_id is not None else ""
        
        while True:
            try:
                response = self.gp_client.chat.completions.create(
                    model=self.gp_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    stream=False
                )
                
                self.error_counts["gp_model"] = 0
                content = response.choices[0].message.content
                return content
                
            except Exception as e:
                self.error_counts["gp_model"] += 1
                current_time = time.time()
                # Silently handle errors and prepare for retry
                if current_time - self.last_error_time["gp_model"] > 5:
                    self.last_error_time["gp_model"] = current_time
                time.sleep(1) 