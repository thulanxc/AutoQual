# -*- coding: utf-8 -*-

import os
import json
import config
import prompts
from llm_provider import LLMProvider
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MemoryManager:
    """
    Manages the dual-level memory architecture for AutoQual.
    Handles saving and retrieving cross-task experiences.
    """
    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.memory_file = config.CROSS_TASK_MEMORY_FILE
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump([], f) # Initialize with an empty list

    def summarize_and_save_experience(self, task_name: str, scene_description: str, best_features: list):
        """
        Generates a structured summary of a completed task and saves it to the long-term memory.
        """
        print(f"  Synthesizing experience for task: {task_name}")
        prompt = prompts.CROSS_TASK_MEMORY_SUMMARIZATION_PROMPT.format(
            task_name=task_name,
            scene_description=scene_description,
            best_features="\n".join([f"- {f}" for f in best_features])
        )
        
        summary = self.llm.get_completion(prompt, f"temp_summary_{task_name}.txt", overwrite=True)
        
        memory_entry = {
            "task_name": task_name,
            "scene_description": scene_description,
            "summary": summary
        }
        
        try:
            with open(self.memory_file, 'r+', encoding='utf-8') as f:
                memories = json.load(f)
                # Avoid duplicates
                if not any(mem['task_name'] == task_name for mem in memories):
                    memories.append(memory_entry)
                    f.seek(0)
                    json.dump(memories, f, indent=4)
                    print(f"  Successfully saved experience from '{task_name}' to cross-task memory.")
                else:
                    print(f"  Experience for '{task_name}' already exists in memory. Skipping save.")
        except (IOError, json.JSONDecodeError) as e:
            print(f"  Error accessing or updating memory file: {e}")

    def retrieve_relevant_memories(self, new_scene_description: str, top_k: int = 3) -> str:
        """
        Retrieves the top_k most relevant memories for a new task based on semantic similarity.
        """
        print("\n--- Retrieving relevant experiences from cross-task memory... ---")
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memories = json.load(f)
        except (IOError, json.JSONDecodeError):
            print("  Could not read cross-task memory file. Proceeding without past experiences.")
            return ""

        if not memories:
            print("  Cross-task memory is empty. No past experiences to retrieve.")
            return ""

        # Use TF-IDF and cosine similarity to find the most relevant memories
        documents = [mem['scene_description'] for mem in memories]
        documents.append(new_scene_description)
        
        try:
            vectorizer = TfidfVectorizer().fit_transform(documents)
            vectors = vectorizer.toarray()
            cosine_matrix = cosine_similarity(vectors)
            
            # Get similarities between the new scene and all past scenes
            similarity_scores = cosine_matrix[-1][:-1]
            
            # Get indices of top_k most similar memories
            top_indices = similarity_scores.argsort()[-top_k:][::-1]
            
            if not any(similarity_scores > 0): # Check if there is any similarity
                print("  No relevant memories found based on similarity. Proceeding without past experiences.")
                return ""

            retrieved_summaries = [memories[i]['summary'] for i in top_indices if similarity_scores[i] > 0]
            
            if not retrieved_summaries:
                print("  No relevant memories found. Proceeding without past experiences.")
                return ""
                
            print(f"  Retrieved {len(retrieved_summaries)} relevant memories.")
            return "\n\n".join(retrieved_summaries)

        except Exception as e:
            print(f"  An error occurred during memory retrieval: {e}. Proceeding without past experiences.")
            return "" 