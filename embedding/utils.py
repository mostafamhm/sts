import torch
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import yaml
import os
import gc
from datetime import datetime
from preprocessing import dynamic_preprocess, PreprocessingOptions

class SimilarityModel:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_name_or_path = "BAAI/bge-m3"  
        self.load_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.reload_model()

    def _read_config(self):

        current_model_path = "BAAI/bge-m3"
        
        if not os.path.exists(self.config_path):
            print(f"Config file '{self.config_path}' not found. Using default: {current_model_path}")
            return current_model_path

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config and 'inference' in config and 'model_path' in config['inference']:
                path = config['inference']['model_path']
                if path:
                    current_model_path = path
                    print(f"Model path loaded from config: {current_model_path}")
                else:
                    print("model_path is empty in config. Using default.")
            else:
                print("No 'inference.model_path' in config. Using default.")
        except Exception as e:
            print(f"Error reading config: {e}. Using default: {current_model_path}")
        
        return current_model_path

    def reload_model(self) -> dict:

        new_model_name = self._read_config()
        
        if self.model is not None and new_model_name == self.model_name_or_path:
            print(f"Model configuration unchanged. Using existing model: {new_model_name}")
            return {"status": "success", "message": f"Model already loaded: {new_model_name}"}

        print(f"Loading model: '{new_model_name}' on {self.device}...")
        
        try:

            new_model = SentenceTransformer(new_model_name, device=self.device)
            
            old_model = self.model
            self.model = new_model
            self.model_name_or_path = new_model_name
            
            del old_model
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            self.load_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"? Successfully loaded model: {new_model_name}")
            return {"status": "success", "message": f"Model loaded: {new_model_name}"}
            
        except Exception as e:
            print(f"? Failed to load model: {e}")
            return {"status": "error", "message": str(e)}

    def encode_texts(self, texts: List[str], max_length: int = 8192) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Similarity model is not available.")
        
        try:
            legacy_preprocessing_options = PreprocessingOptions(
                character_normalization=False,
                number_normalization=None
            )
            processed_texts = [dynamic_preprocess(text, legacy_preprocessing_options) for text in texts]
            
            original_max_length = self.model.max_seq_length
            self.model.max_seq_length = max_length

            embeddings = self.model.encode(processed_texts)

            self.model.max_seq_length = original_max_length
            
            return embeddings
        except Exception as e:
            print(f"? Error during text encoding: {e}")
            raise

    def get_model_details(self) -> dict:
        return {
            "name": self.model_name_or_path,
            "deployed_at": self.load_time,
            "device": self.device
        }


similarity_model = SimilarityModel()
