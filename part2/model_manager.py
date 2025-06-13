from sentence_transformers import SentenceTransformer
from config import AVAILABLE_MODELS, MODEL_CACHE_DIR

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.model_name = None
        self.available_models = AVAILABLE_MODELS
    def list_available_models(self):
        return {
            model_id: {
                "name": info["name"],
                "description": info["description"],
                "dimension": info["dimension"]
            }
            for model_id, info in self.available_models.items()
        }
    def load_model(self, model_id):
        if model_id not in self.available_models:
            raise ValueError(f"Model {model_id} not found in available models")
        self.current_model = SentenceTransformer(model_id, cache_folder=str(MODEL_CACHE_DIR))
        self.model_name = model_id
        return True
    def get_current_model_info(self):
        if not self.current_model:
            return None
        return {
            "model_id": self.model_name,
            "name": self.available_models[self.model_name]["name"],
            "dimension": self.available_models[self.model_name]["dimension"],
            "description": self.available_models[self.model_name]["description"]
        }
    def encode(self, texts):
        if not self.current_model:
            raise ValueError("No model loaded. Please load a model first.")
        return self.current_model.encode(texts, convert_to_tensor=True) 