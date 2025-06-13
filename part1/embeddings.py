from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingManager:
    def __init__(self, model_name='jhgan/ko-sroberta-multitask'):
        self.model = SentenceTransformer(model_name)
    def create_embeddings(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    def create_query_embedding(self, query):
        return self.create_embeddings([query])[0]
    def calculate_similarities(self, query_emb, emb_matrix):
        return np.dot(emb_matrix, query_emb) 