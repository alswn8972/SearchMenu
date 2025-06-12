from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from config import MODEL_NAME

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
    
    def create_embeddings(self, texts, show_progress=True):
        """텍스트 리스트에 대한 임베딩을 생성합니다."""
        return self.model.encode(texts, show_progress_bar=show_progress)
    
    def create_query_embedding(self, query):
        """검색 쿼리에 대한 임베딩을 생성합니다."""
        return self.model.encode(query)
    
    def calculate_similarities(self, query_embedding, embeddings):
        """쿼리와 임베딩들 간의 유사도를 계산합니다."""
        return cosine_similarity([query_embedding], embeddings)[0] 