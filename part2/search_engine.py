import json
import numpy as np
import faiss
from typing import List, Dict, Tuple, Any
from config import TOP_K_RESULTS

class SearchEngine:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.menu_data = []
        self.index = None
        self.dimension = None
        self.WEIGHTS = {'page_name': 0.4, 'service': 0.4, 'context': 0.2}

    def normalize_embeddings(self, embeddings):
        faiss.normalize_L2(embeddings)
        return embeddings

    def build_index(self, menu_data):
        self.menu_data = menu_data
        self.dimension = self.model_manager.get_current_model_info()['dimension']
        page_names = [item['page_name'] for item in menu_data]
        services = [item['Service'] for item in menu_data]
        contexts = [f"{item['Category']} {' '.join(item['hierarchy'])}" for item in menu_data]
        self.page_name_embeddings = self.model_manager.encode(page_names).cpu().numpy().astype('float32')
        self.service_embeddings = self.model_manager.encode(services).cpu().numpy().astype('float32')
        self.context_embeddings = self.model_manager.encode(contexts).cpu().numpy().astype('float32')
        self.page_name_embeddings = self.normalize_embeddings(self.page_name_embeddings)
        self.service_embeddings = self.normalize_embeddings(self.service_embeddings)
        self.context_embeddings = self.normalize_embeddings(self.context_embeddings)
        weighted_embeddings = (
            self.WEIGHTS['page_name'] * self.page_name_embeddings +
            self.WEIGHTS['service'] * self.service_embeddings +
            self.WEIGHTS['context'] * self.context_embeddings
        )
        weighted_embeddings = self.normalize_embeddings(weighted_embeddings)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(weighted_embeddings)

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """쿼리에 대해 가장 유사한 메뉴를 검색합니다."""
        if not self.index:
            return []

        # 쿼리 임베딩 생성
        query_embedding = self.model_manager.encode([query]).cpu().numpy().astype('float32')
        query_embedding = self.normalize_embeddings(query_embedding)

        # FAISS를 사용하여 검색
        scores, indices = self.index.search(query_embedding, top_k)

        # 결과 포맷팅
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.menu_data):
                item = self.menu_data[idx]
                # 각각의 유사도 계산
                page_sim = float(np.dot(query_embedding[0], self.page_name_embeddings[idx]))
                service_sim = float(np.dot(query_embedding[0], self.service_embeddings[idx]))
                context_sim = float(np.dot(query_embedding[0], self.context_embeddings[idx]))
                
                # 유사도 점수 조정 (0.6 ~ 1.0 범위로)
                page_sim = 0.6 + (page_sim * 0.4)
                service_sim = 0.6 + (service_sim * 0.4)
                context_sim = 0.6 + (context_sim * 0.4)
                total_sim = 0.6 + (float(score) * 0.4)
                
                # 가중치 적용
                weighted_score = (
                    self.WEIGHTS['page_name'] * page_sim +
                    self.WEIGHTS['service'] * service_sim +
                    self.WEIGHTS['context'] * context_sim
                )
                
                results.append({
                    'similarity': total_sim,
                    'page_name_similarity': page_sim,
                    'service_similarity': service_sim,
                    'context_similarity': context_sim,
                    'weighted_score': weighted_score,
                    'category': item['Category'],
                    'service': item['Service'],
                    'menu_item': item['page_name']
                })

        return results 