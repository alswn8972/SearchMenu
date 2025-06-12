import pandas as pd
from embeddings import EmbeddingManager
from menu_processor import MenuProcessor
from config import TOP_K_RESULTS

class SearchEngine:
    def __init__(self, json_file_path):
        self.menu_processor = MenuProcessor(json_file_path)
        self.embedding_manager = EmbeddingManager()
        self._create_embeddings()
    
    def _create_embeddings(self):
        """모든 텍스트에 대한 임베딩을 생성합니다."""
        print("임베딩 생성 중...")
        self.full_embeddings = self.embedding_manager.create_embeddings(
            self.menu_processor.full_texts
        )
        self.page_embeddings = self.embedding_manager.create_embeddings(
            self.menu_processor.page_names
        )
        self.context_embeddings = self.embedding_manager.create_embeddings(
            self.menu_processor.context_texts
        )
        print("임베딩 생성 완료!")
    
    def search(self, query):
        """검색 쿼리에 대한 결과를 반환합니다."""
        # 쿼리 임베딩 생성
        query_embedding = self.embedding_manager.create_query_embedding(query)
        
        # 각 유형별 유사도 계산
        full_similarities = self.embedding_manager.calculate_similarities(
            query_embedding, self.full_embeddings
        )
        page_similarities = self.embedding_manager.calculate_similarities(
            query_embedding, self.page_embeddings
        )
        context_similarities = self.embedding_manager.calculate_similarities(
            query_embedding, self.context_embeddings
        )
        
        # 결과 데이터프레임 생성
        results = []
        for i in range(len(self.menu_processor.menu_data)):
            weighted_similarity = self.menu_processor.calculate_weighted_similarity(
                full_similarities[i],
                page_similarities[i],
                context_similarities[i]
            )
            
            menu_item = self.menu_processor.get_menu_item(i)
            results.append({
                'Category': menu_item['Category'],
                'Service': menu_item['Service'],
                'page_name': menu_item['page_name'],
                'hierarchy': menu_item['hierarchy'],
                'full_similarity': full_similarities[i],
                'page_similarity': page_similarities[i],
                'context_similarity': context_similarities[i],
                'weighted_similarity': weighted_similarity
            })
        
        # 가중치가 적용된 유사도 기준으로 정렬
        results_df = pd.DataFrame(results)
        return results_df.sort_values('weighted_similarity', ascending=False).head(TOP_K_RESULTS) 