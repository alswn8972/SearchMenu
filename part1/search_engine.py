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
        self.full_embeddings = self.embedding_manager.create_embeddings(self.menu_processor.full_texts)
        self.page_embeddings = self.embedding_manager.create_embeddings(self.menu_processor.page_names)
        self.context_embeddings = self.embedding_manager.create_embeddings(self.menu_processor.context_texts)
    def search(self, query):
        query_embedding = self.embedding_manager.create_query_embedding(query)
        full_sim = self.embedding_manager.calculate_similarities(query_embedding, self.full_embeddings)
        page_sim = self.embedding_manager.calculate_similarities(query_embedding, self.page_embeddings)
        context_sim = self.embedding_manager.calculate_similarities(query_embedding, self.context_embeddings)
        results = []
        for i in range(len(self.menu_processor.menu_data)):
            weighted = self.menu_processor.calculate_weighted_similarity(full_sim[i], page_sim[i], context_sim[i])
            menu_item = self.menu_processor.get_menu_item(i)
            results.append({
                'Category': menu_item['Category'],
                'Service': menu_item['Service'],
                'page_name': menu_item['page_name'],
                'hierarchy': menu_item['hierarchy'],
                'full_similarity': full_sim[i],
                'page_similarity': page_sim[i],
                'context_similarity': context_sim[i],
                'weighted_similarity': weighted
            })
        results_df = pd.DataFrame(results)
        return results_df.sort_values('weighted_similarity', ascending=False).head(TOP_K_RESULTS) 