import json
from config import SIMILARITY_WEIGHTS

class MenuProcessor:
    def __init__(self, json_file_path):
        self.menu_data = self._load_data(json_file_path)
        self.full_texts = []
        self.page_names = []
        self.context_texts = []
        self._prepare_texts()
    
    def _load_data(self, json_file_path):
        """JSON 파일에서 메뉴 데이터를 로드합니다."""
        with open(json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _prepare_texts(self):
        """메뉴 데이터에서 검색에 사용할 텍스트들을 준비합니다."""
        for item in self.menu_data:
            # 전체 텍스트
            full_text = f"{item['Category']} {item['Service']} {item['page_name']} {' '.join(item['hierarchy'])}"
            self.full_texts.append(full_text)
            
            # 페이지명
            self.page_names.append(item['page_name'])
            
            # 컨텍스트 텍스트
            context_text = f"{item['Category']} {item['Service']}"
            self.context_texts.append(context_text)
    
    def calculate_weighted_similarity(self, full_sim, page_sim, context_sim):
        """가중치가 적용된 유사도를 계산합니다."""
        return (
            SIMILARITY_WEIGHTS['full'] * full_sim +
            SIMILARITY_WEIGHTS['page'] * page_sim +
            SIMILARITY_WEIGHTS['context'] * context_sim
        )
    
    def get_menu_item(self, index):
        """인덱스에 해당하는 메뉴 항목을 반환합니다."""
        return self.menu_data[index] 