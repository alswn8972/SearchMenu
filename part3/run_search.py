from menu_data_loader import MenuDataLoader
from vector_llm_search import VectorLLMSearch
from config import MENU_DATA_PATH

# 메뉴 데이터 로드
data_loader = MenuDataLoader(MENU_DATA_PATH)
if not data_loader.load_data():
    print("❌ 메뉴 데이터 로드에 실패했습니다.")
    exit(1)

# 검색 시스템 초기화
searcher = VectorLLMSearch()

# 검색어 입력
query = input("검색어를 입력하세요: ").strip()
menu_data = data_loader.get_menu_data()

# 검색 실행
results = searcher.search(query, menu_data, max_results=5)

# 결과 표시
searcher.display_results(results, query) 