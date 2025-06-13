import json
from model_manager import ModelManager
from search_engine import SearchEngine
from config import DATA_DIR
import os

def format_similarity_score(score):
    return f"{score:.4f}"

def display_results(query, results):
    print(f"\n검색어: {query}")
    print("\n유사한 항목들:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 전체 유사도: {format_similarity_score(result['similarity'])}")
        print(f"   - 페이지별 유사도: {format_similarity_score(result['page_name_similarity'])}")
        print(f"   - 서비스 유사도: {format_similarity_score(result['service_similarity'])}")
        print(f"   - 전체 컨텍스트 유사도: {format_similarity_score(result['context_similarity'])}")
        print(f"   - 종합 점수: {format_similarity_score(result['weighted_score'])}")
        print(f"   카테고리: {result['category']}")
        print(f"   서비스: {result['service']}" )
        print(f"   페이지명: {result['menu_item']}")

def load_menu_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"메뉴 데이터 로드 중 오류 발생: {str(e)}")
        return []

def main():
    print("검색 시스템 초기화 중...")
    
    # 모델 매니저 초기화
    model_manager = ModelManager()
    search_engine = SearchEngine(model_manager)
    menu_data = load_menu_data("ia-data.json")
    if not menu_data:
        print("메뉴 데이터를 찾을 수 없습니다. ia-data.json 파일을 확인해주세요.")
        return
    print("\n사용 가능한 모델 목록:")
    models = list(model_manager.list_available_models().items())
    for idx, (model_id, info) in enumerate(models, 1):
        print(f"\n{idx}. {info['name']}")
        print(f"   설명: {info['description']}")
        print(f"   차원: {info['dimension']}")
    while True:
        try:
            choice = input("\n사용할 모델의 번호를 입력하세요 (종료하려면 'q' 입력): ")
            if choice.lower() == 'q':
                return
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models):
                model_id = models[choice_idx][0]
                if model_manager.load_model(model_id):
                    print(f"\n모델 '{model_manager.get_current_model_info()['name']}' 로드 완료!")
                    break
            else:
                print("올바른 번호를 입력해주세요.")
        except ValueError:
            print("숫자를 입력해주세요.")
    print("\n검색 인덱스 구축 중...")
    search_engine.build_index(menu_data)
    print("검색 인덱스 구축 완료!")
    while True:
        query = input("\n검색어를 입력하세요 (종료하려면 'q' 입력): ")
        if query.lower() == 'q':
            break
        results = search_engine.search(query)
        display_results(query, results)

if __name__ == "__main__":
    main() 