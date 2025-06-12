from search_engine import SearchEngine

def format_similarity_score(score):
    """유사도 점수를 소수점 4자리까지 포맷팅합니다."""
    return f"{score:.4f}"

def display_results(query, results):
    """검색 결과를 보기 좋게 출력합니다."""
    print(f"\n검색어: {query}")
    print("\n유사한 항목들:")
    
    for idx, row in results.iterrows():
        print(f"\n{idx + 1}. 전체 유사도: {format_similarity_score(row['full_similarity'])}")
        print(f"   - 페이지별 유사도: {format_similarity_score(row['page_similarity'])}")
        print(f"   - 전체 컨텍스트 유사도: {format_similarity_score(row['context_similarity'])}")
        print(f"   - 종합 점수: {format_similarity_score(row['weighted_similarity'])}")
        print(f"   카테고리: {row['Category']}")
        print(f"   서비스: {row['Service']}")
        print(f"   페이지명: {row['page_name']}")

def main():
    # 검색 엔진 초기화
    print("검색 엔진 초기화 중...")
    search_engine = SearchEngine('ia-data.json')
    
    while True:
        # 검색어 입력
        query = input("\n검색어를 입력하세요 (종료하려면 'q' 입력): ")
        if query.lower() == 'q':
            break
        
        # 검색 실행
        results = search_engine.search(query)
        
        # 결과 출력
        display_results(query, results)

if __name__ == "__main__":
    main() 