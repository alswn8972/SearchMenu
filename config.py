# 모델 설정
MODEL_NAME = 'jhgan/ko-sroberta-multitask'

# 유사도 가중치 설정
SIMILARITY_WEIGHTS = {
    'full': 0.5,
    'page': 0.3,
    'context': 0.2
}

# 검색 결과 설정
TOP_K_RESULTS = 5 