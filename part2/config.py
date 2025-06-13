from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "part2" / "data"
MODEL_CACHE_DIR = ROOT_DIR / "part2" / "model_cache"

AVAILABLE_MODELS = {
    "jhgan/ko-sroberta-multitask": {
        "name": "Ko-SRoBERTa",
        "description": "한국어에 특화된 SOTA 모델로, 문장 유사도와 의미적 검색에 최적화",
        "dimension": 768
    },
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS": {
        "name": "KR-SBERT",
        "description": "KLUE 데이터셋으로 학습된 한국어 SBERT 모델",
        "dimension": 768
    },
    "klue/bert-base": {
        "name": "KLUE-BERT",
        "description": "KLUE에서 제공하는 기본 BERT 모델",
        "dimension": 768
    }
}

TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.5 