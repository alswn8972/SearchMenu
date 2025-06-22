import os
from dotenv import load_dotenv

# .env 파일을 part3 폴더에서 명시적으로 로드
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# OpenAI API 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-3.5-turbo"  # 또는 "gpt-4" 사용 가능

# 메뉴 데이터 경로
MENU_DATA_PATH = "ia-data.json"

# 매칭 설정
MAX_RESULTS = 5  # 최대 결과 수
SIMILARITY_THRESHOLD = 0.7  # 유사도 임계값

# 프롬프트 템플릿
MATCHING_PROMPT_TEMPLATE = """
다음은 한국 음식점 메뉴 목록입니다:

{menu_list}

사용자가 입력한 메뉴 이름: "{user_input}"

위 메뉴 목록에서 사용자 입력과 가장 유사한 메뉴들을 찾아주세요.
다음 기준으로 매칭해주세요:
1. 메뉴 이름의 유사성 (한글, 영어, 음차 등)
2. 음식 종류의 유사성
3. 조리 방법의 유사성
4. 재료의 유사성

가장 유사한 {max_results}개의 메뉴를 JSON 형식으로 반환해주세요:
[
  {{
    "menu_name": "메뉴명",
    "similarity_score": 0.95,
    "reason": "매칭 이유"
  }}
]

유사도 점수는 0.0~1.0 사이로 주세요.
""" 