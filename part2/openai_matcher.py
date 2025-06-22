import openai
import json
import os
from typing import List, Dict, Tuple
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIMenuMatcher:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        OpenAI를 사용한 메뉴 매칭 시스템 초기화
        
        Args:
            api_key: OpenAI API 키 (환경변수 OPENAI_API_KEY에서도 읽을 수 있음)
            model: 사용할 OpenAI 모델명
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY를 설정하거나 api_key 매개변수를 전달하세요.")
        
        openai.api_key = self.api_key
        self.menu_data = []
        
    def load_menu_data(self, menu_file: str = "ia-data.json"):
        """메뉴 데이터 로드"""
        try:
            with open(menu_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 데이터 구조에 따라 메뉴 리스트 추출
                if isinstance(data, list):
                    self.menu_data = data
                elif isinstance(data, dict) and "menu" in data:
                    self.menu_data = data["menu"]
                else:
                    self.menu_data = data
            logger.info(f"메뉴 데이터 {len(self.menu_data)}개 로드 완료")
        except Exception as e:
            logger.error(f"메뉴 데이터 로드 실패: {e}")
            raise
    
    def find_similar_menus(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        OpenAI를 사용해서 유사한 메뉴 찾기
        
        Args:
            query: 검색할 메뉴 이름
            top_k: 반환할 결과 개수
            
        Returns:
            유사한 메뉴 리스트 (유사도 점수 포함)
        """
        if not self.menu_data:
            raise ValueError("메뉴 데이터가 로드되지 않았습니다. load_menu_data()를 먼저 호출하세요.")
        
        # 메뉴 이름 리스트 생성
        menu_names = []
        for item in self.menu_data:
            if isinstance(item, dict):
                name = item.get('name', item.get('menu_name', str(item)))
            else:
                name = str(item)
            menu_names.append(name)
        
        # OpenAI API 호출을 위한 프롬프트 생성
        prompt = self._create_matching_prompt(query, menu_names, top_k)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "당신은 한국 음식 메뉴 매칭 전문가입니다. 사용자가 입력한 메뉴와 가장 유사한 메뉴들을 찾아주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            return self._parse_ai_response(result, top_k)
            
        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {e}")
            raise
    
    def _create_matching_prompt(self, query: str, menu_names: List[str], top_k: int) -> str:
        """AI 매칭을 위한 프롬프트 생성"""
        menu_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(menu_names)])
        
        prompt = f"""
사용자가 입력한 메뉴: "{query}"

다음 메뉴 리스트에서 가장 유사한 메뉴 {top_k}개를 찾아주세요:

{menu_list}

다음 JSON 형식으로 응답해주세요:
{{
    "matches": [
        {{
            "index": 1,
            "name": "메뉴명",
            "similarity_reason": "유사한 이유 설명",
            "confidence": 0.95
        }}
    ]
}}

유사도 판단 기준:
1. 음식 종류 (한식, 중식, 양식, 일식 등)
2. 조리 방법 (구이, 튀김, 볶음, 찜 등)
3. 주요 재료 (고기, 해산물, 채소 등)
4. 맛의 특성 (매운맛, 단맛, 신맛 등)
5. 음식 형태 (국, 반찬, 면, 밥 등)

confidence는 0.0~1.0 사이의 값으로, 높을수록 더 유사함을 의미합니다.
"""
        return prompt
    
    def _parse_ai_response(self, response: str, top_k: int) -> List[Dict]:
        """AI 응답을 파싱하여 결과 반환"""
        try:
            # JSON 부분 추출
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning("AI 응답에서 JSON을 찾을 수 없습니다.")
                return []
            
            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)
            
            matches = parsed.get('matches', [])
            results = []
            
            for match in matches[:top_k]:
                index = match.get('index', 0) - 1  # 1-based to 0-based
                if 0 <= index < len(self.menu_data):
                    menu_item = self.menu_data[index]
                    results.append({
                        'menu': menu_item,
                        'name': match.get('name', ''),
                        'similarity_reason': match.get('similarity_reason', ''),
                        'confidence': match.get('confidence', 0.0)
                    })
            
            return results
            
        except json.JSONDecodeError as e:
            logger.error(f"AI 응답 파싱 실패: {e}")
            logger.error(f"응답 내용: {response}")
            return []
        except Exception as e:
            logger.error(f"응답 처리 중 오류: {e}")
            return []
    
    def get_menu_details(self, menu_item: Dict) -> str:
        """메뉴 상세 정보 문자열 생성"""
        if isinstance(menu_item, dict):
            name = menu_item.get('name', menu_item.get('menu_name', '알 수 없음'))
            category = menu_item.get('category', '')
            price = menu_item.get('price', '')
            description = menu_item.get('description', '')
            
            details = f"메뉴: {name}"
            if category:
                details += f"\n카테고리: {category}"
            if price:
                details += f"\n가격: {price}"
            if description:
                details += f"\n설명: {description}"
            
            return details
        else:
            return str(menu_item) 