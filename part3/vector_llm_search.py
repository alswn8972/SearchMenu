#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
import hashlib
import pickle
import os

class VectorLLMSearch:
    """벡터 임베딩 + LLM 2단계 검색 시스템"""
    
    def __init__(self):
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.embeddings_cache = {}
        self.cache_file = "embeddings_cache.pkl"
        self.load_cache()
    
    def load_cache(self):
        """임베딩 캐시 로드"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                print(f"✅ 임베딩 캐시 로드됨 ({len(self.embeddings_cache)}개)")
        except Exception as e:
            print(f"캐시 로드 실패: {e}")
            self.embeddings_cache = {}
    
    def save_cache(self):
        """임베딩 캐시 저장"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            print(f"✅ 임베딩 캐시 저장됨 ({len(self.embeddings_cache)}개)")
        except Exception as e:
            print(f"캐시 저장 실패: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트의 임베딩 벡터를 가져옵니다."""
        # 캐시 확인
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding
            self.embeddings_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            print(f"임베딩 생성 실패: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        if not vec1 or not vec2:
            return 0.0
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def vector_search(self, query: str, menu_data: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """1단계: 벡터 임베딩 기반 검색"""
        print("🔍 1단계: 벡터 임베딩 검색 수행 중...")
        
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        results = []
        # 검색 범위를 늘려서 더 많은 후보 확보
        search_data = menu_data[:500] if len(menu_data) > 500 else menu_data
        
        for item in search_data:
            if isinstance(item, dict):
                # 메뉴 이름 추출
                menu_name = self._extract_menu_name(item)
                if menu_name:
                    # 키워드 매칭 우선 확인
                    keyword_score = self._keyword_matching_score(query, menu_name)
                    
                    # 메뉴 이름의 임베딩 계산
                    menu_embedding = self.get_embedding(menu_name)
                    if menu_embedding:
                        vector_similarity = self.cosine_similarity(query_embedding, menu_embedding)
                        
                        # 키워드 매칭이 있으면 우선 선택
                        if keyword_score > 0:
                            final_score = keyword_score * 0.7 + vector_similarity * 0.3
                        else:
                            # 키워드 매칭이 없으면 벡터 유사도가 높은 것만 선택
                            final_score = vector_similarity
                            if final_score < 0.3:  # 임계값을 더 엄격하게
                                continue
                        
                        results.append({
                            'menu_name': menu_name,
                            'menu_data': item,
                            'vector_score': final_score,
                            'keyword_score': keyword_score,
                            'vector_similarity': vector_similarity
                        })
        
        # 점수 순으로 정렬
        results.sort(key=lambda x: x['vector_score'], reverse=True)
        print(f"✅ 벡터 검색 완료: {len(results)}개 결과 발견")
        return results[:top_k]
    
    def _keyword_matching_score(self, query: str, menu_name: str) -> float:
        """키워드 매칭 점수 계산"""
        query_words = set(query.lower().split())
        menu_words = set(menu_name.lower().split())
        
        # 완전 일치
        if query.lower() in menu_name.lower():
            return 1.0
        
        # 부분 일치
        common_words = query_words.intersection(menu_words)
        if common_words:
            return len(common_words) / max(len(query_words), len(menu_words))
        
        return 0.0
    
    def _extract_menu_name(self, item: Dict[str, Any]) -> Optional[str]:
        """메뉴 데이터에서 페이지명 추출"""
        # 우선순위에 따라 페이지명 키 확인
        name_keys = ['page_name', 'name', 'menu_name', 'title', 'page_title']
        
        for key in name_keys:
            if key in item and item[key]:
                name = str(item[key]).strip()
                if name and name != " " and len(name) > 0:
                    return name
        
        return None
    
    def llm_refinement(self, query: str, vector_results: List[Dict[str, Any]], max_results: int = 5) -> List[Dict[str, Any]]:
        """2단계: LLM을 통한 검색 결과 정교화"""
        print("🤖 2단계: LLM 정교화 수행 중...")
        
        if not vector_results:
            return []
        
        # 상위 10개만 LLM에 전달 (더 정확한 결과를 위해)
        top_candidates = vector_results[:10]
        
        # 후보 텍스트 생성
        candidates_text = ""
        for i, result in enumerate(top_candidates, 1):
            menu_name = result['menu_name']
            score = result['vector_score']
            keyword_score = result.get('keyword_score', 0)
            candidates_text += f"{i}. {menu_name} (총점: {score:.3f}, 키워드: {keyword_score:.3f})\n"
        
        # LLM 프롬프트 생성
        prompt = self._create_refinement_prompt(query, candidates_text, max_results)
        
        try:
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "당신은 메뉴 검색 전문가입니다. 사용자의 검색 의도를 정확히 파악하고 관련성 높은 메뉴만 선택해주세요. 관련성이 낮은 메뉴는 절대 선택하지 마세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 더 일관된 결과를 위해 낮춤
                max_tokens=500
            )
            
            llm_response = response.choices[0].message.content.strip()
            print(f"🤖 LLM 응답: {llm_response}")
            
            # JSON 파싱
            refined_results = self._parse_llm_response(llm_response, vector_results)
            
            # 유사도가 0.4 이상인 결과만 최종 반환 (완화)
            final_results = [r for r in refined_results if r.get('similarity_score', 0) >= 0.4]
            
            # 결과가 없으면 벡터 유사도 상위 5개라도 무조건 출력
            if not final_results:
                print("⚠️ LLM 필터를 통과한 결과가 없어, 벡터 유사도 상위 5개를 강제 출력합니다.")
                return vector_results[:5]
            
            print(f"✅ LLM 정교화 완료: {len(final_results)}개 최종 결과")
            return final_results[:max_results]
            
        except Exception as e:
            print(f"❌ LLM 정교화 오류: {e}")
            # LLM 실패 시 키워드 매칭이 있는 결과만 반환, 그래도 없으면 벡터 상위 5개라도 출력
            keyword_results = [r for r in vector_results if r.get('keyword_score', 0) > 0]
            if not keyword_results:
                print("⚠️ LLM 오류 및 키워드 매칭 결과 없음, 벡터 유사도 상위 5개 강제 출력")
                return vector_results[:5]
            return keyword_results[:max_results]
    
    def _format_candidates_for_llm(self, search_results: List[Dict[str, Any]]) -> str:
        """검색 결과를 LLM용 텍스트로 변환"""
        formatted_items = []
        
        for i, result in enumerate(search_results, 1):
            menu_name = result['menu_name']
            menu_data = result['menu_data']
            vector_score = result['vector_score']
            
            # 상세 정보 수집
            details = []
            if 'Category' in menu_data:
                details.append(f"카테고리: {menu_data['Category']}")
            if 'Service' in menu_data:
                details.append(f"서비스: {menu_data['Service']}")
            if 'hierarchy' in menu_data:
                details.append(f"계층: {' > '.join(menu_data['hierarchy'])}")
            
            detail_text = f" ({', '.join(details)})" if details else ""
            formatted_items.append(f"{i}. {menu_name} (벡터점수: {vector_score:.3f}){detail_text}")
        
        return "\n".join(formatted_items)
    
    def _create_refinement_prompt(self, query: str, candidates_text: str, max_results: int) -> str:
        """LLM 정교화용 프롬프트 생성"""
        return f"""
사용자가 "{query}"를 검색했습니다.

다음은 벡터 검색으로 찾은 상위 후보들입니다:

{candidates_text}

위 후보들 중에서 사용자 검색어 "{query}"와 **실제로 관련이 있는** {max_results}개의 페이지/메뉴만 선택해주세요.

**엄격한 매칭 기준:**
1. **직접적인 단어 포함**: 검색어가 페이지명에 직접 포함되어야 함
2. **기능적 연관성**: 검색어와 페이지 기능이 실제로 연관되어야 함
3. **의미적 일치**: 검색어의 의미와 페이지 기능이 일치해야 함

**절대 제외할 것:**
- 검색어와 전혀 관련 없는 페이지
- 단순히 비슷한 단어만 있는 페이지
- 기능적으로 연관되지 않은 페이지
- 추상적이거나 모호한 연관성

**선택한 메뉴들을 JSON 형식으로 반환해주세요:**
[
  {{
    "menu_name": "페이지명",
    "similarity_score": 0.95,
    "reason": "구체적인 연관성 이유"
  }}
]

유사도 점수는 0.0~1.0 사이로 주세요. 
- 완전 일치: 0.9~1.0
- 높은 연관성: 0.7~0.8
- 중간 연관성: 0.5~0.6
- 낮은 연관성: 0.3~0.4
- 관련 없음: 0.0~0.2 (선택하지 마세요)
"""
    
    def _parse_llm_response(self, content: str, vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM 응답을 파싱하여 결과 리스트로 변환"""
        try:
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                llm_results = json.loads(json_str)
                
                final_results = []
                for llm_result in llm_results:
                    if isinstance(llm_result, dict) and 'menu_name' in llm_result:
                        menu_name = llm_result['menu_name']
                        
                        # 벡터 검색 결과에서 해당 메뉴 찾기
                        for vector_result in vector_results:
                            if vector_result['menu_name'] == menu_name:
                                final_result = {
                                    'menu_name': menu_name,
                                    'menu_data': vector_result['menu_data'],
                                    'vector_score': vector_result['vector_score'],
                                    'llm_similarity': float(llm_result.get('similarity_score', 0.0)),
                                    'llm_reason': str(llm_result.get('reason', 'LLM 매칭')),
                                    'final_score': vector_result['vector_score'] * float(llm_result.get('similarity_score', 0.0))
                                }
                                final_results.append(final_result)
                                break
                
                final_results.sort(key=lambda x: x['final_score'], reverse=True)
                return final_results
            else:
                print("LLM 응답에서 JSON을 찾을 수 없습니다.")
                return []
                
        except json.JSONDecodeError as e:
            print(f"LLM 응답 JSON 파싱 오류: {e}")
            return []
        except Exception as e:
            print(f"LLM 응답 파싱 중 오류 발생: {e}")
            return []
    
    def search(self, query: str, menu_data: List[Dict[str, Any]], max_results: int = 5) -> List[Dict[str, Any]]:
        """벡터 임베딩 + LLM 2단계 검색 수행"""
        print(f"🔍 '{query}' 2단계 검색 시작...")
        print("-" * 50)
        
        # 1단계: 벡터 검색
        vector_results = self.vector_search(query, menu_data, top_k=20)
        
        if not vector_results:
            print("❌ 벡터 검색 결과가 없습니다.")
            return []
        
        # 2단계: LLM 검색 결과 정교화
        refined_results = self.llm_refinement(query, vector_results, max_results)
        
        # 캐시 저장
        self.save_cache()
        
        return refined_results
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """검색 결과를 보기 좋게 포맷팅"""
        if not results:
            return "유사한 메뉴를 찾을 수 없습니다."
        
        formatted_lines = ["🔍 최종 검색 결과 (벡터 + LLM):"]
        
        for i, result in enumerate(results, 1):
            menu_name = result['menu_name']
            vector_score = result['vector_score']
            llm_similarity = result['llm_similarity']
            final_score = result['final_score']
            reason = result['llm_reason']
            
            menu_data = result['menu_data']
            
            formatted_lines.append(f"\n{i}. {menu_name}")
            formatted_lines.append(f"   벡터 점수: {vector_score:.3f}")
            formatted_lines.append(f"   LLM 연관도: {llm_similarity:.3f}")
            formatted_lines.append(f"   최종 점수: {final_score:.3f}")
            formatted_lines.append(f"   연관성 이유: {reason}")
            
            # 상세 정보 표시
            if 'Category' in menu_data:
                formatted_lines.append(f"   카테고리: {menu_data['Category']}")
            if 'Service' in menu_data:
                formatted_lines.append(f"   서비스: {menu_data['Service']}")
            if 'hierarchy' in menu_data:
                formatted_lines.append(f"   계층: {' > '.join(menu_data['hierarchy'])}")
        
        return "\n".join(formatted_lines)
    
    def display_results(self, results: List[Dict[str, Any]], query: str):
        """검색 결과 표시"""
        print(f"\n🔍 '{query}' 검색 결과:")
        print("=" * 60)
        
        if not results:
            print("❌ 관련된 메뉴를 찾을 수 없습니다.")
            return
        
        for i, result in enumerate(results, 1):
            menu_name = result.get('menu_name', '알 수 없음')
            similarity_score = result.get('similarity_score', result.get('vector_score', 0))
            reason = result.get('reason', '')
            menu_data = result.get('menu_data', {})
            
            print(f"\n{i}. 📄 {menu_name}")
            print(f"   점수: {similarity_score:.3f}")
            
            # 카테고리 정보 표시
            if 'Category' in menu_data and menu_data['Category']:
                print(f"   카테고리: {menu_data['Category']}")
            
            # 서비스 정보 표시
            if 'Service' in menu_data and menu_data['Service']:
                print(f"   서비스: {menu_data['Service']}")
            
            # 계층 정보 표시
            if 'hierarchy' in menu_data and menu_data['hierarchy']:
                hierarchy_str = ' > '.join(menu_data['hierarchy'])
                print(f"   계층: {hierarchy_str}")
            
            # 연관성 이유 표시
            if reason:
                print(f"   이유: {reason}")
            
            print("-" * 40) 