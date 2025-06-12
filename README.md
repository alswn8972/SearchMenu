# 메뉴 검색 시스템 (Menu Search System)

의미적 유사도 기반의 메뉴 검색 시스템
JSON 형식의 메뉴 데이터를 기반으로 벡터 임베딩을 활용한 유사도 검색을 제공

## 기술 스택 (Tech Stack)

### 핵심 기술
- **Python 3.9+**: 메인 프로그래밍 언어
- **sentence-transformers**: 벡터 임베딩을 위한 라이브러리
  - `jhgan/ko-sroberta-multitask`: 한국어에 특화된 다국어 모델
- **scikit-learn**: 유사도 계산을 위한 라이브러리
- **pandas**: 데이터 처리 및 결과 정렬
- **numpy**: 수치 연산

### 프로젝트 구조
```
├── config.py           # 설정 파일 (모델, 가중치 등)
├── embeddings.py       # 벡터 임베딩 관리 모듈
├── menu_processor.py   # 메뉴 데이터 처리 모듈
├── search_engine.py    # 검색 엔진 코어
├── main.py            # 메인 실행 파일
├── ia-data.json       # 메뉴 데이터
└── README.md          # 프로젝트 문서
```

## 주요 기능

1. **벡터 임베딩**
   - 메뉴 데이터를 고차원 벡터로 변환
   - 한국어에 최적화된 모델 사용
   - 의미적 유사도를 보존하는 벡터 표현
   - 확장 가능한 임베딩 구조

2. **유사도 검색**
   - 전체 유사도 (50% 가중치)
   - 페이지명 유사도 (30% 가중치)
   - 컨텍스트 유사도 (20% 가중치)

3. **검색 결과**
   - 상위 5개 유사 항목 제공
   - 각 항목별 상세 유사도 점수 표시
   - 카테고리, 서비스, 페이지명 정보 포함

## 설치 방법

1. Python 3.9 이상 설치
2. 필요한 패키지 설치:
```bash
pip install sentence-transformers
pip install scikit-learn
pip install pandas
pip install numpy
```

## 사용 방법

1. 프로그램 실행:
```bash
python main.py
```

2. 검색어 입력
3. 검색 결과 확인
4. 종료하려면 'q' 입력

## 검색 결과 예시
```
검색어: 카드 이용내역

유사한 항목들:

1. 전체 유사도: 0.9725
   - 페이지별 유사도: 1.0000
   - 전체 컨텍스트 유사도: 0.8710
   - 종합 점수: 0.9448
   카테고리: 결제
   서비스: 카드관리
   페이지명: 카드관리
```

## 모듈 설명

### config.py
- 모델 설정
- 유사도 가중치 설정
- 검색 결과 수 설정

### embeddings.py
- 벡터 임베딩 생성
- 유사도 계산
- 모델 관리

### menu_processor.py
- JSON 데이터 로드
- 데이터 전처리
- 가중치 계산

### search_engine.py
- 검색 로직 구현
- 결과 정렬
- 데이터프레임 생성

### main.py
- 사용자 인터페이스
- 결과 출력 포맷팅
- 프로그램 실행 관리 