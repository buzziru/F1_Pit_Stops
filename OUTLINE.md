# Kaggle ML 프로젝트 시작

## 역할
너는 Kaggle Grandmaster 수준의 ML 엔지니어이자 프로젝트 아키텍트야.

## 프로젝트 개요
바이브 코딩 방식으로 Kaggle 컴피티션에 참가한다.
아래 워크플로우와 문서 구조를 기반으로 프로젝트를 설계해줘.

## 워크플로우
1. EDA: Jupyter MCP Server 활용 (eda.ipynb)
2. 피처 엔지니어링 / 모델링: .py 중심 작업 (src/)
3. 실행 환경: Kaggle Notebook (GPU)
4. 실험 결과: 구조화 JSON 로그로 관리

## 문서 구조
CLAUDE.md, docs/eda.md, docs/feature_engineering.md, docs/modeling.md

## 토큰 절약 원칙
- DataFrame 출력은 .head(5) / .shape / .dtypes / .isnull().sum() 만 허용
- 플롯은 EDA 단계에서만 생성, 이후엔 수치 요약으로 대체
- 플롯 생성 후 즉시 plt.close() 호출

## 컴피티션 정보
- URL: [내일 붙여넣기]
- 데이터 설명: [내일 붙여넣기]

## 요청
위 내용을 바탕으로 프로젝트 설계에 필요한 질문지를 만들어줘.
질문은 구체적이고 답변 후 즉시 문서 작성이 가능한 수준으로 구성해.