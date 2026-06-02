# Kaggle 프로젝트 설계 질문지

> OUTLINE.md 기반. 답변 직후 `CLAUDE.md`, `docs/*.md`, `src/` 골격, JSON 로거 작성으로 이어진다.
> 표기: 🟢 = 컴피티션 정보 없이 지금 답변 가능 / 🔵 = 내일 데이터 확보 후 답변

---
컴피티션 링크 : https://www.kaggle.com/competitions/playground-series-s6e5

## A. 컴피티션 기본 정보 ✅ (2026-06-02 데이터 확인 완료)

1. **문제 유형**: 이진 분류 (F1 다음 랩 피트스톱 여부 예측)
2. **평가 지표(Metric)**: ROC-AUC → 제출은 **확률값**
3. **데이터 규모**:
   - train: **439,140 행 × 16 컬럼**
   - test: **188,165 행 × 15 컬럼**
   - 결측치: **없음** (train 전 컬럼 0)
4. **제출 형식**: `id, PitNextLap` (PitNextLap = 양성 확률) / Kaggle CLI 제출
5. **타깃 변수 `PitNextLap`**: 단일 이진. **불균형 (positive rate 19.9%, ≈1:4)**

### 컬럼 상세
- **범주형(object)**: `Driver`(nunique 887, 고카디널리티), `Compound`(5), `Race`(26)
- **범주형(int, 저카디널리티)**: `Year`(2022~2025, 4), `PitStop`(0/1), `Stint`(1~8)
- **수치형(float/int)**: `LapNumber, TyreLife, Position, LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change`
- **식별자**: `id`

### 데이터 구조 & 누수 분석 🔑
- `(Race, Year, Driver)` 그룹 = 한 스틴트의 랩 시퀀스 (train 기준 40,869그룹, 평균 10.7행/그룹)
- train/test가 **같은 (Race,Year) 104개를 100% 공유**, test 그룹의 **96%(35,674/37,038)가 train에도 존재**
- → 같은 레이스-드라이버의 랩이 train/test에 나뉘는 **row-level split**
- → **GroupKFold 불필요. StratifiedKFold가 대회 셋업과 일치** (B번 확정과 부합)
- ⚠️ 단, `LapTime_Delta`, `Cumulative_Degradation`, `Position_Change` 등은 시퀀스 파생 피처일 수 있어 **미래 정보 누수 여부 EDA에서 점검** 필요

## B. 검증 전략 (CV) — 설계 핵심 🟢

6. **CV 분할 방식**: StratifiedKFold (AUC 이진분류 → 클래스 비율 유지)
7. **Fold 수**: 5-fold 
8. **seed 정책**: 마지막 seed averaging 이전 단일 seed 전략 

## C. 모델링 방향 ✅

9. **1차 베이스라인 모델**: LightGBM (범주형 `Driver/Compound/Race`는 LGBM native categorical, 고카디널리티 `Driver`는 추후 target encoding 검토)
10. **앙상블 계획**: 단일 모델 → 스태킹/블렌딩까지? 목표 범위? *(추후 확정)*
11. **GPU 필요성**: 베이스라인은 CPU LGBM. 훈련시간 관찰 후 XGB/CatBoost 시 GPU 고려
    - 불균형(19.9%) 대응: `scale_pos_weight` 또는 `is_unbalance` 고려, metric은 AUC라 임계값 튜닝 불필요

## D. 실행 환경 & 도구 🔵

12. **로컬 vs Kaggle 역할 분담**: EDA/피처는 로컬, 학습은 Kaggle
13. **의존성 관리**: `uv`
14. **실험 추적**: 로그 파일 생성. W&B 연동

## E. 실험 로그(JSON) 스키마 🟢

15. 로그 필드 확정. 제안 스키마:
    ```json
    {
      "exp_id": "exp_001",
      "timestamp": "2026-06-03T10:00:00",
      "model": "lgbm",
      "features": ["f1", "f2"],
      "cv_strategy": "StratifiedKFold_5",
      "cv_scores": [0.81, 0.82],
      "cv_mean": 0.815,
      "cv_std": 0.004,
      "lb_score": null,
      "params": {},
      "notes": ""
    }
    ```
    → 확정: 저장 경로 `experiments/logs/*.json`, OOF `experiments/oof/`, 제출 `experiments/submissions/`

## F. 프로젝트 구조 🟢

16. `src/` 모듈 분리 제안 — 동의?
    ```
    src/
      config.py        # 경로, seed, 상수
      data.py          # 로드/IO
      features.py      # 피처 엔지니어링
      cv.py            # 검증 분할
      train.py         # 학습 루프
      predict.py       # 추론/제출
      utils.py         # 로깅(JSON), 공통
    ```
17. **CLAUDE.md 범위**: 토큰 절약 원칙 + 워크플로우 + 코딩 컨벤션 모두 포함

## G. 협업 규칙 🟢

18. **함수 컨벤션**: 타입힌트 필수. Google 스타일 docstring. 함수당 ~50줄 권장.
19. **재현성**: 모든 실험 seed 고정 + 커밋 해시 로깅 의무화

---
- 캐글 api 사용 위한 토큰은 kaggle.json에 있음

### 답변 우선순위
- **먼저 (지금)**: B, E, F, G → `CLAUDE.md` + `src/` 골격 + JSON 로거 작성 가능
- **나중 (내일 데이터 확인 후)**: A, C, D
