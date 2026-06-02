# EDA — Playground Series S6E5

> `eda.ipynb` 작업 결과를 **수치 요약** 중심으로 정리한다. 플롯은 노트북에만, 여기엔 결론만.

## 1. 데이터 개요 (2026-06-02 확인)
| | rows | cols |
|---|---|---|
| train | 439,140 | 16 |
| test | 188,165 | 15 (target 제외) |
| sample_submission | 188,165 | 2 (`id`, `PitNextLap`) |

- **결측치**: train·test 전 컬럼 0 (둘 다 확인 완료)

## 2. 타깃 `PitNextLap`
- 이진 (0/1), **양성률 19.9%** (0: 351,759 / 1: 87,381) → 불균형 ≈ 1:4
- 대응: 지표가 순위 기반 ROC-AUC → **기본 `is_unbalance=False`** (가중은 실험 비교만, decisions #004). 임계값 튜닝 불필요.

## 3. 컬럼 분류
- **범주형(object)**: `Driver`(887, 고카디널리티), `Compound`(5), `Race`(26)
- **범주형(int)**: `Year`(2022~2025), `PitStop`(0/1), `Stint`(1~8)
- **수치형**: `LapNumber, TyreLife, Position, LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change`

## 4. 그룹/누수 분석 🔑
- 시퀀스 단위 = `(Race, Year, Driver)` (한 스틴트의 랩 시퀀스), train 40,869그룹·평균 10.7행
- train/test **(Race,Year) 104개 100% 공유**, test 그룹 **96%가 train 에도 존재**
- → **row-level split** → CV는 **StratifiedKFold** (GroupKFold 불필요)
- ⚠️ `LapTime_Delta`, `Cumulative_Degradation`, `Position_Change` 는 backward/forward shift로 재현 불가 (corr<0.25, LapNumber 비연속=부분샘플 구조). 미래 누수 직접 증거 없음. 단 정의 불명확, `LapTime_Delta` target corr=-0.005로 예측력 의문 → ablation 권장.

## 5. EDA 결과 (2026-06-02 완료, 이슈 #1)
- [x] **test 결측치**: 전 컬럼 0
- [x] **수치형 분포/이상치**: `LapTime (s)`·`LapTime_Delta`·`Cumulative_Degradation` 에 1~99%ile 외 ~2% 극단값 (`LapTime_Delta` -2403~2423). 트리 모델 영향 없음, 스케일 모델은 클리핑 필요
- [x] **타깃-피처 관계**:
  - target corr 상위: TyreLife(0.274) > LapNumber(0.267) > Stint(0.198) > RaceProgress(0.186) > Cumulative_Degradation(-0.167)
  - Compound 양성률: HARD 32.8% > SOFT 19.4% > INTERMEDIATE 15.2% > MEDIUM 10.1% > WET 2.5% (13배 차)
  - Stint: Stint 2가 39.1% 최고, Stint 1이 6.0% 최저
  - TyreLife: 1~5랩 5.1% → 31+랩 41.4% (단조 증가, 강력)
  - RaceProgress: 40~60% 구간 36.9% (피트 윈도우 집중), 초반 8.6%·후반 11.2%
- [x] **Driver(887)**: test Driver 801개 전원 train 존재 (coverage 100%), train only 86개. 양성률 std=0.099 → native categorical 우선, OOF TE와 실험 비교(exp_002 vs exp_003)
- [x] **train/test 드리프트**: Adversarial AUC=0.5012 (seed=42) → 드리프트 없음, StratifiedKFold 유효
- [x] **파생피처 누수**: shift 재현 불가(corr<0.25), 미래 누수 직접 증거 없음. `LapTime_Delta` 예측력 낮음(corr-0.005) → ablation 모니터링

### 피처 우선순위 (모델링 반영)
TyreLife · LapNumber · Stint · Compound · RaceProgress(구간화 후보). `LapTime_Delta` 는 ablation 으로 유지/제거 판단.
