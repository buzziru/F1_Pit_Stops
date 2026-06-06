# EDA — Playground Series S6E5

> 주제별 노트북(`notebooks/eda_*.ipynb`) 작업 결과를 **수치 요약** 중심으로 정리한다. 플롯은 노트북에만, 여기엔 결론만.

## 1. 데이터 개요 (2026-06-02 확인)


|                   | rows    | cols                   |
| ----------------- | ------- | ---------------------- |
| train             | 439,140 | 16                     |
| test              | 188,165 | 15 (target 제외)         |
| sample_submission | 188,165 | 2 (`id`, `PitNextLap`) |


- **결측치**: train·test 전 컬럼 0 (둘 다 확인 완료)

## 2. 타깃 `PitNextLap`

- 이진 (0/1), **양성률 19.9%** (0: 351,759 / 1: 87,381) → 불균형 ≈ 1:4
- 대응: 지표가 순위 기반 ROC-AUC → **기본 `is_unbalance=False`** (가중은 실험 비교만, decisions #004). 임계값 튜닝 불필요.

## 3. 컬럼 분류

- **범주형(object)**: `Driver`(887, 고카디널리티), `Compound`(5), `Race`(26)
- **범주형(int)**: `Year`(2022~~2025), `PitStop`(0/1), `Stint`(1~~8)
- **수치형**: `LapNumber, TyreLife, Position, LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change`

## 4. 그룹/누수 분석 🔑

- 시퀀스 단위 = `(Race, Year, Driver)` (한 스틴트의 랩 시퀀스), train 40,869그룹·평균 10.7행
- train/test **(Race,Year) 104개 100% 공유**, test 그룹 **96%가 train 에도 존재**
- → **row-level split** → CV는 **StratifiedKFold** (GroupKFold 불필요)
- ⚠️ `LapTime_Delta`, `Cumulative_Degradation`, `Position_Change` 는 backward/forward shift로 재현 불가 (corr<0.25, LapNumber 비연속=부분샘플 구조). 미래 누수 직접 증거 없음. 단 정의 불명확, `LapTime_Delta` target corr=-0.005로 예측력 의문 → ablation 권장.

## 5. EDA 결과 (2026-06-02 완료, 이슈 #1)

- **test 결측치**: 전 컬럼 0
- **수치형 분포/이상치**: `LapTime (s)`·`LapTime_Delta`·`Cumulative_Degradation` 에 1~~99%ile 외 ~2% 극단값 (`LapTime_Delta` -2403~~2423). 트리 모델 영향 없음, 스케일 모델은 클리핑 필요
- **타깃-피처 관계**:
  - target corr 상위: TyreLife(0.274) > LapNumber(0.267) > Stint(0.198) > RaceProgress(0.186) > Cumulative_Degradation(-0.167)
  - Compound 양성률: HARD 32.8% > SOFT 19.4% > INTERMEDIATE 15.2% > MEDIUM 10.1% > WET 2.5% (13배 차)
  - Stint: Stint 2가 39.1% 최고, Stint 1이 6.0% 최저
  - TyreLife: 1~5랩 5.1% → 31+랩 41.4% (단조 증가, 강력)
  - RaceProgress: 40~60% 구간 36.9% (피트 윈도우 집중), 초반 8.6%·후반 11.2%
- **Driver(887)**: test Driver 801개 전원 train 존재 (coverage 100%), train only 86개. 양성률 std=0.099 → native categorical 우선, OOF TE와 실험 비교(exp_002 vs exp_003)
- **train/test 분포 변화**: Adversarial AUC=0.5012 (seed=42) → 분포 변화 없음, StratifiedKFold 유효
- **파생피처 누수**: shift 재현 불가(corr<0.25), 미래 누수 직접 증거 없음. `LapTime_Delta` raw corr=-0.005로 낮지만 **|delta|≤0.3 안정 구간 피트율 2.6% vs 나머지 26.1%** (W자 비선형) → 이진/구간 피처로 변환 시 강력. `Cumulative_Degradation`: 10분위 단조성 corr=-0.83, <-50(피트 직후) 27.9% / -5~0 11.7%. 둘 다 구간화 가치 있음. (eda_02)

## 6. LapTime·열화 심층 (2026-06-02, eda_02)
- **retire(DNF) 판별**: 명시적 라벨 없음. DNF 프록시(MaxRP<0.85 등) 기준, 세 변수의 효과크기 작음(rank-biserial 최대 0.19=LapTime_Delta, Cumulative_Degradation rbc=-0.11). **세 변수만으로 retire 이진 판별은 불충분**. Degradation 극단(>50)이 그나마 신호. row-level split + 합성데이터로 프록시 노이즈 큼 → DNF 피처는 보류, `is_last_lap_in_group` 정도만 안전.
- **구간화 vs 타깃**:
  - `LapTime_Delta`: **W자(비단조)**. `|delta|≤0.3`=2.6%, 그 외=26.1% (10배). 10분위 spread 0.279
  - `Cumulative_Degradation`: 단조 corr=-0.83 (가장 강한 단조 신호). 단 Stint/TyreLife와 공선성 주의
  - `LapTime (s)`: spread 0.147, 단조 corr=-0.48, 이상치 구간 특별 신호 없음 (약함)

### 피처 우선순위 (모델링 반영)

## 7. 원본(src_raw) vs train 합성 변형 분석 (2026-06-06, eda_05)
- **LapNumber sparsity = 합성 산물 확정**: src consec_frac **0.9921**(dense, gap median 1) → train **0.3195**(sparse, gap median 3, span/n 3.73). 합성이 그룹당 ~3랩 간격으로 의도적 서브샘플링.
- **그룹 팽창**: src 1,838그룹(중앙 56랩) → train 36,829그룹(중앙 12랩). Driver **31명(실제)→887명(합성 ID 분할)**, train 중 실제 Driver 행 6.9%뿐.
- **타깃률**: src 25.48% → train 19.90%(−5.58%p). 2023 아티팩트(train 0.96%, src 3.08%)는 sparse sampling 시 **피트 실시 랩(LapNumber+1) 71.5% 누락**으로 PitNextLap==1 손실 — 2023 특이처리가 아니라 샘플링 부작용.
- **inherited 파생피처 훼손**: `LapTime_Delta`·`Cumulative_Degradation`·`Position_Change`는 원본(dense)에서 계산된 값을 합성행에 상속. sparse 후 **연속랩 차분 의미 붕괴** — `LapTime_Delta` mean src −0.20 → train −3.77. 컬럼차는 `Normalized_TyreLife`(src only, 누수 의도 드롭)·`id`(train only)뿐.
- **함의**: ① 연속 delta/slope 불신(원칙4 근거 확정) ② **`is_consec_lap`/lap-gap 마스크 피처**(inherited delta 신뢰구간을 트리에 알림 — raw 없는 새 축) 유망 ③ 증강 weight 는 분포차에도 OOF 실측상 w1.0 최선(exp_013~015) 유지.
- 노트북: `notebooks/eda_05_source_vs_train_drift.ipynb`

TyreLife · LapNumber · Stint · Compound · RaceProgress(구간화 후보). **`is_stable_delta`(|LapTime_Delta|≤0.3 이진): 피트율 2.6% vs 26.1% — feature-smith 1순위 후보.** `Cumulative_Degradation` 구간/클리핑 피처도 ablation 가치. `LapTime (s)` 단독은 약함.