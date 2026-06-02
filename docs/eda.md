# EDA — Playground Series S6E5

> `eda.ipynb` 작업 결과를 **수치 요약** 중심으로 정리한다. 플롯은 노트북에만, 여기엔 결론만.

## 1. 데이터 개요 (2026-06-02 확인)
| | rows | cols |
|---|---|---|
| train | 439,140 | 16 |
| test | 188,165 | 15 (target 제외) |
| sample_submission | 188,165 | 2 (`id`, `PitNextLap`) |

- **결측치**: train 전 컬럼 0 (test 미확인 → EDA에서 확인)

## 2. 타깃 `PitNextLap`
- 이진 (0/1), **양성률 19.9%** (0: 351,759 / 1: 87,381) → 불균형 ≈ 1:4
- 대응: LGBM `is_unbalance=True` 또는 `scale_pos_weight≈4`. AUC 지표라 임계값 튜닝 불필요.

## 3. 컬럼 분류
- **범주형(object)**: `Driver`(887, 고카디널리티), `Compound`(5), `Race`(26)
- **범주형(int)**: `Year`(2022~2025), `PitStop`(0/1), `Stint`(1~8)
- **수치형**: `LapNumber, TyreLife, Position, LapTime (s), LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change`

## 4. 그룹/누수 분석 🔑
- 시퀀스 단위 = `(Race, Year, Driver)` (한 스틴트의 랩 시퀀스), train 40,869그룹·평균 10.7행
- train/test **(Race,Year) 104개 100% 공유**, test 그룹 **96%가 train 에도 존재**
- → **row-level split** → CV는 **StratifiedKFold** (GroupKFold 불필요)
- ⚠️ `LapTime_Delta`, `Cumulative_Degradation`, `Position_Change` 가 그룹 내 미래 정보를 포함하는지 점검 필요 (누수면 피처 재정의)

## 5. EDA 체크리스트 (eda.ipynb 에서 수행)
- [ ] test 결측치 확인
- [ ] 수치형 분포 / 이상치 (요약 통계로 대체)
- [ ] 타깃과 각 피처의 관계 (그룹별 양성률, `Compound`/`Stint`/`TyreLife` vs target)
- [ ] `Driver` 고카디널리티 처리 방향 (target encoding vs native categorical)
- [ ] train/test 분포 차이 (adversarial validation 고려)
- [ ] 파생 피처 누수 검증 (`shift` 기반 재현 가능 여부)
