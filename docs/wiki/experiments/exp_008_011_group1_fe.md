# 실험 회고 — 1번 그룹 파생 피처 (exp_008 / 009 / 010 / 011)

> 2026-06-03 · 이슈 #7 연관 · 결과: **A·B·A+B 모두 기각, baseline 유지(코드 revert)** · 관련 결정 [[decisions]] #009 #010

## 배경 / 가설
exp_004(Driver OOF TE, 0.94952) 위에 EDA 기반 파생 2종을 얹어 추가 이득을 노림.
- **A) `TyreLife_LifeFrac`** = `TyreLife / Compound기대수명`. 컴파운드별 "수명 소진 비율"(1.0 부근=피트 임박)로 공통 축화. 기대수명은 **도메인 상수**(SOFT15/MED22/HARD30/INT20/WET15, 타깃 미사용).
- **B) `CumDeg_Delta`** = 그룹 `(Race,Year,Driver)` 내 `Cumulative_Degradation.diff()`(과거 랩만). per-lap 한계 열화(열화 가속도).

## 설정
- 모델/CV 동일: LightGBM, StratifiedKFold 5-fold, seed=42, `features=driver_te`.
- 비교 기준: exp_004 OOF **0.94952**, fold std≈0.0007.
- ablation: `conf/features` 의 `drop_cols` 노브로 A/B 격리. exp_008=둘다 drop(재현), 009=A만, 010=B만, 011=A+B.
- 누수 검증 통과: A 는 `COMPOUND_EXPECTED_LIFE` 가 `PitNextLap` 미참조, B 는 미래 랩 마스킹 후 prefix 불변(past-only) + 그룹 첫행 NaN(=그룹 수 40,869).

## 결과
| 실험 | 구성 | OOF AUC | Δ vs base | 판정 |
|---|---|---|---|---|
| exp_008 | baseline 재현(둘 다 drop) | 0.949522 | +0.000002 | 재현 OK |
| exp_009 | +A `TyreLife_LifeFrac` | 0.949358 | −0.00016 | 기각 |
| exp_010 | +B `CumDeg_Delta` | 0.949172 | −0.00035 | 기각 |
| exp_011 | +A+B | 0.949095 | −0.00043 | 기각 |

(fold std ≈ 0.0004~0.0007, 셋 다 노이즈 이내~동급의 마이너스)

## 기각 원인 분석 (메커니즘)
### A — 트리가 **불변(invariant)**인 재매개화
- LifeFrac 은 Compound 가지마다 TyreLife 축을 **상수배 스케일**할 뿐. **GBDT 분할은 스케일 불변** → SOFT의 `TyreLife>18` = `LifeFrac>1.2` 로 동일 분할. 트리는 `Compound`(native cat) 분기 안에서 TyreLife 를 데이터-최적 임계로 이미 나눔 → **새 분할력 0**.
- "컴파운드 공통 축" 이점은 단일 전역 임계를 쓰는 모델(선형)에서만 유효. GBDT 는 컴파운드별 임계를 공짜로 만들어 트리가 안 가진 문제를 푼 셈.
- ⚠️ **기대수명 상수를 더 잘 맞춰도 무효** — 완벽한 상수조차 단조 스케일이라 트리엔 불변. 이 방향은 구조적으로 막힘.

### B — 블랙박스 컬럼의 **노이즈 미분** (A보다 더 악화)
- B 는 "이전 행 참조"라 진짜 새 정보인데도 더 나쁨. 이유:
  - **이중 노이즈**: `Cumulative_Degradation` 은 S6E5 에서 정의 재현 불가(corr≈0)한 블랙박스. 그 1차 차분은 노이즈를 **증폭**.
  - **서브샘플이 per-lap 의미 파괴**: 연속 관측행 간격이 들쭉날쭉(1랩 vs 5랩) → diff 가 일관된 "가속도"가 아님.
  - **타깃은 레벨에 반응, 미분엔 거의 무반응**: 피트는 절대 마모 상태와 연동, 그건 raw 가 이미 제공.
- → A=깨끗한 컬럼의 결정적 재포장(중립), B=더러운 컬럼의 노이즈 미분(중립+잡음) → **B 가 더 마이너스**.

### A+B — 상호작용 아닌 **가산 희석**
- 시너지 없이 중복·노이즈 두 축 누적 → 분할 후보만 늘려 미세 희석. 곱이 아닌 합으로 악화(−0.00043).

## 결론 / 교훈
- 일반 법칙은 [[decisions]] #010 으로 승격: **GBDT 파생 피처는 트리가 raw 에서 split 으로 못 뽑는 정보를 줄 때만 채택.**
- 무용 유형(이번 A/B): 단조 재스케일, 기존 컬럼의 단순 비율/차분. 유용 유형(exp_004): 희소 고카디널리티 정규화 인코딩.
- 남은 #7 후보(TyreLife/LapNumber 비율, cumcount, 직전 LapTime/Position 변화)도 같은 두 함정에 속해 기대값 낮음 → **#7 보류**(이슈 오픈 유지, 미실험).
- 방법론 신뢰: baseline 정확 재현 + 누수 검증 통과 → null 은 측정오차/누수착시 아닌 진짜 무신호.
