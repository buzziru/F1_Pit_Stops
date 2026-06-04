# 스태킹 전환 계획 (M4 블렌딩 → 메타러너) — 2026-06-04

> 권고: 손튜닝 블렌드 가중(균등 vs 가중) 논쟁을 **CV'd 메타러너**로 데이터 기반 종결하고,
> 쌓아둔 다양성(LGBM/XGB/CatBoost/RealMLP)을 짜낸다. 관련 [[decisions]] #015·#017·#006.
> ⚠️ 채택/기각 시 ADR 추가. 결정 기준은 §6.

## 1. 왜 스태킹
- 현재 3-way **균등 vs 최적가중**을 손으로 비교 중(#017): 균등 0.951642, 가중은 OOF 과적합 우려로 보류. RealMLP(exp_023) 편입 가중(0.10~0.15)도 미결.
- 메타러너는 **같은 5-fold로 가중을 CV 학습** → 균등/가중/모델선택을 한 번에 원칙적으로 푼다. 균등 블렌드는 "모든 가중=1/n"인 특수해이므로 **스태킹은 균등을 포함·일반화**(못 이기면 균등 유지).

## 2. 전제 (충족)
- 모든 base OOF가 **동일 StratifiedKFold(seed=42, 5-fold)** 정렬 — 스태킹 정합의 필수조건. exp_016/019/022/023(+025) 모두 충족(#016).
- base OOF는 **행별 누수 없음**(행 i 예측은 i 미포함 모델에서) → 메타 피처로 안전.

## 3. 설계
- **Level-0 (base, 이미 보유)**: 각 모델의 OOF(train) + test 예측(submission CSV).
  | 후보 | OOF | corr(vs others) | 비고 |
  |---|---|---|---|
  | exp_016 LGBM(driver_te+aug) | 0.950959 | LGBM-XGB 0.9944 | |
  | exp_019 XGB | 0.951090 | LGBM-CAT 0.9854 | 단독 최고 |
  | ~~exp_022 CatBoost native~~ | 0.949811 | — | **exp_025 로 대체**(아래) |
  | exp_023 RealMLP | 0.944154 | vs GBDT Spearman **0.90** | 최약·**최저상관**(편입 가치) |
  | **exp_025 CatBoost year-cat** | **0.950043** | vs exp_022 **0.993** | exp_022 대체(+0.00023·동일상관). CatBoost 대표 1개만 |
  | LGBM-tuned(Optuna) | (대기) | ~exp_016 높을 듯 | 둘 중 우수 1개 |
- **Level-1 (meta)**: base 예측을 피처로 타깃 예측. **같은 5-fold로 CV 학습** → meta-OOF.
- **최종 test**: 전체 base-OOF로 meta 재적합 → base **test 예측**에 적용.

## 4. 누수/정합 보장
1. base-OOF는 leak-free(§2) → 그 위 메타 CV(동일 split)도 leak-free. meta-OOF AUC = 무편향 추정.
2. 메타 CV split = base와 **동일 seed=42** (`cv.get_folds`). fold당: train-fold base-OOF로 meta fit → valid-fold 예측.
3. base test 예측은 각 모델 5-fold 평균(기존 파이프라인) → meta 재적합본을 그대로 적용.

## 5. 메타러너 선택 (단순·정규화 우선)
- ⚠️ **GBDT 메타 금지** — 피처 4~6개에 과적합, 이미 보유 알고리즘.
- 1순위: **비음수 제약 선형 블렌드**(weights≥0, Σw=1, OOF AUC 최대화). 균등의 일반화, 가중 해석 가능, 안정.
- 2순위: **Logistic Regression(L2)** on base **logit**(prob→logit) — 표준 스태킹.
- 대조: **rank-mean**(AUC=순위 → 비모수 평균), 3-way 균등(현 최고).
- collinearity(LGBM-XGB 0.994): 비음수 제약/​L2가 흡수. corr 0.99↑ 쌍은 한쪽 가중 0 수렴 예상 → 사후 가지치기.

## 6. 판정 프로토콜
- 비교축: **meta-OOF AUC** vs 3-way 균등 **0.951642**(+ 4-way 균등 0.951708).
- 채택: meta-OOF가 균등을 **fold std(~0.0008~0.002) 넘는 마진**으로 상회 AND 가중이 극단(단일모델 쏠림)이 아닐 때. OOF≈Private(#006) 신뢰, **Public 단일점 무시**.
- 동률이면 **균등 유지**(Occam·Public갭). 마일스톤이면 제출.
- 부산물: base별 meta 가중 = 각 모델의 한계기여 정량화(RealMLP "편입 가치" 논쟁 종결).

## 7. 구현 (`src/stack.py`, ~100줄)
```
load: experiments/oof/{ids}.csv (id,oof) + train target  →  X_oof(n×k), y
      experiments/submissions/{ids}.csv (id,PitNextLap)   →  X_test(m×k)
meta CV: for fold in cv.get_folds(y, seed=42):
            fit meta on X_oof[tr], y[tr]; pred X_oof[va] → meta_oof[va]
meta_oof_auc = roc_auc_score(y, meta_oof)
refit: meta.fit(X_oof, y); final = meta.predict(X_test)
save: experiments/oof/stack_*.csv, submissions/stack_*.csv; report weights + AUC vs baselines
```
- 메타러너는 sklearn(LogisticRegression / 비음수 최소제곱 or scipy.optimize 제약 AUC). base 선택은 corr 행렬 출력 후 결정.

## 8. 단계
1. 대기 OOF 회수(exp_023 ✓, exp_025 CatBoost year-cat, LGBM-tuned) → corr 행렬 갱신.
2. base 풀 확정(corr 0.99↑ 중복 제거, RealMLP 포함).
3. `src/stack.py` 구현 + 3 메타러너 + 균등/rank-mean 대조.
4. §6로 판정 → 채택 시 제출(마일스톤) + ADR.
5. (이후 엔드게임) 스택이 천장이면 **새 모델군**(TabM 등) 1개 추가 후 재스택, 아니면 마감.

## 리스크
- 메타 가중의 OOF 과적합: 439k행·소수 피처·정규화/비음수 제약 → 위험 낮음(가중블렌드보다 안전). 그래도 §6 균등 대조 필수.
- exp_025/LGBM-tuned가 기존과 corr 0.99↑면 다양성 0 → 제외(메타 가중 0로도 자동 처리되나 명시 가지치기 권장).
