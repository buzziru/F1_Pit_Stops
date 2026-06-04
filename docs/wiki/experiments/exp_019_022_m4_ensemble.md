# 실험 회고 — M4 앙상블 다양성 (exp_019 XGBoost · exp_020~022 CatBoost · 블렌드)

> 2026-06-04 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **채택 — 3-way 블렌드 제출 신기록** · 관련 [[decisions]] #017(채택 근거)·#015(다양성 레버)·#016(seed)·#013(튜닝 연기)

LGBM 단일 모델 FE 공간 소진(ADR #014) 후, **상관 낮은 모델군 추가**로 블렌드 이득을 노린 M4 라운드. 동일 fold(seed=42)·driver_te·외부증강(ADR #011)을 **그대로 고정**하고 모델만 교체(ADR #015 — 신규 FE 금지, 표현·알고리즘 분기만).

## 결과 요약 (OOF AUC, 동일 fold/seed=42)
| 모델 | exp | 단독 OOF | vs LGBM | LGBM corr | XGB corr | 비고 |
|------|-----|----------|---------|-----------|----------|------|
| LGBM(베스트) | exp_016 | **0.950959** | — | — | 0.9944 | driver_te+aug 기준선 |
| XGBoost | exp_019 | **0.951095** | +0.00014 | 0.9944 | — | 단독 최고, `src/train_xgb.py` |
| CatBoost(OOF TE) | exp_020 | 0.949349 | −0.00161 | — | — | 대조군(기각) |
| CatBoost(native, 5k) | exp_021 | 0.949376 | −0.00158 | 0.9871 | 0.9872 | cap 미발화(미완) |
| CatBoost(native, 15k) | exp_022 | **0.949820** | −0.00114 | 0.9854 | 0.9858 | 수렴·채택 |

| 블렌드(균등) | 구성 | OOF | 비고 |
|------|------|-----|------|
| 2-way | exp_016+019 | 0.951402 | LGBM+XGB(미제출) |
| 3-way | exp_016+019+**021** | 0.951507 | native CatBoost |
| 3-way | exp_016+019+**022** | **0.951642** | 최종, 제출 |

**🏁 제출(균등 1/3, exp_016+019+022)**: Public **0.95084** / Private **0.95165** (vs exp_016 단독 Public 0.95065/Private 0.95139, Δ+0.00019/+0.00026). **OOF 0.951642 ≈ Private 0.95165**(갭 +0.00001).

## exp_019 — XGBoost (채택)
- **설정**: `src/train_xgb.py`(exp_016 파이프라인 미러, 모델만 교체). `tree_method=hist`, `grow_policy=lossguide`, `max_leaves=63`, `lr=0.05`, `subsample/colsample=0.8`, `min_child_weight=5`, `reg_lambda=1.0`, early-stopping. CPU 31.5분/run.
- **⚠️ XGB native categorical 정렬**: train/valid/test/증강의 category dtype 카테고리 집합이 일치해야 코드가 안 어긋나므로 Compound/Race 를 **고정 `CategoricalDtype`** 으로 정렬(`train_xgb.py` 주석).
- **결과**: 단독 OOF **0.951095**(>LGBM 0.950959) — fold 5/5 안정(0.9503~0.9521). LGBM 과 corr **0.9944**(매우 높음, leaf-wise GBDT 동질). 단독은 이겼으나 블렌드 이득은 corr 한계로 +0.0004대.

## exp_020~022 — CatBoost (ADR #017, 채택=exp_022)
- **Driver 표현 분기(ADR #015 레버1)**: LGBM/XGB 는 OOF TE(float), **CatBoost 는 native ordered target statistics**(`features=base`). 같은 피처를 *다른 표현*으로 주입 → 구조적 decorrelation.
- **TE vs native ablation**: 단독은 동률(TE 0.949349 ≈ native 0.949376)이나, **native 가 corr 더 낮음**(LGBM 0.9856/XGB 0.9859 < TE 0.9871/0.9872) → 블렌드 우위 → **native 채택, exp_020(TE)은 대조군 보존**.
- **best_iter cap 정상화(exp_021→022)**: exp_021(native, `iterations=5000`)은 fold별 best_iter 가 **5000 cap 에 붙어 early-stopping 미발화 = 미완 학습**. `iterations=15000` 으로 상향한 **exp_022 가 best_iter 6961~9377(cap 미발화=수렴)** → 단독 0.949376→**0.949820**(Δ+0.00044), **corr 거의 불변**(다양성 손실 없이 단독·블렌드 동시 상승). → CatBoost 최종 = **exp_022**.
- **best_iter 로깅 원칙 신설**: 이 발견으로 early-stopping 모델은 fold별 best_iter 를 반드시 기록·검수(cap 에 붙으면 미완 신호) — CLAUDE.md + `utils.log_experiment(best_iters=...)` + 3 train.py 반영.
- CatBoost 는 GPU(T4, Kaggle 아닌 별도 환경) ~30분. 단독은 가장 약하나(symmetric tree) corr 가 가장 낮아 블렌드를 견인.

## 판정 (ADR #015 기준 — 단독 아닌 블렌드)
- 다양성 변경은 **블렌드 OOF + corr** 로 판정(단독 손해여도 블렌드가 이기면 채택). XGB·CatBoost 모두 단독 약세/혼재였으나 3-way 블렌드가 단독 최고(exp_016/019)를 넘어 **채택**.
- **균등가중 우선**: 최적가중(w_cat≈0.28, OOF 0.951655)은 OOF 과적합 소지 → 참고용. 제출은 균등 1/3.

## 교훈
1. **GBDT 3종은 본질적으로 상관 높다**(LGBM↔XGB **0.9944**). 표현·알고리즘 분기로 corr 를 0.985까지 낮춰도 블렌드 절대이득은 **+0.0002대 LB** 에 그침 → 큰 도약은 **모델군 추가(non-GBDT)** 필요. → exp_023 RealMLP(ADR #018, `realmlp_kaggle_plan.md`)로 이어짐.
2. **CatBoost 는 native ordered TS 가 외부 OOF TE 보다 우월**(다양성·정확도 모두). 인코딩 분기는 비용 0의 다양성 레버.
3. **early-stopping cap 미발화 = 미완 학습** — best_iter 검수 필수(exp_021 뒤늦게 발견).
4. **OOF≈Private 재확인**(갭 +0.00001) — 앙상블에서도 OOF 1차 신뢰 유지(ADR #006).

## 산출물·참조
- OOF: `experiments/oof/exp_016·019·020·021·022.csv` / 제출: `experiments/submissions/blend_3way_eq.csv`
- 학습 경로: `src/train.py`(LGBM) · `src/train_xgb.py` · `src/train_catboost.py` / conf `model: lgbm|xgb|catboost`
- 결정 근거: [[decisions]] #017. 다음 라운드(non-GBDT): `realmlp_kaggle_plan.md`
