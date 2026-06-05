# 실험 회고 — RealMLP v2 · GBDT-FE 트랙 → stack_v5/v6 (Private 0.95386)

> 2026-06-05 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10)(앙상블)·[#12](https://github.com/buzziru/F1_Pit_Stops/issues/12)(FE) · 상태: **채택 — 스택 신기록 Private 0.95386**(stack_v6) · 관련 [[decisions]] #021(RealMLP v2)·#022(GBDT-FE 개방)·#024(LGBM 결합 채택)·#025(XGB/Cat i_* park)·#023(divergence 게이트)

stack_v4(Private 0.95273) 이후 두 갈래로 도약: **① RealMLP 강화(배깅) ② GBDT에 곱 상호작용(i_*) 전이**. 동일 fold(seed=42) 유지.

## 결과 요약 (OOF AUC, 동일 fold/seed=42)
| 모델 | exp | 단독 OOF | vs 이전 | 비고 |
|------|-----|----------|---------|------|
| RealMLP v2 | exp_032 | **0.951978** | +0.0033 | ep64×n_ens15(배깅)+Stint_cat+arch. exp_024 대체(ADR #021) |
| LGBM A (base) | exp_033_A | 0.943936 | — | GBDT-FE A/B 대조군(상호작용 없음) |
| LGBM B (+i_*) | exp_033_B | 0.946674 | **+0.00274** | 곱 상호작용 5종. 게이트 9배 통과→트랙 개방(ADR #022) |
| LGBM 결합FE | exp_034 | **0.953818** | +0.00168 | i_*+year-cat+stint-cat+튜닝+aug. 스택 LGBM **채택**(ADR #024) |
| XGB +i_* | exp_035 | 0.953013 | +0.00175 | 개별 큼이나 스택 무용 → **park**(ADR #025) |
| CatBoost +i_* | exp_036 | 0.951882 | +0.00184 | 개별 큼이나 스택 무용 → **park**(ADR #025) |

| 스택 | 멤버 | meta-OOF(logistic) | Private | 비고 |
|------|------|------|------|------|
| stack_v5 | exp_030+028+025+**032** | 0.953504 | **0.95329** | exp_024→exp_032 스왑 +0.0007 |
| **stack_v6** | **034**+028+025+032 | **0.954204** | **0.95386** | exp_030→exp_034 스왑 +0.0007. **현 최고** |

## exp_032 — RealMLP v2 (ADR #021)
- 레시피: ep64 × **n_ens=15**(배깅, 핵심 레버) + **Stint_cat(5+)** + yekenot arch(hidden[512,256,128]·silu·plr_sigma2.33). 1단계 fold0 스크리닝(exp_031 +0.0013) 선검증 → 2단계 본run(Kaggle P100 ~60분).
- 개별 0.948773→**0.951978**(+0.0033). 스택 swap 게이트 통과(+0.000626). 단 강해지며 GBDT corr 0.90→0.95(다양성 일부↓)이나 강도가 압도해 순+.

## exp_033/034 — GBDT-FE 트랙 (ADR #022/#024)
- **#010 곱 공백 실증**: #010("GBDT 단조변환 불변")은 단일 피처에만 유효. 곱(`laptime×deg`)·비율은 **2-피처 상호작용**이라 트리가 axis-split로 근사만 함 = "트리가 못 뽑는 정보". LGBM A/B Δ+0.00274로 확증.
- **선행 버그픽스**: LGBM 경로(`src/train.py`)가 ADR #019 `feature_builder` 훅·`extra_categorical_cols` 미적용(train_common 분기 누락) → 수정(ADR #023, 노브 패리티 게이트 `scripts/check_knob_parity.py` 도입).
- **exp_034**(결합) 단독 **0.953818** 이 구 stack_v5(0.953504)를 **넘음** → 누수 아닌 실신호(전부 per-row/OOF/train-fold 한정, LB로 검증). stack_v6 = 0.954204 → **Private 0.95386 신기록**.

## exp_035/036 — i_* 의 XGB/CatBoost 전이 = 스택 무용 (ADR #025, park)
- 사용자 가설("개별 향상 > 상관 비용 → 스택 +") 검증. **개별은 크게 향상**(XGB +0.00175, CatBoost +0.00184) 이나 **스택 swap 게이트 전부 FAIL**: XGB 스왑 0.954210(Δ+0.000006)·CatBoost 0.954193(−0.000011)·둘 다 0.954177(−0.000027) vs stack_v6 0.954204.
- 원인: i_*가 XGB/Cat를 exp_034(LGBM+i_*)의 **거의 복제**로(corr 0.9864→**0.9951**) → 강도 이득이 다양성 손실로 상쇄. equal은 소폭↑이나 logistic(최적·제출 메타)은 0. **LOO 포화(#021) 재현.**

## 교훈
1. **강도 vs 다양성의 경계**: 개별 강화는 **decorrelated 축에서만 순이득**(RealMLP v2 ✓). **중복 GBDT 강화는 corr만↑ → 스택 무용**(exp_035/036 ✗). 동일 +0.0017 이득이 정반대 결과.
2. **단독 OOF가 스택을 넘을 수 있다**(exp_034) — 강한 멤버 등장 시 스택은 변동감소 위주로. 단일 의존도↑(coef 0.70) 주의.
3. **곱 상호작용은 #010 예외**(GBDT에 유효), 단 **스택엔 한 모델만 충분**(나머지엔 중복).
4. **divergence는 게이트로**(ADR #023) — 분리 경로 유지 시 노브 패리티 정적 검사 필수.

## 산출물·참조
- OOF: `experiments/oof/exp_032·033_gbdt_fe_{A,B}·034·035·036*.csv` / 제출: `stack_v5_{logistic,equal}.csv`·`stack_v6_{logistic,equal}.csv`
- 학습: `src/train.py`(LGBM)·`train_xgb`·`train_catboost`·`train_realmlp`·`stack.py` / conf `features: {lgbm,xgb,catboost}_combined·realmlp_fe_v2`
- 결정: [[decisions]] #021·#022·#024·#025·#023 / 이전: [[exp_023_030_realmlp_yearcat_tuning]] / 다음: TabM+floor/bin(`tabm_fe_floorbin`, decorrelated 축)
