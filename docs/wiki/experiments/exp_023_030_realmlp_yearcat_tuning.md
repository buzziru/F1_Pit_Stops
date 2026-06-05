# 실험 회고 — non-GBDT(RealMLP) 도입 · year/stint-cat · LGBM 튜닝 → stack_v4

> 2026-06-04 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10)(앙상블)·[#11](https://github.com/buzziru/F1_Pit_Stops/issues/11)(튜닝)·[#12](https://github.com/buzziru/F1_Pit_Stops/issues/12)(RealMLP FE) · 상태: **채택 — M4 스태킹 신기록 Private 0.95273** · 관련 [[decisions]] #018(RealMLP 도입)·#019(RealMLP FE 분기)·#020(스태킹 채택)·#013개정(튜닝 선행)

3-GBDT 블렌드(exp_022, Private 0.95165)가 corr 한계(0.985~0.994)로 +0.0002대에 갇히자, **① 비-GBDT 모델군(RealMLP) ② 범주형 표현 레버(year/stint-cat) ③ LGBM 튜닝**을 더해 4-모델 **스태킹**으로 넘어간 라운드. 동일 fold(seed=42)·외부증강(ADR #011) 고정.

## 결과 요약 (OOF AUC, 동일 fold/seed=42)
| 모델 | exp | 단독 OOF | vs 기준 | 비고 |
|------|-----|----------|---------|------|
| RealMLP baseline | exp_023 | 0.944533 | — | 공유피처(driver_te). 비-GBDT 축 corr 측정용 |
| RealMLP FE+year-cat | exp_024 | **0.948773** | +0.0046 | 상호작용5+cross2 TE+Year-cat. 스택 최대 기여(ADR #019). Kaggle P100 3h45m |
| CatBoost year-cat | exp_025 | **0.950043** | +0.00023 | vs exp_022(native, Year수치). Year범주형 이득. **채택** |
| LGBM Optuna | exp_026 | 0.951738 | +0.0008 | tuned HP(`src/tune_lgbm.py`). fold3 cap5000 미수렴 |
| LGBM tuned recap | exp_027 | 0.951744 | — | cap8000 재학습(수렴 확인) |
| XGB year+stint-cat | exp_028 | **0.951261** | +0.00017 | year+stint-cat 둘 다 +. **채택** |
| CatBoost year+stint | exp_029 | 0.949932 | −0.00011 | stint-cat이 CatBoost엔 **해로움 → 기각**(exp_025 유지) |
| LGBM tuned v2 | exp_030 | **0.952132** | +0.0012 | best params recap cap12000(수렴). 스택 LGBM 멤버 **채택** |

## 도약 동력
- **RealMLP FE(exp_024)**: non-GBDT라 ADR #010(GBDT 단조불변) 비적용 → FE 재개방(ADR #019). 0.944→0.948773(+0.0046), 스택 logistic 가중 0.06→0.26으로 최대 기여.
- **LGBM Optuna(exp_026→030)**: M5(튜닝) 선행 허용(ADR #013개정). 0.950959→**0.952132**(+0.0012). study-level no-improvement stop(`--patience`) 추가.
- **year/stint-cat(모델별 상이)**: Year-cat = 전 모델 + (Cat +0.00023·XGB +0.00017·RealMLP fold0 +0.00084). **Stint-cat = XGB만 + / CatBoost − / LGBM 미측정**(당시 LGBM 경로 `extra_categorical_cols` 버그, 후일 ADR #023). → "전 GBDT 대칭" 불성립, `extra_categorical_cols` 노브로 분기.

## 판정 (ADR #020 — 스태킹)
- **stack_v4** = LGBM-tuned(exp_030) + XGB ys-cat(exp_028) + CatBoost y-cat(exp_025) + RealMLP FE+yc(exp_024). 메타러너 nnls·logistic·rank·균등 비교, 4 멤버 다 강해 logistic≈균등(meta-OOF 0.9529).
- **🏁 제출**: stack_v4 **균등 Private 0.95273 / Public 0.95203**(logistic 0.95271/0.95210). 3-way(0.95165) 대비 **+0.00108**. OOF≈Private 갭 +0.00013(#006). 균등 미세 우위 → 균등 권장.

## 교훈
1. **non-GBDT(RealMLP)가 진짜 도약**: GBDT corr 천장을 넘는 유일 축. FE 분기(#019)는 모델 메커니즘 차이(MLP≠GBDT)에 근거.
2. **튜닝은 앙상블 구성 후가 정석**이나 LGBM은 선행 이득 확인(+0.0012, ADR #013개정).
3. **범주형 표현 레버는 모델별 비대칭** — Year-cat 보편적 +, Stint-cat은 XGB만. 일괄 적용 금지.
4. **best_iter 검수 필수**: exp_026 fold3가 cap5000 미수렴 → cap 상향 재학습(#017 원칙 재확인).

## 산출물·참조
- OOF: `experiments/oof/exp_023·024_rmlp_fe_yc·025·026·027·028·029·030*.csv` / 제출: `stack_v4_{equal,logistic}.csv`
- 학습: `src/train.py`(LGBM)·`src/train_xgb.py`·`src/train_catboost.py`·`src/train_realmlp.py`·`src/tune_lgbm.py`·`src/stack.py`
- 결정: [[decisions]] #018·#019·#020 / 계획: [[realmlp]]·`stacking_plan.md` / 다음: [[exp_032_036_realmlp_v2_gbdt_fe]]
