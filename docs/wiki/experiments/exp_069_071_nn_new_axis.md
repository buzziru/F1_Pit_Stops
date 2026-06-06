# 실험 회고 — NN 신축 (FTT·TabICL) → TabICL 5번째 멤버 채택 (Private 0.95400)

> 2026-06-06 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **TabICL 채택**(stack_v9 Private **0.95400** 신기록) · **FTT 경계·미채택** · 관련 [[decisions]] #033·[[tabicl]]·[[ftt]]·[[pytabkit_params]]

기존 4멤버 강화 전부 소진([[exp_047_068_nn_strengthen_parked]]) → **다른 메커니즘 NN축**만 남은 주경로(#031). FTT(attention transformer)·TabICL(foundation model/ICL) 둘 다 fold0 corr 게이트로 분기 여부 1차 판정.

## 결과 요약 (동일 fold/seed=42)
| 모델 | exp | 개별 OOF | corr(↔RealMLP/CatB) | 결론 |
|------|-----|----------|---------------------|------|
| FTT_D fold0 | exp_069 | 0.9494 | 0.972 / **0.957** | **경계·미채택**(혼재 동화, 최약체) |
| TabICL cat.codes fold0 | exp_070 | 0.950616 | 0.9705 / 0.9712 | T4 OOM→L4 전환 |
| **TabICL raw full** | **exp_071** | **0.949358** | 0.9692 / 0.9701 | **채택**(5번째 멤버) |

| 스택 | 멤버 | meta-OOF(logistic) | Private | 비고 |
|------|------|------|------|------|
| stack_v8 | 034+043+046+025 (4) | 0.954338 | 미제출 | base |
| **stack_v9** | +**071 TabICL** (5) | **0.954357** | **0.95400** | +0.000019(OOF)/+0.00005(LB). **제출 신기록** |

## FTT (exp_069) — 경계, 미채택
- `FTT_D` default 무튜닝 + `realmlp_fe_v2`(피처 통제) + **val_fraction=0.1**(n_refit=0 데이터 손실 64%→72% 완화). Kaggle T4 ~17.7분, early-stop epoch 25 수렴.
- fold0 개별 **0.9494 = 최약체**. corr: CatBoost **0.957**(앵커 0.969보다 분기↑) / XGB 0.9695 / RealMLP **0.972** / LGBM **0.9711**. → **혼재**: CatBoost·XGB와는 분기, RealMLP·LGBM과는 ≥0.97 동화. 일관 분기 아님 + 개별 최약 → **full 미진행, 보류**.
- 판단: TabICL(개별 0.9506·유사 corr)이 +0.000019였는데, **더 약한 FTT가 full ~1.5h 들여 게이트 넘길 EV 낮음**. CatBoost 분기 0.957 한 가지에 베팅하는 셈. ([[ftt]])

## TabICL (exp_070→071) — 채택
- **메모리/GPU**: 추론 단일 GPU, 440k가 T4 16GB OOM(exp_070 DeadKernel) → **L4 Colab 전환**([[colab_jobs]]). full 5-fold ~2h(offload_mode=auto, batch_size=2).
- **범주형 raw 개선 = 무효(가설 기각)**: cat.codes(exp_070 fold0 0.950616) → 문자열 자동인코딩(exp_071 fold0 0.950613) **Δ 3e-6**. TabICL 내부 `TransformToNumerical`도 ordinal → 사전 cat.codes와 등가. "cat.codes가 Driver(887) 왜곡으로 개별↓" 가설 **기각**.
- **스택 게이트**: 5-member logistic 0.954357(+0.000019, Δ≥+0.0001 미달). corr 0.969~0.976 = 앵커 수준, 분기 미약. logistic coef 0.081.
- **그러나 LB 우호 → 채택**: 제출 **Private 0.95400 / Public 0.95349**. OOF +0.000019 → **Private +0.00005**(LB가 OOF보다 우호, exp_034·v7 패턴 재현). 다운사이드 0 drop-in. ([[decisions]] #033)

## 교훈
1. **메커니즘 차이가 corr↓를 보장하지 않는다(재확인)** — foundation model(TabICL)·attention(FTT) 둘 다 corr 0.97 전후로 GBDT/PLR-MLP와 부분 동화. #031 "NN 동화"가 신축 축에서도 반복.
2. **OOF 게이트 미달이라도 LB로 확정 가치** — TabICL OOF +0.000019(미달)가 Private +0.00005. **약한 멤버도 비복제면 LB+**, 단 천장 낮음(목표 격차 +0.00052를 단독 불가).
3. **개별 강도 vs 분기 트레이드오프** — FTT는 CatBoost 분기(0.957)는 더 좋으나 개별(0.9494) 최약. TabICL은 개별 약간↑·분기 평이. 둘 다 borderline = NN 신축 천장이 낮음을 시사.

## 산출물·참조
- OOF: `experiments/oof/exp_069·070·071*.csv` / 제출: `stack_v9_5mem_tabicl_logistic.csv`(Private 0.95400)
- 학습: `src/train_ftt.py`·`train_tabicl.py`·`stack.py` / 노트북 `kaggle/exp_069_ftt_baseline_fold0.ipynb`·`exp_071_tabicl_raw_full.ipynb`
- 결정: [[decisions]] #033 / 모델: [[ftt]]·[[tabicl]] / 이전: [[exp_047_068_nn_strengthen_parked]]
