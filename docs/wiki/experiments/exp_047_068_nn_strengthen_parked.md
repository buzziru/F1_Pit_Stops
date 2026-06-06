# 실험 회고 — 기존 4멤버 강화 전부 park (포화 확인) — exp_047~068

> 2026-06-05 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10)·[#11](https://github.com/buzziru/F1_Pit_Stops/issues/11)(튜닝) · 상태: **전부 park — 스택 전이 0~음**(stack_v8 OOF 0.954338 불변) · 관련 [[decisions]] #030(CatBoost 튜닝)·#031(TabM 7레버)·#032(RealMLP n_refit)·[[pytabkit_params]]·[[tabm]]·[[realmlp]]·[[catboost]]

stack_v8(OOF 0.954338) 고정 후, **기존 4멤버 각각을 개별 강화**해 스택을 밀어올리려는 트랙. 결론: **전부 park** — 개별은 오르나 멤버 간 corr 포화로 스택 전이 0. "MU(marginal utility)>0인데 천장 합<격차"의 전형(천장 게이트).

## 결과 요약 (스택 swap 게이트, vs stack_v8 0.954338)
| 멤버 | 레버 | exp | 개별 Δ | 스택 swap Δ | 결론 |
|------|------|-----|--------|------------|------|
| TabM | hash64+pwl+k64 (7레버) | exp_058~064 | 개별 0.951↑ | corr↑(동화) 게이트 동시충족 불가 | **park**(#031) |
| RealMLP | ep64/lr0.02 | exp_056 | +0.000381 | **−0.000008** | **park**(포화) |
| RealMLP | n_refit=1(64%→80%) | exp_065 | +0.000353 | **−0.000039** | **park**(#032) |
| TabM | val_fraction0.1+Stint수치 | exp_066 | — | — | **park**(동화) |
| CatBoost | HP 튜닝(cat-tune) | — | 천장 ≈0 | — | **park**(#030, 수동중단) |
| CatBoost | Driver OOF-TE(조합제외)+mc4 | exp_067 | 0.950029 | native ctr 손실 | **park** |
| CatBoost | ctr 정규화 묶음(CPU) | exp_068 | 0.951079 vs 0.951129 | −0.00005 | **park** |

## TabM hash/pwl 트랙 = 개별↑이지만 corr↑ (ADR #031, 7레버 소진)
- 7레버(hash·pwl·tabm_k·tabm-mini·val_fraction·Stint·cross) 소진. 곡선: exp_058(hash64) 0.948/corr0.965 → exp_061(+pwl) 0.9528/0.983 → exp_062(+k64) 0.9514/0.978 → exp_063(+tabm-mini) 0.9512/0.977.
- **개별↑ = corr↑ 가 한 곡선** → 게이트(개별 0.951+ & corr<0.97) **동시충족 불가**. TabM=PLR-MLP라 RealMLP와 구조적 수렴. exp_064(full) 사용자 중단. [[pytabkit_params]] 1.7.3 제약(periodic·n_refit 미지원)도 실증.

## RealMLP ep/lr·n_refit = 포화 멤버라 전이 0 (ADR #029 note·#032)
- **ep/lr(exp_056)**: 개별 +0.000381이나 스택 −0.000008. RealMLP↔LGBM/XGB corr **0.984~0.987 포화**.
- **n_refit=1(exp_065)**: 데이터 손실 64%→80% 해결이 **개별 레버로는 유효**(+0.000353, 가설 검증)나 corr↔exp_046 **0.9947(복제)** → 스택 −0.000039. 비용 큼(fold0 80분/full ~7h).
- **교훈(핵심)**: **포화 멤버(RealMLP)는 개별↑이 스택에 전이 안 됨**. 개별↑ 레버는 **비포화·고분기 멤버(TabM hash64 corr0.965·CatBoost 0.959)** 또는 **새 NN축**에만 가치. n_refit(데이터 손실 해결)은 비포화 신축 NN과 결합할 백로그.

## CatBoost = 전부 소진 (ADR #030)
- HP 튜닝 천장 ≈0(#030, 수동중단). Driver OOF-TE 분리(exp_067)는 native ctr 손실로 개별↓. ctr 정규화 묶음(exp_068, CPU 통제) −0.00005. → **exp_025 default 유지.** (Driver-TE 분리 GPU 재시도는 EV 매우 낮음·사실상 park.)

## 교훈
1. **포화 진단이 우선** — 멤버 간 corr 0.98+면 개별 강화는 스택 전이 0. swap Δ로 즉시 판정(seed-avg #028·n_ens #029·ep/lr·n_refit 전부 동형 −0~+0.00003).
2. **개별↑ 레버의 유효 조건 = 비포화/고분기 멤버** — 같은 +0.0003대 개별 이득이 RealMLP(포화)엔 0, 고분기 멤버엔 +. → 강화 대상 선택이 레버 자체보다 중요.
3. **기존 멤버 강화 트랙 종료** → 남은 격차는 **새 메커니즘 NN축**(decorrelated)으로만. → [[exp_069_071_nn_new_axis]] (FTT/TabICL).

## 산출물·참조
- OOF/log: `experiments/{oof,logs}/exp_056·065·067·068*`(TabM hash 트랙 일부는 Kaggle/Colab fold0 스크린, 표준 로그 미보존 — 수치는 [[pytabkit_params]]·[[decisions]]).
- 결정: [[decisions]] #029·#030·#031·#032 / 이전: [[exp_037_046_stackv7_track]] / 다음: [[exp_069_071_nn_new_axis]]
