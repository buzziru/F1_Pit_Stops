# CatBoost — 모델별 SSOT (피처 전략 · 성능 개선 스택 분기 앵커 강화)

> 2026-06-05 · 이슈 [#11](https://github.com/buzziru/F1_Pit_Stops/issues/11) · 상태: **계획(보류 — TabM Phase2 후 착수)** · 관련 [[decisions]] #030(HP튜닝 park)·#025(i_* 동화)·#020(stint-cat 해로움)·#024(GBDT-FE) · 실행 [[lightning_jobs]]/[[kaggle_jobs]]

## 피처 전략

현 적용(exp_025, `base_yearcat`): Driver = **native**(내부 ordered-ctr), Year-cat.

| 현 적용 인코딩 | 분기 근거 | 후보/백로그 | 게이트 |
|---|---|---|---|
| Driver = **native(내부 ordered-ctr)** | CatBoost 강점축. TE(LGBM/XGB)·freq(XGB)와 다른 인코딩 → stack 최저 corr 앵커(↔RealMLP 0.969) | **Driver hash A/B**(hash64 native, Phase2) | 개별 유지~↑ AND corr 보존 |
| Year = **cat** | 저카디 native | — | — |
| (제외) i_* 상호작용 | 동화 확정(#025, GBDT 곱 단조불변) → **금지** | — | 기각 |
| (제외) freq-enc | CatBoost는 native ctr이 강점 → freq는 그 신호를 버려 개별↓ 위험(XGB는 ctr 없어 freq가 순이득이던 것과 다름) | — | 제외 |

- corr 참고(stack_v8): CatBoost↔RealMLP **0.969**(최저=다양성 앵커), ↔LGBM 0.977·↔XGB 0.976, GBDT끼리 0.98+(포화). **유일하게 GBDT 포화권 밖 = 다양성 가치 최대 멤버.**

## 목표
CatBoost(exp_025, 개별 OOF **0.950043**, 스택 logistic coef **0.104**)의 **스택 기여**를 높인다. CatBoost는 stack_v8 풀에서 **최저 corr 앵커**(↔RealMLP 0.969·↔LGBM 0.977·↔XGB 0.976 = 유일하게 GBDT 포화권 밖) = **다양성 가치가 가장 큰 멤버**. 목표 Private 0.95452(격차 +0.00057)엔 약한 멤버 강화가 한 축([[target-score]]).
- **전이 가설(포화 교훈, #029)**: RealMLP는 포화(corr 0.984+)라 개별↑ 전이 0(exp_056). **CatBoost는 분기 앵커(0.969)라 개별↑ 또는 corr↓가 스택에 전이될 여지가 RealMLP보다 크다** — 이것이 본 계획의 상방 근거. ([[tabm]] hash64 corr0.965와 동일 논리.)

## park 맥락 (#030 — "HP 튜닝은 천장 0")
- cat-tune(Optuna, lr·depth·l2·random_strength·subsample·max_ctr_complexity) → best **+0.000036**(14 trial 수렴) → **park**. **즉 탐색한 HP 공간은 소진**. 본 계획은 **HP 외 미탐색 레버**만 다룬다(HP 재튜닝이면 #030 park 유지).
- 고정/미탐색이던 노브: `grow_policy`(SymmetricTree 고정)·`bootstrap_type`(Bernoulli 고정)·ctr 설정(type/prior)·피처.

## 레버 후보 (ROI 순, 각 fold0 게이트)

### Phase 1 — 구조 노브 (미탐색) · 최고 ROI 후보
- **`grow_policy`**: SymmetricTree(default) → **Depthwise/Lossguide**. 표현력↑ 가능. ⚠️ **양날**: symmetric이 CatBoost의 LGBM/XGB(leaf-wise) 대비 **다양성 강점**(catboost.yaml 주석) → Depthwise면 개별↑하나 **corr↑(앵커 가치 상실)** 위험. 게이트=개별↑ **AND** corr↔LGBM 유지(<0.98).
- **`bootstrap_type`**: Bernoulli → **Bayesian**(`bagging_temperature` 튜닝). 정규화 다양성 — corr↓ 기대(개별 중립~↑).
- **ctr**: `one_hot_max_size`↑·`max_ctr_complexity`(cat-tune 1로 수렴) 재검토·ctr prior. native 범주 인코딩이 CatBoost 강점축이라 ctr 세밀화가 개별↑ 여지.

### Phase 2 — Driver 인코딩 분기 (hash A/B)
- exp_025는 Driver native(CatBoost 내부 ordered-ctr). i_* 추가는 **동화 확정**(#025, GBDT 곱 단조불변) → **금지**.
- **Driver hashing A/B** (TabM hash64 동류, 사용자 2026-06-05): native Driver(887) → **Driver_hash(N버킷) native**. high-card ctr을 더 robust하게(rare driver prior 의존↓) + 타깃 비의존이라 TE/ctr 멤버와 **분기 유지**. `add_driver_hash_features` 재사용(`_DRIVER_HASH_BUCKETS` 스크린: 64 등). **freq-enc는 제외** — CatBoost는 native ctr이 강점이라 freq는 그 신호를 버려 개별↓ 위험(XGB는 ctr 없어 freq가 순이득이었던 것과 다름).
- 게이트(fold0 A/B): hash vs exp_025 native — 개별 유지~↑ **AND** corr↔타 멤버 보존/↓(앵커 가치). high-card가 underperf 원인인지의 직접 검증.

### Phase 3 — full 5-fold + 스택 게이트
- best config full(seed 42 동결) → 4-member 스택 swap: **logistic > 0.954338(Δ≥+0.0001)이면 채택**, 아니면 park.

## 성공 기준
| Phase | 통과 |
|---|---|
| 1 | fold0 개별 > 0.9500 **AND** corr↔LGBM 유지(<0.98, 앵커 보존) |
| 2 | corr↓(↔RealMLP <0.969) 또는 개별↑, 동화(corr↑) 아님 |
| 3 | 스택 swap logistic > 0.954338 (Δ≥+0.0001) |

## 정직한 ROI·리스크 (과몰입 가드 — [[workflow-timeboxing]])
- **천장 추정 vs 격차(+0.00057)**: HP는 천장 0 확정(#030). 구조 노브(Phase1)는 미탐색이나 **여전히 "튜닝류"**라 천장이 HP와 비슷할 위험(개별 마진 + 포화). **천장 < 격차일 가능성 높음 → 단독 목표달성 레버 아님**, 스택 다양성 누적용 보조 성격. **주스레드 = TabM(NN 신축), CatBoost는 병렬 강등 권장.**
- **kill criterion**: Phase1 fold0 2-3 config에서 (개별↑ AND corr 보존) 동시충족 **0이면 종료**(patience 2). grow_policy가 corr↑만 유발하면(앵커 상실) 즉시 kill. 타임박스: fold0 스크린 1라운드(동시 GPU 2슬롯, ~30-50분) 내 판단.
- **⚠️ 결정 주체 = 사용자.** 어시스턴트는 결과 보고 + 기각/park 의견만. 임의 발사·중단 금지(GPU 발사 전 피처 confirm, [[confirm-features-before-gpu]]).
- **상방**: 분기 앵커가 개별↑/corr↓로 강화되면 스택 +α. 단 EV는 TabM(새 NN축) < CatBoost(기존축 강화) 순으로 낮음 — **CatBoost는 TabM 결과가 격차를 못 닫을 때의 누적 레버**로 자리매김.

## Sources
CatBoost docs(grow_policy/bootstrap_type/ctr) · 자체 실측 exp_025/029·cat-tune l4b/l4c(#030) · stack_v8 corr(앵커 0.969).
