# CatBoost — 모델별 SSOT (피처 전략 · 성능 개선 스택 분기 앵커 강화)

> 2026-06-05 · 이슈 [#11](https://github.com/buzziru/F1_Pit_Stops/issues/11) · 상태: **HP·hash park / ctr 방향(Driver 적절처리+max_ctr↑)은 유효 — 구현 개선 후 재시도 백로그**(exp_067은 구현문제, 사용자 견해). 현 멤버=exp_025 native 유지 · 관련 [[decisions]] #030·#025·#020·#024 · 실행 [[lightning_jobs]]/[[kaggle_jobs]]

## 피처 전략

현 적용(exp_025, `base_yearcat`): Driver = **native**(내부 ordered-ctr), Year-cat.

| 현 적용 인코딩 | 분기 근거 | 후보/백로그 | 게이트 |
|---|---|---|---|
| Driver = **native(내부 ordered-ctr)** | CatBoost 강점축. TE(LGBM/XGB)·freq(XGB)와 다른 인코딩 → stack 최저 corr 앵커(↔RealMLP 0.969) | **Driver=OOF-TE numeric + max_ctr_complexity=4**(Phase1, 조합서 분리). ~~hash~~(하향: hash 위 또 ctr→분기약) | 개별↑(vs 0.950043) AND corr 보존 |
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
- **ctr — Driver를 조합서 분리 + max_ctr_complexity↑** (사용자 가설 2026-06-05, ROI 최고): cat-tune이 `max_ctr_complexity=1`로 수렴(#030)한 게 **Driver(887)가 combinations_ctr에 들어가 887×Race(26)=2.3만+ 조합 → 극sparse 노이즈**라서일 가능성. Driver를 조합에서 빼면 저카디(Race 26·Compound 5·Year 4)끼리 의미있는 조합 ctr 가능.
  - ⚠️ **`per_feature_ctr`로는 불가** (검증 2026-06-05 + CatBoost 공식): per_feature_ctr은 개별 ctr만 설정, **조합 참여 제어 안 됨**. CatBoost엔 feature별 조합 제외 옵션 없음(max_ctr_complexity는 전역).
  - **유일 경로 = Driver numeric화** → `cat_features`에서 빠져 조합 ctr 자동 제외. 인코딩은 **OOF-TE**(신호 보존, CatBoost ctr 대체 numeric / freq는 약신호·hash는 순서무의미) 권장. 나머지 native + `max_ctr_complexity=4`.
  - **결과(exp_067, 2026-06-05)**: Driver=OOF-TE + mc4 fold0 **0.950029 < exp_025 fold0 0.951265 (Δ −0.001235)**.
  - ⚠️ **방향 기각 아님 — 구현 방법 문제로 판단 (사용자 견해 2026-06-05).** "Driver를 적절히 처리(조합 폭발 회피) + max_ctr_complexity↑"는 **유효한 방향**이고, exp_067의 미달은 다음 **구현 결함 후보** 때문:
    - ① **OOF-TE numeric화가 native ordered-ctr 신호를 약화** — Driver를 조합에서 빼는 유일 수단이 numeric화였으나(per_feature_ctr로 조합 제외 불가), 그게 Driver 강신호를 죽임. **이상적 구현 = Driver native(simple) ctr 유지 + 조합에서만 제외** (CatBoost가 직접 미지원 → 우회 방법 탐색 필요).
    - ② **task_type 불일치** — exp_067=**CPU**(로컬) vs exp_025=**GPU**. CatBoost CPU/GPU는 ctr 알고리즘이 달라 비교가 오염됐을 수 있음.
    - ③ **best_iter 4986/5000 cap 근접 = 미완 학습**(#017) → cap 상향 재학습 필요.
  - → **재시도 백로그(방향 유효)**: 동일 **GPU 환경** + **cap 상향** + Driver 처리 개선(native ctr 보존하며 조합 분리, 또는 TE smoothing/인코딩 강화)으로 재검증. 단독 천장은 낮으나(보조 레버) 방향은 맞음.
- **ctr default 발견 + 정규화 묶음 백로그 (2026-06-05)**: CatBoost default `simple_ctr` = `Borders`(prior 3개: 0/1·0.5/1·1/1) + `Counter`(prior 1개: 0/1), **`max_ctr_complexity` default=1**. → cat-tune의 1 수렴 = default(조합 무익 확정). **Driver native+mc1+Counter는 이미 default(신규성 없음)**. **실질 미탐색 백로그 = ctr 정규화 묶음**(exp_068 설계):
  - ① **Counter prior 다양화**: Counter는 default prior 1개뿐 → 여러 개(0.5/1, 1/1 추가).
  - ② **`model_size_reg`↑**(default 0.5): 고카디 ctr feature 선택 정규화.
  - ③ **`store_all_simple_ctr=True`**(default False): simple ctr 다양성↑.
  - ④ **`ctr_leaf_count_limit`**(default 무제한): 고카디 Driver(887) ctr leaf 폭발 제한 → 과적합↓(가장 직접적).
  - **결과(exp_068, 2026-06-06, 기각)**: GPU는 store_all_simple_ctr 미지원 → **CPU 통제 비교**(nbr15000·augment 동일). ctr 정규화 묶음 fold0 **0.951079 vs exp_025 CPU baseline 0.951129 (Δ −0.000050)** = 노이즈, **효과 없음 → 기각**. Counter prior·model_size_reg·store_all·leaf_limit 모두 마진 0(best_iter만 8222로 빨리 수렴). default가 이미 충분.
- **(기각) Driver hash · Driver-TE 조합분리 · driver_te numeric 추가**: hash=hash위 ctr 분기약 / TE분리=native ctr 손실(exp_067) / TE추가=중복. Driver native 유지가 결론.

### Phase 2 — Driver hash (하향, 평가 2026-06-05)
- **Driver hash A/B는 EV 낮음으로 하향**. 평가 결론: CatBoost는 native cat에 **ordered-ctr을 적용**하므로, `Driver_hash`를 native로 줘도 **hash 버킷 위에 또 ctr**(타깃 기반) → ① TabM의 "순수 embedding 분기"와 달리 **분기(corr↓) 약함** ② 887은 ctr이 이미 robust 처리(TabM native embedding의 sparse 문제 없음) → cardinality 축소 이득 없음 + 충돌 손실. rare driver 풀링은 잠재이득이나 빈번 driver 희석과 trade-off로 불확실.
- → Phase1 ctr(Driver numeric화 + max_ctr↑)이 high-card 대응의 더 나은 경로. hash는 park.
- i_* 추가는 **동화 확정**(#025) → 금지. freq-enc도 native ctr 강점 버려 비권장.

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
