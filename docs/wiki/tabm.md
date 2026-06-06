# TabM — 모델별 SSOT (피처 전략 · 성능 개선 5번째 멤버 추가)

> 2026-06-05 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **park 확정** ([[decisions]] #031 — 정식 개선 7레버 소진, corr<0.97 불가) · 관련 [[decisions]] #029·#031·#021·#025·#017 · 실행 [[kaggle_jobs]]/[[lightning_jobs]]

## park 결론 (#031, 2026-06-05)
**7레버 소진해도 개별 0.951↔corr 0.977~0.983 고정** → corr<0.97 & 개별 0.951 동시충족 불가. progression: exp_058(0.948/0.965)→exp_061 pwl(0.953/0.983)→exp_062 k64(0.951/0.978)→exp_063 tabm-mini(0.951/0.977)→exp_066 vf0.1+stint수치형+cross제거(0.951/0.977). **cross 제거가 corr 못 낮춤 = "NN 강한 numeric 표현 수렴"이 corr 원인(구조적)**, RealMLP와 둘 다 PLR-MLP라 동화. 데이터손실(val_fraction)·n_refit은 TabM 미지원/효과미미([[pytabkit_params]]).

## 다음 NN 축 후보 — 다른 메커니즘 (RealMLP/TabM 동화 돌파)
PLR-MLP 계열(RealMLP·TabM)은 서로 동화 → **메커니즘이 근본적으로 다른 NN**으로 분기 재시도:
| 모델 | 메커니즘 | 분기 기대 | fold0 비용(T4) |
|---|---|---|---|
| **TabR**(`TabR_S_D`) | retrieval/instance-based | ★★★ (MLP/tree와 예측 패턴 최대 차이) | ~30-80분+ (retrieval 무거움) |
| **FTT**(`FTT_D`) | attention | ★★ | ~15-30분 |
| Resnet/MLP_RTDL | MLP 계열 | ★ (동화 위험, 비권장) | ~10-20분 |
- 게이트: **fold0 corr<0.97 먼저**(개별 수렴 전, 동화면 즉시 kill). 비용 싼 FTT부터 권장, 분기 확인 시 TabR 확장.

## 피처 전략

현 적용(exp_064, `tabm_fe_driverhash`): Driver = **hash64(native)**, cross native, +i_*, Stint_cat native, `num_emb_type=pwl`(이 버전 TabM은 pwl만 지원).

| 현 적용 인코딩 | 분기 근거 | 후보/백로그 | 게이트 |
|---|---|---|---|
| Driver = **hash64(native)** | high-card를 robust 버킷으로(rare driver 의존↓), 타깃 비의존이라 TE/ctr 멤버와 분기 유지(corr↓). exp_055 full-native corr 0.9407 | Driver native(full) vs Driver TE+cross-native(exp_045형) 재비교 | corr↓RealMLP(<0.97) |
| cross(Race_Compound·Race_Year) = **native 임베딩** | TabM 자기 임베딩 학습 → RealMLP(cross-TE)와 분기 | — | corr |
| **i_*(상호작용 5종)** | MLP 저층 직접활용(realmlp_fe와 공유) | — | — |
| Stint_cat = **native 범주** | 저카디 native | **Stint 수치형 입력 A/B** (아래 참조) | 개별↑·corr↓ |
| num_emb_type = **pwl** | 이 버전 TabM은 pwl만 지원(periodic/PLE는 rtdl_num_embeddings) | PLE/periodic 수치임베딩(2위 레버) vs default 비교 | 개별 |
| (제외) floor/quantile 비닝 | exp_037 −0.00114 유해 확정, PLR 수치임베딩과 중복·손실 | — | 기각(실측) |

**후보/백로그 — Stint 수치형 입력 A/B**: 현재 Stint_cat(min(Stint,5)) 범주형으로 입력하나, Stint는 ordinal(순서형)이고 NN num_emb(RealMLP=pbld, TabM=pwl)이 수치형을 분위수 구간 임베딩으로 처리하므로 순서 보존+비선형 표현 가능. extra_categorical_cols에서 Stint_cat 제거→Stint(raw) 수치형 유지. 개별↑·미세 corr↓ 기대(마진 레버). RealMLP/TabM 공통 후보.

- corr 참고(stack_v8): CatBoost↔RealMLP 0.969(최저=다양성 앵커), GBDT끼리 0.98+(포화). TabM hash64 corr 0.965(CatBoost와 동류 분기).

## 목표
TabM을 **RealMLP와 분기된(corr↓) + 충분히 강한 NN 축**으로 만들어 **stack_v8(4-member, logistic 0.954338)에 5번째 멤버로 추가**(스택 Δ≥+0.0001). **[[realmlp]] 교체 아님 — 추가**(사용자 확정 2026-06-05). 대회 증거: **2위 솔루션이 TabM 중심**(`s903124/2nd-place-…-tabm`).
- **추가의 필요조건 = 분기(corr↓)**: 5번째가 기존(특히 RealMLP)과 중복이면 동화로 기여 0(#025/#029, exp_044 실패). 따라서 **corr 하락이 1차 목표**, 개별 강도는 그 위에서 기여 가능선 확보. (교체와 달리 "RealMLP보다 강함"은 불필요 — "다르면서 쓸만함"이 조건.)

## 실패 원인 (#029 — "약해서"가 아니라 "방치해서")
- exp_044(no-bins full) 개별 **0.9508**(튜닝 RealMLP 0.9524보다 낮음), corr↔RealMLP **0.9811** → 약함+중복 → 5번째 게이트 실패·park.
- 근본: ① **default 무튜닝**(tabm_k/lr/n_epochs/arch 전부 default) ② **RealMLP 피처(realmlp_fe_v2, TE-float) 차용** → 자기 표현 못 배우고 RealMLP 복제. RealMLP엔 yekenot arch+n_ens24+ep/lr 쏟고 TabM은 vanilla = 비대칭 투자.

## 3단계 계획 (각 fold0 게이트, ROI 순)

### Phase 1 — TabM-native 피처 (분기 + 기본 강화) · 최고 레버
새 `conf/features/tabm_fe.yaml`:
- `feature_builder: add_realmlp_features` (i_* 5종 유지 — MLP 저층 직접활용)
- **`target_encode_cols: []`** → 범주 전부 **native 임베딩**(Driver·Compound·Race·Year·Race_Compound·Race_Year·Stint_cat). TabM이 자기 임베딩 학습.
- raw 수치 유지(PLR/수치임베딩이 처리), **bins 없음**(exp_037 −0.00114 유해 확정).
- **Driver 인코딩 변형 비교(fold0)**: (a) Driver native(full) vs (b) Driver TE + 나머지 native(exp_045형). 고카디 Driver native가 NN에 나은지 실측.
- **게이트**: **corr↓RealMLP(<0.97)가 1차 조건**(추가=새 축 필수) + 개별이 기여 가능선. corr 충분히 안 떨어지면 = 추가 불가 → 진행 보류 의견.
- **참고(이미 실측)**: full-native(exp_055) **corr 0.9407**(분기 강함, 추가에 유리) but 개별 0.9436(Driver native가 −0.0073). Driver-TE+cross-native(exp_045) corr 0.9741(분기 약함). → **분기는 full-native 방향이 유효, 과제는 개별 회복**(Phase 2).

### Phase 2 — 앙상블 + 튜닝 (강도)
- **tabm_k**(병렬 헤드 = TabM 시그니처 앙상블) default→**32+** (RealMLP n_ens 대응 강도 레버).
- **수치임베딩** 옵션(PLE/periodic — **2위 레버** `rtdl_num_embeddings`) vs default 비교.
- **lr / n_epochs / arch**(hidden·depth) fold0 스크린(RealMLP ep/lr 패턴 재사용).
- **게이트**: 개별 **~0.953+**(GBDT 경쟁선).

### Phase 3 — full 5-fold + 스택 추가 게이트
- best config(분기된 native + 튜닝) full 5-fold(seed 42 동결).
- **스택 게이트 = 5번째 멤버 추가**: 5-member 스택 vs 현 4-member(0.954338), **Δ≥+0.0001이면 채택**. corr↓ 확인 필수(동화면 추가해도 0). 교체 비교 없음.

## 성공 기준
| Phase | 통과 |
|---|---|
| 1 | corr↓RealMLP(<0.97) — 추가 필요조건 |
| 2 | 분기 유지하며 개별 회복(기여 가능선) |
| 3 | 5-member 스택 > 0.954338 (Δ≥+0.0001) |

## 정직한 ROI·리스크 (과몰입 가드)
- **시퀀싱**: cat-tune·ep/lr **이후 착수**(#029 보류). 그것들이 목표 격차를 닫으면 우선순위↓.
- ⚠️ **둘 다 PLR-NN이라 corr↓가 추가의 핵심 난점.** 분기는 native 인코딩(exp_055 corr 0.94)으로 가능하나 **개별 회복(Driver native −0.007)이 Phase 2 과제**. 튜닝으로 분기+강도 동시 확보 실패 시 → TabM 추가 불가, **park 의견**(결정은 사용자).
- **상방**: 분기된 강한 TabM = **새 decorrelated NN 축 1개 추가**(스택 풀 4→5, NN 2축). 목표(Private 0.95452, 격차 +0.00057)엔 새 축 필요 — GBDT 슬롯 포화(#021)·기존멤버 강화 한계(#028) 고려 시 **NN 축 추가가 천장 돌파 주 경로**.
- **인프라**: Kaggle T4(torch, [[kaggle_jobs]]) 또는 Lightning L4. fold0 ~15–25분, full ~60–100분. 동시 실행 가능.

## Sources
2위 TabM 노트북(`s903124/2nd-place-…-tabm`) · pytabkit TabM/TabM_D(tabm_k 병렬헤드) · rtdl-num-embeddings(PLE/periodic) · 자체 실측 exp_037/038/044/045.
