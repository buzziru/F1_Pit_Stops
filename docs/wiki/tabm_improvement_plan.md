# 계획 — TabM 성능 개선 (NN 축 강화/교체)

> 2026-06-05 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **계획(보류 — cat-tune·ep/lr 후 착수)** · 관련 [[decisions]] #029(TabM park·원인)·#021(RealMLP v2)·#025(동화)·#017(인코딩 분기) · 실행 [[kaggle_jobs]]/[[lightning_jobs]]

## 목표
TabM을 **강하고(개별 ~0.953+) + RealMLP와 분기된(corr↓) NN 축**으로 만들어 stack_v8(logistic 0.954338)을 넘는다. 귀결 = **5번째 멤버 추가**(스택 Δ≥+0.0001) **또는 RealMLP 교체**(더 강한 NN 축). 대회 증거: **2위 솔루션이 TabM 중심**(`s903124/2nd-place-…-tabm`), TabM=본 대회 상위 모델.

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
- **게이트**: fold0 개별 ≥ exp_044(0.9510) **그리고** corr↓RealMLP(목표 **<0.97**, 현 0.9811). corr 하락폭이 "추가 vs 교체" 분기점.

### Phase 2 — 앙상블 + 튜닝 (강도)
- **tabm_k**(병렬 헤드 = TabM 시그니처 앙상블) default→**32+** (RealMLP n_ens 대응 강도 레버).
- **수치임베딩** 옵션(PLE/periodic — **2위 레버** `rtdl_num_embeddings`) vs default 비교.
- **lr / n_epochs / arch**(hidden·depth) fold0 스크린(RealMLP ep/lr 패턴 재사용).
- **게이트**: 개별 **~0.953+**(GBDT 경쟁선).

### Phase 3 — full 5-fold + 스택 결정
- best config full 5-fold(seed 42 동결).
- **스택 게이트 2가지**: (a) **5번째 추가** Δ≥+0.0001 (b) **RealMLP 교체**(TabM-대신-RealMLP 스택 vs 현재) → **높은 쪽 채택**.

## 성공 기준
| Phase | 통과 |
|---|---|
| 1 | corr↓RealMLP(<0.97) + 개별 ≥0.951 |
| 2 | 개별 ~0.953+ |
| 3 | 스택(추가 or 교체) > 0.954338 |

## 정직한 ROI·리스크 (과몰입 가드)
- **시퀀싱**: cat-tune·ep/lr **이후 착수**(#029 보류). 그것들이 목표 격차를 닫으면 우선순위↓.
- ⚠️ **둘 다 PLR-NN** → 튜닝 후도 corr 높을 위험 → **"추가"보다 "교체"가 현실적**(Phase 1 corr 하락폭이 판단점).
- **상방**: 대회급 TabM은 **최강 단일 NN** 가능 → NN 축 자체 업그레이드(단순 +1 아님). 목표(Private 0.95452, 격차 +0.00057)엔 새 축이 필요 — GBDT 슬롯 포화(#021)·기존멤버 강화 한계(#028) 고려 시 **NN 축 확장이 천장 돌파 주 경로**.
- **인프라**: Kaggle T4(torch, [[kaggle_jobs]]) 또는 Lightning L4. fold0 ~15–25분, full ~60–100분. 동시 실행 가능.

## Sources
2위 TabM 노트북(`s903124/2nd-place-…-tabm`) · pytabkit TabM/TabM_D(tabm_k 병렬헤드) · rtdl-num-embeddings(PLE/periodic) · 자체 실측 exp_037/038/044/045.
