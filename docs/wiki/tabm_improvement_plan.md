# 계획 — TabM 성능 개선 (5번째 멤버 추가)

> 2026-06-05 · 이슈 [#10](https://github.com/buzziru/F1_Pit_Stops/issues/10) · 상태: **계획(보류 — cat-tune·ep/lr 후 착수)** · 관련 [[decisions]] #029(TabM park·원인)·#021(RealMLP v2)·#025(동화)·#017(인코딩 분기) · 실행 [[kaggle_jobs]]/[[lightning_jobs]]

## 목표
TabM을 **RealMLP와 분기된(corr↓) + 충분히 강한 NN 축**으로 만들어 **stack_v8(4-member, logistic 0.954338)에 5번째 멤버로 추가**(스택 Δ≥+0.0001). **RealMLP 교체 아님 — 추가**(사용자 확정 2026-06-05). 대회 증거: **2위 솔루션이 TabM 중심**(`s903124/2nd-place-…-tabm`).
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
