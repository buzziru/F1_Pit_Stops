# Project Wiki — S6E5

작업 지식을 **오래 남는 형태**로 기록하는 공간. (일회성 할 일/진행상황은 GitHub Issues, 상시 가이드는 루트 `CLAUDE.md`.)

## 문서 역할 구분
| 위치 | 역할 | 수명 |
|---|---|---|
| `CLAUDE.md` | 프로젝트 상시 가이드 (규칙·구조·실행법) | 항상 최신 |
| GitHub **Issues** | 실행 단위 (task / experiment / bug) | 열림→닫힘 |
| `docs/wiki/` | 결정·발견·회고 등 **지식 베이스** | 영속 |
| `docs/{eda,feature_engineering,modeling}.md` | 영역별 살아있는 작업 노트 | 갱신형 |

## 위키 구성
- [`decisions.md`](decisions.md) — 의사결정 기록 (ADR-lite). 왜 그렇게 정했는지.
- `experiments/` — 주요 실험 회고 (가설 → 결과 → 결론). exp 단위로 추가.
  - [`exp_002_003_is_stable_delta.md`](experiments/exp_002_003_is_stable_delta.md) — is_stable_delta ablation (기각): 낮은 corr≠무용, 이진화 정보손실
  - [`exp_004_driver_te.md`](experiments/exp_004_driver_te.md) — Driver OOF 타깃 인코딩 (채택, +0.00559): 고카디널리티는 OOF TE > native cat
  - [`exp_008_011_group1_fe.md`](experiments/exp_008_011_group1_fe.md) — 1번 그룹 파생피처 (기각·revert): #010 법칙 — 트리가 raw 에서 뽑는 정보는 무용
  - [`exp_012_016_external_aug.md`](experiments/exp_012_016_external_aug.md) — 외부 원본 데이터 train 증강 (채택·신기록): fold train 에만 추가, 검증은 대회 only
  - [`exp_019_022_m4_ensemble.md`](experiments/exp_019_022_m4_ensemble.md) — M4 앙상블 XGB/CatBoost/3-way 블렌드 (채택·제출 신기록): GBDT 동질(corr 0.99)→non-GBDT 필요
  - [`exp_023_030_realmlp_yearcat_tuning.md`](experiments/exp_023_030_realmlp_yearcat_tuning.md) — RealMLP 도입·year/stint-cat·LGBM 튜닝 → stack_v4 (채택, Private 0.95273): non-GBDT가 도약, 범주레버는 모델별 비대칭
  - [`exp_032_036_realmlp_v2_gbdt_fe.md`](experiments/exp_032_036_realmlp_v2_gbdt_fe.md) — RealMLP v2·GBDT-FE 트랙 → stack_v5/v6 (채택, Private 0.95386): 강도는 decorrelated 축에서만 순+(곱 i_*는 GBDT 1종만 유효)
  - [`exp_037_046_stackv7_track.md`](experiments/exp_037_046_stackv7_track.md) — TabM bins 스크린·RealMLP n_ens24·XGB freq-enc → stack_v7/v8 (채택, Private 0.95395): drop-in 강화는 게이트 미달도 채택, 죽은멤버 인코딩분기로 부활
  - [`exp_047_068_nn_strengthen_parked.md`](experiments/exp_047_068_nn_strengthen_parked.md) — 기존 4멤버 강화 전부 park (포화): 개별↑ 레버는 비포화·고분기 멤버에만 전이, RealMLP/TabM/CatBoost 소진
  - [`exp_069_071_nn_new_axis.md`](experiments/exp_069_071_nn_new_axis.md) — NN 신축 FTT·TabICL → TabICL 5번째 채택 (Private 0.95400): 메커니즘 차이도 corr↓ 불보장, OOF 미달도 LB로 확정
- `domain/` — F1 피트전략 등 도메인 지식 메모.

## 작성 규칙
- 한 문서 = 한 주제. 제목에 날짜/이슈번호를 남긴다 (예: `exp_001 baseline (#3)`).
- 결론은 **수치 + 근거** 중심 (토큰 절약 원칙과 동일).
- 관련 Issue/PR/로그 경로를 상호 링크한다.
